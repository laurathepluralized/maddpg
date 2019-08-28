#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""MADDPG algorithm.

This code is a combination of the original OpenAI MADDPG code, my own
modifications, and modifications made in
https://github.com/sunshineclt/maddpg/blob/master/maddpg/trainer/maddpg.py
and https://github.com/jarbus/maddpg/blob/master/maddpg/trainer/maddpg.py,
all of which are under the MIT license.
"""
import random
import numpy as np
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer
try:
    import lvdb as pdb  # noqa
except ImportError:
    import ipdb as pdb
import pysnooper

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name),
                               sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(
            polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer,
            grad_norm_clipping=None, local_q_func=False, num_units=64,
            scope="trainer", reuse=None):
    """Train the policy."""
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder(
                    [None],
                    name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]
        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]),
                   scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()

        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True,
                   num_units=num_units)[:, 0]
        pg_loss = -tf.reduce_mean(q)
        p_loss_summary = tf.summary.scalar('p_loss', pg_loss)
        p_cov_summary = tf.summary.scalar('p_cov', tf.reduce_mean(tf.square(act_pd.std)))

        loss = pg_loss + p_reg * 1e-3

        optimize_expr, hist = U.minimize_and_clip(optimizer, loss, p_func_vars,
                                                  grad_norm_clipping,
                                                  histogram_name='p_gradient')

        p_loss_summary_merge = tf.summary.merge([p_loss_summary, p_cov_summary, hist])

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n,
                           outputs=[loss, p_loss_summary_merge],
                           updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]],
                              [act_pd.mean, act_pd.logstd])

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]),
                          scope="target_p_func", num_units=num_units)
        target_p_func_vars = \
            U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]],
                                outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values,
                                             'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer,
            grad_norm_clipping=None, local_q_func=False, scope="trainer",
            reuse=None, num_units=64):
    """Train the critic."""
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder(
            [None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:, 0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        # q_loss = tf.reduce_mean(tf.square(q - target_ph))
        q_loss = tf.reduce_mean(U.huber_loss(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an
        # initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss  # + 1e-3 * q_reg
        q_loss_summary = tf.summary.scalar('q_loss', loss)

        optimize_expr, hist = U.minimize_and_clip(optimizer, loss,
                                                  q_func_vars,
                                                  grad_norm_clipping,
                                                  histogram_name='q_gradient')

        q_train_summary_merge = tf.summary.merge([q_loss_summary, hist])

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph],
                           outputs=[loss, q_train_summary_merge],
                           updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func",
                          num_units=num_units)[:, 0]
        target_q_func_vars = \
            U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values,
                                        'target_q_values': target_q_values}

class MADDPGAgentTrainer():
    """Train MADDPG Agent.

    The vast majority of the modifications to this class (as well as other
    parts of this file) are drawn from
    https://github.com/sunshineclt/maddpg/blob/master/maddpg/trainer/maddpg.py.
    """
    def __init__(self, name, model_value, model_policy, obs_shape_n,
                 act_space_n, agent_index, args, summary_writer,
                 local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i],
                            name="observation" + str(i)).get())

        # Create all the functions necessary to train the model

        # train critic
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model_value,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )

        # train policy
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model_policy,
            q_func=model_value,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        # self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.max_replay_buffer_len = 100
        self.replay_sample_index = None
        self.summary_writer = summary_writer

    def action(self, obs):
        # return self.act(obs[None])[0]
        theac = self.act(obs[None])[0]
        # print("p", self.p_debug["p_values"](obs[None])[0])
        # print("act", self.act(obs[None])[0])
        if np.isnan(theac):
            print('NaN action in MADDPGAgentTrainer')
            pdb.set_trace()
            print('NaN action in MADDPGAgentTrainer')
        return theac

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def set_memory_index(self, replay_sample_index):
        self.replay_sample_index = replay_sample_index

    def get_memory_index(self, batch_size):
        return self.replay_buffer.make_index(batch_size)

    def get_replay_data(self):
        return self.replay_buffer.sample_index(self.replay_sample_index)

    def get_target_act(self, obs):
        pdb.set_trace()
        return self.p_debug['target_act'](obs[self.agent_index])

    def update_q(self, t, obs_n, act_n, obs_next_n, target_act_next_n):
        """Update critic network."""
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(
            self.replay_sample_index)

        # train q network
        target_q = 0.0
        target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
        target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        q_loss, q_loss_summary = self.q_train(*(obs_n + act_n + [target_q]))

        self.summary_writer.add_summary(q_loss_summary, global_step=t)

        self.q_update()  # update critic

    def update_p(self, t, obs_n, target_act_next_n):
        """Update policy network."""
        p_loss, p_summary = self.p_train(*(obs_n + target_act_next_n))

        self.summary_writer.add_summary(p_summary, global_step=t)
        self.p_update()

    def update(self, agents, t):
        # replay buffer is not large enough
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = \
                self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = \
                    agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # num_sample = 1
        # for i in range(num_sample):
        target_act_next_n = \
            [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
        self.update_q(t, obs_n, act_n, obs_next_n, target_act_next_n)
        self.update_p(t, obs_n, target_act_next_n)
        # please actually write things to file, summary_writer
        self.summary_writer.flush()
