# maddpg/

## AgentTrainer

- **THIS IS THE AGENT POLICY**
- Class in `maddpg/__init__.py`
    - `__init__(name, model, obs_shape, act_space, args)`
    - `action(obs)`
    - `process_experience(obs, act, rew, new_obs, done, terminal)`

## maddpg/trainer

- `maddpg.py`
    - `discount_with_dones(rewards, dones, gamma)`
    - `make_update_exp(vals, target_vals)`
    - `p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer)`
    - `q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer)`
    - `Class MADDPGAgentTrainer(AgentTrainer)`
        - `__init__(self, name, model, obs_shape_n, act_space_n, agent_index, args)`
        - `action(obs)`
            - returns `self.act(obs[None])[0]`
        - `experience(obs, act, rew, new_obs, done, terminal)`
        - `preupdate()`
            - `self.replay_sample_index = None`
        - `update(agents,t)`
- `replay_buffer.py` - Contains ReplayBuffer class
    - `__init__()` Creates a prioritized replay buffer
    - `__len__()` returns `len(self.storage)`
    - `clear()` clears storage and sets self._next_idx to 0
    - `add(obs_t, action, reward, obs_tp1, done)`
    - `_encode_sample(idxes)`
    - `make_index(batch_size)`
    - `make_latest_index(batch_size)`
    - `sample_index(idxes)`
    - `sample(batch_size)`
    - `collect()`

## maddpg/common

- `distributions.py` - contains functions assisting with calculating probability distributions
- `tf_util.py` - Tensorflow utility functions

# experiments/train.py

- `parse_args()`
- `mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None)`
    - This model takes as input an observation and returns values of all actions
- `make_env(scenario_name, arglist, benchmark=False)`
- `set_trainers(env, num_adversaries, obs_shape_n, arglist)`
- `train(arglist)`
    - Most important function, basically runs the train.py experiment
    - import `maddpg/common/tf_util` as U
- main
    - Gets argument list from parse_args and calls train()

## train() Execution:

- Inside a U.single_threaded_session()
    - Create environment
    - Create agent trainers
    - U.initialize()
    - Load previous results if needed
    - Set all variables to starting empty values
    - Infinite loop:
        - Put actions of all agents in `action_n`
        - Perform `environment.step(action_n)`
        - Apply experiences to agents using agent.experience()
        - Handle if cases are done or max iteration reached
    - Benchmark & display policies
    - Update all trainers if not in display or benchmark mode
    - Save model, display training output if terminal or a save iteration (saves model every x iterations)
    - Save final episode reward for plotting training curve




