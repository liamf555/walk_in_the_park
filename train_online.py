#! /usr/bin/env python
import os
import pickle
import shutil
import time

import numpy as np
import tqdm

import gym
import wandb
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from rl.agents import SACLearner
from rl.data import ReplayBuffer
from rl.evaluation import evaluate
from rl.wrappers import wrap_gym
import abc

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'Ardupilot-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 1000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1000),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('wandb', False, 'Log wandb.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_float('action_filter_high_cut', None, 'Action filter high cut.')
flags.DEFINE_integer('action_history', 1, 'Action history.')
flags.DEFINE_integer('control_frequency', 20, 'Control frequency.')
flags.DEFINE_integer('utd_ratio', 1, 'Update to data ratio.')
flags.DEFINE_string('task', 'knife', 'Task to train on')
flags.DEFINE_boolean('real_robot', False, 'Use real robot.')
config_flags.DEFINE_config_file(
    'config',
    'configs/sac_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

class Agent:
    """Abstract class for all agents."""

    @abc.abstractmethod
    def act(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues an action given an observation.

        Args:
            obs (np.ndarray): the observation for which the action is needed.

        Returns:
            (np.ndarray): the action.
        """
        pass

    def plan(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues a sequence of actions given an observation.

        Unless overridden by a child class, this will be equivalent to :meth:`act`.

        Args:
            obs (np.ndarray): the observation for which the sequence is needed.

        Returns:
            (np.ndarray): a sequence of actions.
        """

        return self.act(obs, **_kwargs)

    def reset(self):
        """Resets any internal state of the agent."""
        pass
class AcroAgent(Agent):
    """An agent that performs aerobatic actions to populate the replay buffer

    Args:
        env (gym.Env): the environment on which the agent will act.

    """
    def __init__(self, env: gym.Env):
        self.env = env
        

    def act(self, *_args, **_kwargs) -> np.ndarray:
        return self.env.manoeuvre_instance.do_manoeuvre()

def main(_):
    wandb.init(project='winp')
    wandb.config.update(FLAGS)

    if FLAGS.real_robot:
        if FLAGS.task == 'knife':
            from ardupilot_gym.ardupilot_gym.envs.aeropytics_knife import AeropyticsEnvKnife
            env = AeropyticsEnvKnife(acro_mode=True)
        elif FLAGS.task == 'hover':
            from ardupilot_gym.ardupilot_gym.envs.aeropytics_hover import AeropyticsEnvHover
            env = AeropyticsEnvHover(acro_mode=True)

            
    
        acro_agent = AcroAgent(env)
        print("Acro agent initialised")
    env = gym.wrappers.TimeLimit(env, env.episode_n_steps)
    env = wrap_gym(env, rescale_actions=True)
    
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    # print(vars(env))
    # env = gym.wrappers.RecordVideo(
    #     env,
    #     f'videos/train_{FLAGS.action_filter_high_cut}',
    #     episode_trigger=lambda x: True)
    env.seed(FLAGS.seed)

    # if not FLAGS.real_robot:
    #     eval_env = make_mujoco_env(
    #         FLAGS.env_name,
    #         control_frequency=FLAGS.control_frequency,
    #         action_filter_high_cut=FLAGS.action_filter_high_cut,
    #         action_history=FLAGS.action_history)
    #     eval_env = wrap_gym(eval_env, rescale_actions=True)
    #     eval_env = gym.wrappers.RecordVideo(
    #         eval_env,
    #         f'videos/eval_{FLAGS.action_filter_high_cut}',
    #         episode_trigger=lambda x: True)
    #     eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    agent = SACLearner.create(FLAGS.seed, env.observation_space,
                              env.action_space, **kwargs)

    chkpt_dir = 'saved/checkpoints'
    os.makedirs(chkpt_dir, exist_ok=True)
    buffer_dir = 'saved/buffers'

    last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)

    if last_checkpoint is None:
        start_i = 0
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                     FLAGS.max_steps)
        replay_buffer.seed(FLAGS.seed)
    else:
        start_i = int(last_checkpoint.split('_')[-1])

        agent = checkpoints.restore_checkpoint(last_checkpoint, agent)

        with open(os.path.join(buffer_dir, f'buffer_{start_i}'), 'rb') as f:
            replay_buffer = pickle.load(f)

    previous_time = 0
    total_alt_error = 0

    if FLAGS.task == 'knife':

        total_roll_error = 0
        total_yaw_error = 0

    elif FLAGS.task == 'hover':

        total_pitch_error = 0
        total_x_error = 0
        total_y_error = 0

    total_obs_vel_x =0
    total_obs_vel_y =0
    total_obs_vel_z =0
    total_obs_p =0
    total_obs_q =0
    total_obs_r =0
    
    ep_steps = 0
    reset_flag = True

    # observation, done = env.reset(), False
    for i in tqdm.tqdm(range(start_i, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if reset_flag:
            observation, done = env.reset(), False
            reset_flag = False
        start_time = time.perf_counter()
        # if i < FLAGS.start_training:
        if env.acro_mode:
            # action = env.action_space.sample()
            # print("Acro agent acting")
            action = acro_agent.act()
        else:
            # print("Agent acting")
            action, agent = agent.sample_actions(observation)

        next_observation, reward, done, info = env.step(action)

        total_alt_error += abs(env.get_alt_error())

        if FLAGS.task == 'knife':

            total_roll_error += abs(env.get_roll_error())
            total_yaw_error += abs(env.get_heading_error())
            
        
        elif FLAGS.task == 'hover':
            total_pitch_error += abs(env.get_pitch_error())
            total_x_error += abs(env.get_x_error())
            total_y_error += abs(env.get_y_error())

        total_obs_vel_x += abs(env.get_obs_vel_x())
        total_obs_vel_y += abs(env.get_obs_vel_y())
        total_obs_vel_z += abs(env.get_obs_vel_z())
        total_obs_p += abs(env.get_obs_roll_rate())
        total_obs_q += abs(env.get_obs_pitch_rate())
        total_obs_r += abs(env.get_obs_yaw_rate())
        
        ep_steps += 1


        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(observations=observation,
                 actions=action,
                 rewards=reward,
                 masks=mask,
                 dones=done,
                 next_observations=next_observation))
        observation = next_observation

        if done:
            env.mavlink_mission_set_current()
            env.increment_manoeuvre()
            # env.increment_waypoint()
            reset_flag = True
            # observation, done = env.reset(), False
            if not env.acro_mode:
                # print("Decoding")
                for k, v in info['episode'].items():
                    decode = {'r': 'episode_reward', 'l': 'length', 't': 'time'}
                    wandb.log({f'results/{decode[k]}': v}, step=i)
                wandb.log({'results/alt_error': total_alt_error/ep_steps}, step=i)
                
                if FLAGS.task == 'knife':
                    wandb.log({'results/roll_error': total_roll_error/ep_steps}, step=i)
                    wandb.log({'results/yaw_error': total_yaw_error/ep_steps}, step=i)
                elif FLAGS.task == 'hover':
                    wandb.log({'results/x_error': total_x_error/ep_steps}, step=i)
                    wandb.log({'results/y_error': total_y_error/ep_steps}, step=i)
                    wandb.log({'results/pitch_error': total_pitch_error/ep_steps}, step=i)
                
                wandb.log({'results/obs_vel_x': total_obs_vel_x/ep_steps}, step=i)
                wandb.log({'results/obs_vel_y': total_obs_vel_y/ep_steps}, step=i)
                wandb.log({'results/obs_vel_z': total_obs_vel_z/ep_steps}, step=i)
                wandb.log({'results/obs_p': total_obs_p/ep_steps}, step=i)
                wandb.log({'results/obs_q': total_obs_q/ep_steps}, step=i)
                wandb.log({'results/obs_r': total_obs_r/ep_steps}, step=i)

                total_alt_error = 0

                if FLAGS.task == 'knife':

                    total_roll_error = 0
                    total_yaw_error = 0

                elif FLAGS.task == 'hover':

                    total_pitch_error = 0
                    total_x_error = 0
                    total_y_error = 0

                total_obs_vel_x =0
                total_obs_vel_y =0
                total_obs_vel_z =0
                total_obs_p =0
                total_obs_q =0
                total_obs_r =0
                
                ep_steps = 0
                
    

        if not env.acro_mode:
            print("Agent training")
            # env.hold()
            env.reset_action()
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)
            # print("Agent trained")

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f'training/{k}': v}, step=i)

        if i % FLAGS.eval_interval == 0:
            # if not FLAGS.real_robot:
            #     eval_info = evaluate(agent,
            #                          eval_env,
            #                          num_episodes=FLAGS.eval_episodes)
            #     for k, v in eval_info.items():
            #         wandb.log({f'evaluation/{k}': v}, step=i)

            checkpoints.save_checkpoint(chkpt_dir,
                                        agent,
                                        step=i + 1,
                                        keep=20,
                                        overwrite=True)

            try:
                shutil.rmtree(buffer_dir)
            except:
                pass

            os.makedirs(buffer_dir, exist_ok=True)
            with open(os.path.join(buffer_dir, f'buffer_{i+1}'), 'wb') as f:
                pickle.dump(replay_buffer, f)

if __name__ == '__main__':
    app.run(main)
