#! /usr/bin/env python
import datetime
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
flags.DEFINE_integer('start_training', int(1300),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('wandb', True, 'Log wandb.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_float('action_filter_high_cut', None, 'Action filter high cut.')
flags.DEFINE_integer('action_history', 1, 'Action history.')
flags.DEFINE_integer('control_frequency', 20, 'Control frequency.')
flags.DEFINE_integer('utd_ratio', 1, 'Update to data ratio.')
flags.DEFINE_string('task', 'knife', 'Task to train on')
flags.DEFINE_boolean('real_robot', False, 'Use real robot.')
flags.DEFINE_boolean('save_state', False, 'Save state data to csv')
flags.DEFINE_boolean('acro_init', False, 'Initialise in acro mode')
flags.DEFINE_boolean('wind', False, 'Use wind in simulation')
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
            # from ardupilot_gym_old.ardupilot_gym.envs.sitl_2 import ArduPlaneSITLEnvWINP
            env = AeropyticsEnvKnife(acro_mode=FLAGS.acro_init)
            wandb.config.update({'manoeuvre': 'knife'})
        elif FLAGS.task == 'knife_pwm':
            from ardupilot_gym.ardupilot_gym.envs.aeropytics_knife_pwm import AeropyticsEnvKnifePWM
            env = AeropyticsEnvKnifePWM(acro_mode=FLAGS.acro_init)
            wandb.config.update({'manoeuvre': 'knife_pwm'})
            # env = ArduPlaneSITLEnvWINP()
        elif FLAGS.task == 'hover':
            from ardupilot_gym.ardupilot_gym.envs.aeropytics_hover import AeropyticsEnvHover
            env = AeropyticsEnvHover(acro_mode=FLAGS.acro_init)
            wandb.config.update({'manoeuvre': 'hover'})
    
        acro_agent = AcroAgent(env)
        print("Acro agent initialised")
    env.time_step = 1 / FLAGS.control_frequency
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

    save_dir = "./saved"
    run_dir = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    wandb.config.update({'run_dir': run_dir})

    chkpt_dir = f"{save_dir}/{run_dir}/checkpoints"
    os.makedirs(chkpt_dir, exist_ok=True)
    buffer_dir = f"{save_dir}/{run_dir}/buffers"

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

    if FLAGS.task == 'knife' or FLAGS.task == 'knife_pwm':

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
    total_roll = 0
    total_pitch = 0
    total_yaw = 0



    ep_state_list = []

    
    if FLAGS.save_state:
        state_dir = f"{save_dir}/{run_dir}/states"
        os.makedirs(state_dir, exist_ok=True)
        #create csv file
        with open(f"{state_dir}/episode_states.csv", 'w') as f:
            f.write('episode,time,x,y,z,roll,pitch,yaw,u,v,w,p,q,r,airspeed,action_1,action_2,action_3,throttle\n')
    
    ep_steps = 0
    ep_num = 0
    reset_flag = True
    loiter_counter = 0

    previous_time = time.perf_counter()

    # observation, done = env.reset(), False
    for i in tqdm.tqdm(range(start_i, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if reset_flag:
            observation, done = env.reset(), False
            reset_flag = False
            ep_state_list = []
            ep_num += 1
            init_state = env.get_full_state()
            init_state = np.insert(init_state, 0, ep_num)
            init_state = np.insert(init_state, 1, 0.0)

            ep_state_list.append(init_state)
            
        # if i < FLAGS.start_training:
        if FLAGS.acro_init:
            if env.acro_mode:
                # action = env.action_space.sample()
                # print("Acro agent acting")
                action = acro_agent.act()
            else:
                # print("Agent acting")
                action, agent = agent.sample_actions(observation)
                # print(f"Action: {action}")
        else:
            if i < FLAGS.start_training:
                action = env.action_space.sample()
            else:
                action, agent = agent.sample_actions(observation)

        next_observation, reward, done, info = env.step(action)

        episode_state = env.get_full_state()

        total_roll += np.degrees(episode_state[3])
        total_pitch += np.degrees(episode_state[4])
        total_yaw += np.degrees(episode_state[5])

        # add episode number and time to start of np array
        episode_state = np.insert(episode_state, 0, ep_num)
        episode_state = np.insert(episode_state, 1, env.episode_time)
        
        ep_state_list.append(episode_state)
        # ep_state_list.

        total_alt_error += abs(env.get_alt_error())

        if FLAGS.task == 'knife' or FLAGS.task == 'knife_pwm':
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
            if FLAGS.acro_init:
                env.increment_manoeuvre()
            # env.increment_waypoint()
            reset_flag = True
            # observation, done = env.reset(), False

            if FLAGS.save_state:
                    #save state data to csv
                    with open(f"{state_dir}/episode_states.csv", 'a') as f:
                        # each row is a list of values separated by commas
                        for row in ep_state_list:
                            # write row to csv
                            f.write(','.join([str(elem) for elem in row]))
                            f.write('\n')
            
               
            if FLAGS.acro_init and not env.acro_mode:
                if loiter_counter < 2:
                    env.set_flight_mode_loiter()
                    loiter_counter += 1
                # print("Decoding")
                # for k, v in info['episode'].items():
                    # decode = {'r': 'episode_reward', 'l': 'length', 't': 'time'}

                # create a dictionary with the common keys and values
                common_dict = {
                    'episode_reward': info['episode']['r'],
                    'length': info['episode']['l'],
                    'time': info['episode']['t'],
                    'alt_error': total_alt_error/ep_steps,
                    'roll': total_roll/ep_steps,
                    'pitch': total_pitch/ep_steps,
                    'yaw': total_yaw/ep_steps,
                    'x_error': total_x_error/ep_steps,
                    'y_error': total_y_error/ep_steps,
                    'u_vel': total_obs_vel_x/ep_steps,
                    'v_vel': total_obs_vel_y/ep_steps,
                    'w_vel': total_obs_vel_z/ep_steps,
                    'p_rate': total_obs_p/ep_steps,
                    'q_rate': total_obs_q/ep_steps,
                    'r_rate': total_obs_r/ep_steps,
                    'ep_num': ep_num
                }

                # add task-specific keys and values to the dictionary
                if FLAGS.task == 'knife' or FLAGS.task == 'knife_pwm':
                    common_dict['roll_error'] = total_roll_error/ep_steps
                    common_dict['yaw_error'] = total_yaw_error/ep_steps
                elif FLAGS.task == 'hover':
                    common_dict['pitch_error'] = total_pitch_error/ep_steps

                # log the dictionary to wandb
                wandb.log(common_dict, step=i)

                total_alt_error = 0

                if FLAGS.task == 'knife' or FLAGS.task == 'knife_pwm':

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
                total_roll = 0
                total_yaw = 0
                total_pitch = 0
                
                
                ep_steps = 0
               
            elif not FLAGS.acro_init and i >= FLAGS.start_training:
                if loiter_counter < 2:
                    env.set_flight_mode_loiter()
                    loiter_counter += 1
                # print("Decoding")
                # for k, v in info['episode'].items():
                    # decode = {'r': 'episode_reward', 'l': 'length', 't': 'time'}

                # create a dictionary with the common keys and values
                common_dict = {
                    'episode_reward': info['episode']['r'],
                    'length': info['episode']['l'],
                    'time': info['episode']['t'],
                    'alt_error': total_alt_error/ep_steps,
                    'roll': total_roll/ep_steps,
                    'pitch': total_pitch/ep_steps,
                    'yaw': total_yaw/ep_steps,
                    'x_error': total_x_error/ep_steps,
                    'y_error': total_y_error/ep_steps,
                    'u_vel': total_obs_vel_x/ep_steps,
                    'v_vel': total_obs_vel_y/ep_steps,
                    'w_vel': total_obs_vel_z/ep_steps,
                    'p_rate': total_obs_p/ep_steps,
                    'q_rate': total_obs_q/ep_steps,
                    'r_rate': total_obs_r/ep_steps,
                    'ep_num': ep_num
                }

                # add task-specific keys and values to the dictionary
                if FLAGS.task == 'knife' or FLAGS.task == 'knife_pwm':
                    common_dict['roll_error'] = total_roll_error/ep_steps
                    common_dict['yaw_error'] = total_yaw_error/ep_steps
                elif FLAGS.task == 'hover':
                    common_dict['pitch_error'] = total_pitch_error/ep_steps

                # log the dictionary to wandb
                wandb.log(common_dict, step=i)

                total_alt_error = 0

                if FLAGS.task == 'knife' or FLAGS.task == 'knife_pwm':

                    total_roll_error = 0
                    total_yaw_error = 0

                elif FLAGS.task == 'hover' or FLAGS.task == 'hover_pwm':

                    total_pitch_error = 0
                    total_x_error = 0
                    total_y_error = 0

                total_obs_vel_x =0
                total_obs_vel_y =0
                total_obs_vel_z =0
                total_obs_p =0
                total_obs_q =0
                total_obs_r =0
                total_roll = 0
                total_yaw = 0
                total_pitch = 0
                
                
                ep_steps = 0


        if FLAGS.acro_init and not env.acro_mode:
            # print("Agent training")
            # env.hold()
            env.reset_action()
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)
            # print("Agent trained")

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f'training/{k}': v}, step=i)
        elif not FLAGS.acro_init and i >= FLAGS.start_training:
            # print("Agent training")
            # env.hold()
            env.reset_action()
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)
            # print("Agent trained")

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f'training/{k}': v}, step=i)


        if i % FLAGS.eval_interval == 0:
      
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

        # time_now = time.perf_counter()
        # time_taken = time_now - previous_time
        # previous_time = time_now
        # wandb.log({'time_taken': time_taken})

if __name__ == '__main__':
    app.run(main)
