import os
import pickle
import shutil
from pprint import pprint

import gym
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from wandb.integration.sb3 import WandbCallback
import abc

import wandb
from rl.agents import SACLearner
from rl.data import ReplayBuffer
from rl.evaluation import evaluate
from rl.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "Ardupilot-v0", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 1, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 1000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e2), "Number of training steps to start training."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("wandb", True, "Log wandb.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_float("action_filter_high_cut", None, "Action filter high cut.")
flags.DEFINE_integer("action_history", 1, "Action history.")
flags.DEFINE_integer("control_frequency", 20, "Control frequency.")
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
flags.DEFINE_boolean("real_robot", True, "Use real robot.")
config_flags.DEFINE_config_file(
    "config",
    "configs/sac_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

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
    wandb.init(
        project="winp",
        sync_tensorboard=True,
    )
    wandb.config.update(FLAGS)

    if FLAGS.real_robot:
        print
        from ardupilot_gym.ardupilot_gym.envs.sitl import ArduPlaneSITLEnv
        env = ArduPlaneSITLEnv()
        acro_agent = AcroAgent(env)
 
    else:
        from env_utils import make_mujoco_env

        env = make_mujoco_env(
            FLAGS.env_name,
            control_frequency=FLAGS.control_frequency,
            action_filter_high_cut=FLAGS.action_filter_high_cut,
            action_history=FLAGS.action_history,
        )

    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env = gym.wrappers.RecordVideo(
        env,
        f"videos/train_{FLAGS.action_filter_high_cut}",
        episode_trigger=lambda x: True,
    )
    env.seed(FLAGS.seed)

    if not FLAGS.real_robot:
        eval_env = make_mujoco_env(
            FLAGS.env_name,
            control_frequency=FLAGS.control_frequency,
            action_filter_high_cut=FLAGS.action_filter_high_cut,
            action_history=FLAGS.action_history,
        )
        eval_env = wrap_gym(eval_env, rescale_actions=True)
        eval_env = gym.wrappers.RecordVideo(
            eval_env,
            f"videos/eval_{FLAGS.action_filter_high_cut}",
            episode_trigger=lambda x: True,
        )
        eval_env.seed(FLAGS.seed + 42)

    # kwargs = dict(FLAGS.config)

    chkpt_dir = "saved/checkpoints"
    # last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)

    # train_freq = 8
    train_freq = 1
    hyperparameters = dict(
        learning_starts=FLAGS.start_training,
        train_freq=train_freq,
        gradient_steps=FLAGS.utd_ratio * train_freq,
        tensorboard_log="saved/runs/",
        use_sde=True,
        # gamma=0.98,
        sde_sample_freq=8,
        # use_sde_at_warmup=False,
        # learning_rate=7.3e-4,
        # ent_coef="auto_0.1",
        # tau=0.02,
        # top_quantiles_to_drop_per_net=0,
        # policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300])
        # policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300], layer_norm=True)
        # policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300], layer_norm=True, dropout_rate=0.01)
        policy_kwargs=dict(layer_norm=True, dropout_rate=0.01),
        acro_agent=acro_agent,
    )

    pprint(hyperparameters)
    # eval_callback = EvalCallback(eval_env, eval_freq=FLAGS.eval_interval)
    wandb_callback = WandbCallback()
    # checkpoint_callback = CheckpointCallback(chkpt_dir, save_freq=FLAGS.eval_interval)

    model = TQC("MlpPolicy", env, verbose=1, **hyperparameters)
    try:
        model.learn(
            total_timesteps=FLAGS.max_steps, callback=[wandb_callback]
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)