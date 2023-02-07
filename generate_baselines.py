from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from envs.walk_to_goal import WalkToGoalEnv
from envs.walk_to_ball import WalkToBallEnv
from envs.push_ball_to_goal import PushBallToGoalEnv
from envs.goalkeeper import GoalKeeperEnv
from envs.dummy_defenders import DummyDefendersEnv
from envs.goalie import GoalieEnv
from envs.base import BaseEnv
from envs.keepaway import KeepawayEnv
from envs.kick_to_goal import KickToGoalEnv
from envs.defender import DefenderEnv

from utils.utils import save_vec_normalize_data
import sys


# This file is provided to generate all the baselines models and vector-normalization parameters and save them in their
# respective subfolders of the vectornormalization folder. You can validate that these models are performing acceptably well
# using validate_baselines.py


models = {
    "base": {
        "env": BaseEnv,
        "path": "base",
        "training_steps": 5000000,
        "starter_model": None,
    },
    "walk_to_goal": {
        "env": WalkToGoalEnv,
        "path": "walk_to_goal",
        "training_steps": 5000000,
        "starter_model": None,
    },
    "walk_to_ball": {
        "env": WalkToBallEnv,
        "path": "walk_to_ball",
        "training_steps": 5000000,
        "starter_model": None,
    },
    "push_ball_to_goal": {
        "env": PushBallToGoalEnv,
        "path": "push_ball_to_goal",
        "training_steps": 10000000,
        "starter_model": None,
    },
    "dummy_defenders": {
        "env": DummyDefendersEnv,
        "path": "dummy_defenders",
        "training_steps": 5000000,
        "starter_model": "push_ball_to_goal",
    },
    "goalie": {
        "env": GoalieEnv,
        "path": "goalie",
        "training_steps": 5000000,
        "starter_model": "push_ball_to_goal",
    },
    "keepaway": {
        "env": KeepawayEnv,
        "path": "keepaway",
        "training_steps": 15000000,
        "starter_model": None,
    },
    "kick_to_goal": {
        "env": KickToGoalEnv,
        "path": "kick_to_goal",
        "training_steps": 5000000,
        "starter_model": None,
    },
     "goalkeeper": {
        "env": GoalKeeperEnv,
        "path": "goalkeeper",
        "training_steps": 5000000,
        "starter_model": None,
    },
    "defender": {
        "env": DefenderEnv,
        "path": "defender",
        "training_steps": 5000000,
        "starter_model": None,
    },
}

if __name__ == "__main__":
    # Get the model name from the command line
    model_name = sys.argv[1]

    # Check model name is valid
    if model_name not in models:
        print("Invalid model name")
        sys.exit(1)

    params = models[model_name]

    env = None
    model = None

    if params["starter_model"] == None:
        env = VecNormalize(
            make_vec_env(params["env"], n_envs=12),
            norm_obs=True,
            norm_reward=True,
            clip_obs=1.0,
        )

        model = PPO("MlpPolicy", env, verbose=1)
    else:
        starter_model_params = models[params["starter_model"]]

        env = VecNormalize.load(
            f"./expert_policies/{starter_model_params['path']}/vector_normalize",
            make_vec_env(params["env"], n_envs=12),
        )
        env.norm_obs = True
        env.norm_reward = True
        env.clip_obs = 1.0
        env.training = True
        model = PPO.load(
            f"./expert_policies/{starter_model_params['path']}/policy",
            env=env,
        )

    env = VecMonitor(venv=env, filename=f"./expert_policies/{params['path']}/")
    model.learn(total_timesteps=params["training_steps"])
    model.save(f"./expert_policies/{params['path']}/policy")
    env.save(f"./expert_policies/{params['path']}/vector_normalize")
    save_vec_normalize_data(env, f"./expert_policies/{params['path']}/vector_normalize.json")