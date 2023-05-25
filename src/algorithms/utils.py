import dataclasses
import glob
import json
import os
import random
import uuid
from typing import Optional

import d4rl
import gym
import h5py
import numpy as np
import torch
import wandb


def get_latest_run_id(save_dir: str) -> int:
    max_run_id = 0
    for path in glob.glob(os.path.join(save_dir, 'run_[0-9]*')):
        filename = os.path.basename(path)
        ext = filename.split('_')[-1]
        if ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id

def make_save_dir(config):
    # create save directories
    if config.run_id is not None:
        config.save_dir += f"/run_{config.run_id}"
    else:
        run_id = get_latest_run_id(save_dir=config.save_dir) + 1
        config.save_dir += f"/run_{run_id}"
    os.makedirs(config.save_dir, exist_ok=True)
    print(f"Results will be saved to: {config.save_dir}")

    # save config
    with open(os.path.join(config.save_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=4)

def wandb_init(config):
    # wandb logging
    with open("../../wandb_credentials.json", 'r') as json_file:
        credential_json = json.load(json_file)
        key = credential_json["wandb_key"]
    wandb.login(key=key)

    if config.use_wandb and config.name is None:
        config.name = config.save_dir.replace('/', '_')

    config = dataclasses.asdict(config)
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def load_dataset(config, env):
    dataset = {}
    if config.dataset_name:
        # local dataset
        data_hdf5 = h5py.File(f"./datasets/{config.dataset_name}", "r")
        for key in data_hdf5.keys():
            dataset[key] = np.array(data_hdf5[key])
    else:
        # remote dataset
        dataset = d4rl.qlearning_dataset(env)

    return dataset