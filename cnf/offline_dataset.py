import gym

import d4rl
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split


def make_dataloader_d4rl(
        batch_size: int,
        env_name: str,
        loader_for_bc: bool = False
):
    eps = 1e-5
    actions_low = -1 + 2 * eps
    actions_high = +1 - 2 * eps
    test_size_fraction = 0.1  # 10%

    d4rl.set_dataset_path('~/d4rl')
    env = gym.make(env_name)
    d4rl_dataset = env.get_dataset()
    env.close()

    if loader_for_bc:
        observations, actions = [
            torch.tensor(d4rl_dataset[key], dtype=torch.float32)
            for key in ['observations', 'actions']
        ]
    else:
        observations, actions, rewards, terminals, next_observations = [
            torch.tensor(d4rl_dataset[key], dtype=torch.float32)
            for key in ['observations', 'actions', 'rewards', 'terminals', 'next_observations']
        ]
        rewards = rewards[..., None]
        terminals = terminals[..., None]
        if 'maze' in env_name:
            rewards -= 1.0

    actions = torch.clamp(actions, actions_low, actions_high)
    if 'hopper' in env_name:
        zeros = torch.zeros(actions.size(0), 1, dtype=torch.float32)
        actions = torch.cat((actions, zeros), dim=-1)

    observations_mean = observations.mean(dim=0)
    observations_std = observations.std(dim=0)

    if loader_for_bc:
        # because actions are 'data' for flows they must come first in the batches
        tensors = [actions, observations]
    else:
        tensors = [observations, actions, rewards, terminals, next_observations]

    train_test_data = train_test_split(*tensors, test_size=test_size_fraction)
    train_data, test_data = train_test_data[::2], train_test_data[1::2]

    train_dataset = TensorDataset(*train_data)
    test_dataset = TensorDataset(*test_data)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )
    return train_dataloader, test_dataloader, observations_mean, observations_std
