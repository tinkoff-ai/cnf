from typing import Optional

import gym
import torch
import torch.nn as nn


class FlowEnv:
    def __init__(
            self,
            flow: Optional[nn.Module],
            env: gym.Env
    ):
        self._env = env
        self._flow = flow
        self._env_state = None

    def step(self, latent_space_action, device):
        with torch.no_grad():
            env_action_t, _ = self._flow.flow_inverse(
                torch.tensor(latent_space_action[None], dtype=torch.float32, device=device),
                torch.tensor(self._env_state[None], dtype=torch.float32, device=device)
            )
        env_action = env_action_t.cpu().numpy()[0]
        # special 'wrapper' for hopper env:
        if self._env.action_space.shape[0] == 3:
            env_action = env_action[:3]
        state, reward, done, info = self._env.step(env_action)
        self._env_state = state
        return state, reward, done, info

    def reset(self):
        self._env_state = self._env.reset()
        return self._env_state

    def get_normalized_score(self, score):
        # noinspection PyUnresolvedReferences
        return self._env.get_normalized_score(score)
