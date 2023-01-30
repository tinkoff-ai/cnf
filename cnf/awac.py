from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn as nn


class AdvantageWeightedActorCritic:
    def __init__(
            self,
            actor: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            critic_1: nn.Module,
            critic_1_optimizer: torch.optim.Optimizer,
            critic_2: nn.Module,
            critic_2_optimizer: torch.optim.Optimizer,
            gamma: float = 0.99,
            tau: float = 5e-3,  # parameter for the soft target update,
            awac_lambda: float = 1.0,
            flow: Optional[nn.Module] = None
    ):
        self._actor = actor
        self._actor_optimizer = actor_optimizer

        self._critic_1 = critic_1
        self._critic_1_optimizer = critic_1_optimizer
        self._target_critic_1 = deepcopy(critic_1)

        self._critic_2 = critic_2
        self._critic_2_optimizer = critic_2_optimizer
        self._target_critic_2 = deepcopy(critic_2)

        self._flow = flow

        self._gamma = gamma
        self._tau = tau
        self._awac_lambda = awac_lambda

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self._tau) * tp.data + self._tau * sp.data)

    def _actor_loss(self, state, action):
        with torch.no_grad():
            pi_action, pi_log_prob = self._actor.act(state)
            if self._flow is not None:
                pi_action, _ = self._flow.flow_inverse(pi_action, state)

            v = torch.min(
                self._critic_1(state, pi_action),
                self._critic_2(state, pi_action)
            )

            q = torch.min(
                self._critic_1(state, action),
                self._critic_2(state, action)
            )
            adv = q - v
            weights = torch.clamp_max(torch.exp(adv / self._awac_lambda), 100.0)

        policy_action, _ = self._actor.act(state)
        if self._flow is not None:
            policy_action, _ = self._flow.flow_inverse(policy_action, state)
        action_diff = torch.nn.functional.l1_loss(policy_action, action, reduction='none')
        loss = torch.mean(weights * action_diff)
        return loss

    def _critic_loss(self, state, action, reward, done, next_state):
        with torch.no_grad():
            next_action, _ = self._actor.act(next_state)

            if self._flow is not None:
                next_action, _ = self._flow.flow_inverse(next_action, next_state)

            q_next = torch.min(
                self._target_critic_1(next_state, next_action),
                self._target_critic_2(next_state, next_action)
            )
            q_target = reward + self._gamma * (1.0 - done) * q_next

        q1 = self._critic_1(state, action)
        q2 = self._critic_2(state, action)

        q1_loss = nn.functional.mse_loss(q1, q_target)
        q2_loss = nn.functional.mse_loss(q2, q_target)
        loss = q1_loss + q2_loss

        assert q1.shape == q_target.shape and q2.shape == q_target.shape

        return loss

    def _update_critic(self, state, action, reward, done, next_state):
        loss = self._critic_loss(state, action, reward, done, next_state)
        self._critic_1_optimizer.zero_grad()
        self._critic_2_optimizer.zero_grad()
        loss.backward()
        self._critic_1_optimizer.step()
        self._critic_2_optimizer.step()
        return loss.item()

    def _update_actor(self, state, action):
        loss = self._actor_loss(state, action)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        return loss.item()

    def update(self, *batch: torch.Tensor) -> Dict[str, float]:
        state, action = batch[0], batch[1]
        critic_loss = self._update_critic(*batch)
        actor_loss = self._update_actor(state, action)

        self._soft_update(self._target_critic_1, self._critic_1)
        self._soft_update(self._target_critic_2, self._critic_2)

        result = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss
        }
        return result
