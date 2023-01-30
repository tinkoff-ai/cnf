import d4rl
import gym
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from cnf.awac import AdvantageWeightedActorCritic
from cnf.actor_critic import TanhActor, Critic
from cnf.make_flow import make_flow
from cnf.replay_buffer import ReplayBuffer
from cnf.flow_env import FlowEnv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_models(
        hidden_size, n_layers,
        action_size, state_size,
        state_mean=None, state_std=None
):
    actor = TanhActor(
        state_size, action_size, hidden_size, state_mean, state_std,
        n_layers=n_layers
    )
    actor.to(device)
    critic_1 = Critic(
        state_size, action_size, hidden_size, state_mean, state_std,
        n_layers=n_layers
    )
    critic_1.to(device)
    critic_2 = Critic(
        state_size, action_size, hidden_size, state_mean, state_std,
        n_layers=n_layers
    )
    critic_2.to(device)
    return actor, critic_1, critic_2


def make_optimizers(actor, critic_1, critic_2, actor_lr, critic_lr):
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=critic_lr)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=critic_lr)
    return actor_optimizer, critic_1_optimizer, critic_2_optimizer


def evaluate(
        env: FlowEnv,
        actor: nn.Module,
        n_episodes: int,
        eval_device: torch.device = device
):
    returns = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        total_reward = 0
        while not done:
            state_t = torch.tensor(state[None], dtype=torch.float32, device=eval_device)
            with torch.no_grad():
                action_t, _ = actor.act(state_t)
            action = action_t.cpu().numpy()[0]
            state, reward, done, _ = env.step(action, eval_device)
            total_reward += reward
        returns.append(total_reward)
    return returns


def run(log_dir, config, logger):
    test_env = gym.make(config['env_name'])
    state_size = test_env.observation_space.shape[0]
    action_size = test_env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(test_env)

    if 'hopper' in config['env_name']:
        action_size += 1
        dataset['actions'] = np.concatenate(
            (dataset['actions'], np.zeros((dataset['actions'].shape[0], 1))),
            axis=-1
        )

    replay_buffer = ReplayBuffer(state_size, action_size)
    replay_buffer.load(dataset)
    states = replay_buffer.storage['state']
    state_mean = torch.tensor(states.mean(axis=0), dtype=torch.float32)
    state_std = torch.tensor(states.std(axis=0), dtype=torch.float32)

    flow = make_flow(
        device,
        12, action_size, state_size,
        256, add_atanh=True, uniform_latent=True,
        state_embedding_mean=state_mean, state_embedding_std=state_std
    )
    flow_state_dict = torch.load(config['flow_checkpoint_path'], map_location='cpu')
    flow.load_state_dict(flow_state_dict)
    flow.to(device)

    test_env = FlowEnv(flow, test_env)

    actor, critic_1, critic_2 = make_models(
        config['hidden_size'], config['n_layers'],
        action_size, state_size,
        state_mean, state_std
    )
    actor_optimizer, critic_1_optimizer, critic_2_optimizer = make_optimizers(
        actor, critic_1, critic_2, config['lr'], config['lr']
    )

    algorithm = AdvantageWeightedActorCritic(
        actor, actor_optimizer,
        critic_1, critic_1_optimizer,
        critic_2, critic_2_optimizer,
        config['gamma'], config['tau'],
        config['awac_lambda'],
        flow=flow
    )

    data_consumed = 0
    best_mean_reward = -float('inf')
    p_bar = tqdm(range(config['n_train_steps']), desc='training', ncols=80)
    for i in p_bar:
        batch = replay_buffer.sample(config['batch_size'], device)
        update_result = algorithm.update(*batch)
        data_consumed += config['batch_size']
        if (i + 1) % config['log_freq'] == 0:
            if update_result is not None:
                update_result.update({'data_consumed': data_consumed})
                logger.log(update_result)

        if (i + 1) % config['eval_freq'] == 0:
            rewards = evaluate(test_env, actor, config['n_test_episodes'])
            mean_rewards = np.mean(rewards)
            mean_rewards = test_env.get_normalized_score(mean_rewards) * 100.0
            if best_mean_reward < mean_rewards:
                best_mean_reward = mean_rewards
            logger.log({
                'best_mean_rewards': best_mean_reward,
                'test_mean_rewards': mean_rewards,
                'data_consumed': data_consumed
            })
            if log_dir is not None:
                torch.save(actor.state_dict(), log_dir + f'/checkpoints/model_{data_consumed}.pt')
