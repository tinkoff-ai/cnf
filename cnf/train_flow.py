from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm

from cnf.offline_dataset import make_dataloader_d4rl
from cnf.make_flow import make_flow


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SAMPLES_FOR_VALID = 100


def init_flow_act_norm(dataloader, flow):
    # initialize act norm layers with the first batch of data
    # it seems to be __VERY__ helpful
    for batch in dataloader:
        if type(batch) is torch.Tensor:
            batch = [batch]
        batch = [b.to(device) for b in batch]
        with torch.no_grad():
            flow.log_prob(*batch)
        break


def optimize_flow(
        flow: nn.Module,
        optimizer: torch.optim.Optimizer,
        data: List[torch.Tensor],
        clip_grad: float = 1e-1
):
    log_prob = flow.log_prob(*data)
    loss = -log_prob.mean()
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), clip_grad)
    optimizer.step()
    return {
        'loss': loss.item(),
        'grad_norm': grad_norm.item()
    }


def train_epoch(epoch, data_loader, flow, optimizer, consumed_data, logger):
    flow.train()
    p_bar = tqdm(data_loader, desc=f'train_{epoch + 1}', ncols=80)
    for batch in p_bar:
        batch = [b.to(device) for b in batch]
        optimization_result = optimize_flow(flow, optimizer, batch)
        consumed_data += batch[0].size(0)
        optimization_result.update({'train_consumed_data': consumed_data})
        logger.log(optimization_result)
    return consumed_data


def valid_batch(flow, batch):
    actions, states = batch
    batch_size = states.size(0)
    # sample data from flow, measure MAE and min MAE
    states_for_sampling = torch.cat([states for _ in range(N_SAMPLES_FOR_VALID)], dim=0)
    actions_samples, samples_log_prob = flow.sample(states_for_sampling.size()[:-1], states_for_sampling)
    actions_for_mae = torch.cat([actions for _ in range(N_SAMPLES_FOR_VALID)], dim=0)
    mae = torch.abs(actions_for_mae - actions_samples).mean(-1)
    mae = mae.view(N_SAMPLES_FOR_VALID, -1)

    # measure std for samples from distributions for each state
    mean_samples_std = actions_samples.view(N_SAMPLES_FOR_VALID, batch_size, -1).std(0).mean()
    mean_mae = mae.mean()
    min_mae = torch.min(mae, dim=0)[0].mean()
    samples_mean_neg_log_prob = -samples_log_prob.mean()  # negative to be on the same scale as loss

    # measure log-prob on a valid data
    valid_mean_neg_log_prob = -flow.log_prob(actions, states).mean()

    return {
        'mean_samples_std': mean_samples_std.item(),
        'mean_mae': mean_mae.item(),
        'min_mae': min_mae.item(),
        'samples_mean_neg_log_prob': samples_mean_neg_log_prob.item(),
        'valid_mean_neg_log_prob': valid_mean_neg_log_prob.item()
    }


def valid_epoch(epoch, data_loader, flow, consumed_data, logger):
    flow.eval()
    with torch.no_grad():
        p_bar = tqdm(data_loader, desc=f'valid_{epoch + 1}', ncols=80)
        for batch in p_bar:
            batch = [b.to(device) for b in batch]
            valid_result = valid_batch(flow, batch)
            consumed_data += batch[0].size(0)
            valid_result.update({'valid_consumed_data': consumed_data})
            logger.log(valid_result)
    return consumed_data


def run(log_dir, config, logger):
    train_dataloader, valid_dataloader, state_mean, state_std = make_dataloader_d4rl(
        config['batch_size'], config['env_name'], loader_for_bc=True
    )

    action_dim, state_dim = None, None
    for action, state in train_dataloader:
        action_dim = action.size(-1)
        state_dim = state.size(-1)
        break

    flow = make_flow(
        device,
        config['n_layers'], action_dim, state_dim,
        config['hidden_size'],
        add_atanh=config['use_atanh'], uniform_latent=config['uniform_latent'],
        state_embedding_mean=state_mean, state_embedding_std=state_std
    )
    optimizer = torch.optim.AdamW(
        flow.parameters(),
        lr=config['lr'], weight_decay=config['wd']
    )
    init_flow_act_norm(train_dataloader, flow)
    train_consumed_data = 0
    valid_consumed_data = 0

    for i in range(config['n_epoch']):
        train_consumed_data = train_epoch(i, train_dataloader, flow, optimizer, train_consumed_data, logger)
        valid_consumed_data = valid_epoch(i, valid_dataloader, flow, valid_consumed_data, logger)
        torch.save(flow.state_dict(), log_dir + f'/checkpoints/epoch_{i + 1}.pt')
    print('done')
