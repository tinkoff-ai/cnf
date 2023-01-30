import argparse

import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help='env name to train on')
    parser.add_argument('--n_epoch', type=int, help='number of training epochs')
    args = parser.parse_args()
    return args


parameters = {
    'n_layers': {'value': 12},
    'hidden_size': {'value': 256},
    'use_atanh': {'value': True},
    'uniform_latent': {'value': True},
    'lr': {'min': 1e-5, 'max': 3e-3},
    'wd': {'min': 0.0, 'max': 1e-2},

    'batch_size': {'values': [512, 1024, 2048]},
}


sweep_config = {
    'method': 'random',
    'metric': {'name': 'valid_mean_neg_log_prob', 'goal': 'minimize'},
    'parameters': parameters
}


def main():
    args = parse_args()
    parameters['env_name'] = {'value': args.env_name}
    parameters['n_epoch'] = {'value': args.n_epoch}
    sweep_config['name'] = f'flow_bc_{args.env_name}'
    wandb.sweep(sweep_config, project='flows')


if __name__ == '__main__':
    main()
