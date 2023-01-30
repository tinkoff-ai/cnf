import os
import argparse

import wandb

from cnf.train_flow import run


config = {
    'n_layers': 12,
    'hidden_size': 256,
    'use_atanh': True,
    'lr': 5e-4,
    'wd': 1e-4,
    'batch_size': 1024
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help='env name to train on')
    parser.add_argument('--normal', action='store_true', help='latent distribution flag')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    env_name = args.env_name
    if 'expert' in env_name:
        n_epoch = 50
    elif 'replay' in env_name:
        n_epoch = 350
    else:  # medium
        n_epoch = 100

    config['env_name'] = env_name
    config['n_epoch'] = n_epoch
    config['uniform_latent'] = not args.normal

    log_dir = f'logs/cnf/flows_pretraining/{env_name}_{"normal" if args.normal else "uniform"}'
    os.makedirs(log_dir + '/checkpoints', exist_ok=True)

    with wandb.init(
        project='flows',
        group='flows_pretrain',
        name=f'{env_name}_{"normal" if args.normal else "uniform"}'
    ) as logger:
        run(log_dir, config, logger)


if __name__ == '__main__':
    main()
