import argparse
import os

import wandb

from cnf.train_rl import run


lr = 3e-4

config = {
    'n_layers': 4,
    'hidden_size': 256,
    'lr': lr,
    'gamma': 0.99,
    'tau': 2e-3,
    'batch_size': 512,
    'n_train_steps': 200_000 + 1,  # +1 just to do one additional evaluation after training
    'eval_freq': 100,
    'log_freq': 100,
    'n_test_episodes': 10,
    # from IQL paper: locomotion -> lamda = 1 / 3; antmaze -> lambda = 1 / 10
    'awac_lambda': 1.0 / 3.0,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help='env name to train on')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    env_name = args.env_name
    config['env_name'] = env_name

    config['flow_checkpoint_path'] = f'flow_checkpoints/{args.env_name}.pt'

    log_dir = f'logs/cnf/{env_name}'
    os.makedirs(log_dir + '/checkpoints', exist_ok=True)
    print(f'training rl with flow for {env_name}')
    with wandb.init(
        project='flows',
        group='cnf',
        name=f'cnf_{env_name}',
        config=config
    ) as logger:
        run(log_dir, config, logger)


if __name__ == '__main__':
    main()
