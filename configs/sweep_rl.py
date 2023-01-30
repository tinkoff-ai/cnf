import argparse

import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help='env name to train on')
    args = parser.parse_args()
    return args


parameters = {
    'n_layers': {'values': [3, 4]},
    'batch_size': {'values': [256, 512, 1024]},
    'awac_lambda': {'value': 1.0 / 3.0},
    'seed': {'values': [0, 1, 2]},

    'hidden_size': {'value': 256},
    'lr': {'value': 3e-4},
    'gamma': {'value': 0.99},
    'tau': {'value': 2e-3},
    'n_train_steps': {'value': 1_000_000 + 1},  # +1 just to do one additional evaluation after training
    'eval_freq': {'value': 5000},
    'log_freq': {'value': 250},
    'n_test_episodes': {'value': 10},
}


sweep_config = {
    'method': 'grid',
    'metric': {'name': 'test_mean_rewards', 'goal': 'maximize'},
    'parameters': parameters
}


def main():
    args = parse_args()
    parameters['env_name'] = {'value': args.env_name}
    sweep_config['name'] = f'flow_rl_{args.env_name}'
    parameters['flow_checkpoint_path'] = {'value': f'flows_checkpoints/{args.env_name}.pt'}
    wandb.sweep(sweep_config, project='flows')


if __name__ == '__main__':
    main()
