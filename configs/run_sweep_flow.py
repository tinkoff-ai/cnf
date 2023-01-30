import os
import wandb

from cnf.fix_random_seeds import fix_random_seeds
from cnf.train_flow import run

seed = 42
count = 5
log_dir = 'cnf_sweep_flows/'
sweep_id = None  # fill sweep id here before running this script


def train():
    fix_random_seeds(seed)
    with wandb.init() as logger:
        log_dir_ = log_dir + wandb.config['env_name'] + '/' + wandb.run.name
        os.makedirs(log_dir_ + '/checkpoints', exist_ok=True)
        run(log_dir_, wandb.config, logger=logger)


def main():
    wandb.agent(
        sweep_id, function=train,
        project='flows',
        count=count
    )


if __name__ == '__main__':
    main()
