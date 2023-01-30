import wandb

from cnf.fix_random_seeds import fix_random_seeds
from cnf.train_rl import run

count = 3
log_dir = 'cnf_sweep/'
sweep_id = None  # fill sweep id here before running this script


def train():
    with wandb.init() as logger:
        fix_random_seeds(wandb.config['seed'])
        run(None, wandb.config, logger=logger)


def main():
    wandb.agent(
        sweep_id, function=train,
        project='flows',
        count=count
    )


if __name__ == '__main__':
    main()
