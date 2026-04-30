"""
Entrypoint: build all components and launch training.
"""
import argparse

from agent.ddpg import DDPGAgent
from config import Config
from environment.dataloader import DataLoader
from environment.env import Env
from environment.evaluator import Evaluator
from environment.trainer import Trainer
from utils.noise import OUNoise


def main() -> None:
    parser = argparse.ArgumentParser(description="IAC Group Recommender — DDPG training")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional path to a YAML config file (e.g. default.yaml)")
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()

    dataloader = DataLoader(config)
    rating_matrix_train = dataloader.load_rating_matrix(dataset_name='val')
    df_eval_user_test = dataloader.load_eval_data(mode='user', dataset_name='test')
    df_eval_group_test = dataloader.load_eval_data(mode='group', dataset_name='test')

    env = Env(config=config, rating_matrix=rating_matrix_train, dataset_name='val')
    noise = OUNoise(config=config)
    agent = DDPGAgent(config=config, noise=noise,
                      group2members_dict=dataloader.group2members_dict, verbose=True)
    evaluator = Evaluator(config=config)

    trainer = Trainer(config=config, env=env, agent=agent, evaluator=evaluator,
                      df_eval_user=df_eval_user_test, df_eval_group=df_eval_group_test)
    trainer.train()


if __name__ == '__main__':
    main()

