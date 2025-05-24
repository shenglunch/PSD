import argparse
from util import get_config
from train_manager import Trainer as manager

parser = argparse.ArgumentParser(description='DCTransformer')
parser.add_argument("--cfg", type=str, default='configs/cfg.yml')
args = parser.parse_args()

if __name__ == "__main__":
    cfg = get_config(args)
    tm = manager(cfg)
    tm.run()