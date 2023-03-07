import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch import seed
import warnings
warnings.filterwarnings("ignore")

from configs import TrainConfig
from jerex import model, util

cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig)

def seed_everything(seed):
    import random
    import os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

@hydra.main(config_name='train', config_path='configs/docred')
def train(cfg: TrainConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    util.config_to_abs_paths(cfg.datasets, 'train_path', 'valid_path', 'test_path', 'types_path')
    util.config_to_abs_paths(cfg.model, 'tokenizer_path', 'encoder_path')
    util.config_to_abs_paths(cfg.misc, 'cache_path')
    seed_everything(100)
    model.train(cfg)


if __name__ == '__main__':
    train()
