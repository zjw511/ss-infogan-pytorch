import torch
from utils import save_config
from trainer import Trainer
from config import get_config
from data_loader import get_loader
from metrics import *


def main(config):
    save_config(config)
    dataloader_unlabel = get_loader(
        config.batch_size, config.project_root, config.dataset)
    trainer = Trainer(config, dataloader_unlabel)
    trainer.train()
    return


if __name__ == "__main__":
    config, unparsed = get_config()
    #for i in range(4):
    main(config)
