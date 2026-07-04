import torch
import torch.distributed as dist
from model import get_model
from loaders import get_train_loader
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None,config_path="configs",cofig_name="config")
def main(cfg:DictConfig):

    dist.init_process_group(backend="nccl")

    model = get_model(cfg)







    dist.destroy_process_group()


if __name__ == "__main__":

    main()