import hydra
from omegaconf import DictConfig
from loaders import get_train_loader

@hydra.main(version_base=None,config_path="configs",config_name="config")
def main(cfg:DictConfig):

    loader = get_train_loader(cfg)
    print(next(iter(loader)).shape)

main()