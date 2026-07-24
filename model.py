import torch.nn as nn
from torchvision.models import resnet34
from omegaconf import DictConfig


def get_model(cfg:DictConfig):

    model = resnet34()
    model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Sequential(
                        nn.Linear(in_features=model.fc.in_features,out_features=128),
                        nn.ReLU(),
                        nn.BatchNorm1d(num_features=128),
                        nn.Linear(in_features=128,out_features=cfg.data.num_class))
    
    return model