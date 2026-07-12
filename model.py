import torch.nn as nn
from torchvision.models import densenet121
from omegaconf import DictConfig


def get_model(cfg:DictConfig):

    model = densenet121()
    model.features[0] = nn.Conv2d(4,64,kernel_size=(7,7),
                                  stride=(2,2),padding=(3,3),
                                  bias=False)
    model.classifier = nn.Sequential(
                        nn.Linear(in_features=model.classifier.in_features,out_features=512),
                        nn.ReLU(),
                        nn.BatchNorm1d(num_features=512),
                        nn.Dropout(p=0.7),
                        nn.Linear(in_features=512,out_features=128),
                        nn.ReLU(),
                        nn.BatchNorm1d(num_features=128),
                        nn.Dropout(p=0.7),
                        nn.Linear(in_features=128,out_features=cfg.data.num_class))
    
    return model