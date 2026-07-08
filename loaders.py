import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.distributed import DistributedSampler


class HPADataset(Dataset):

    def __init__(self,cfg:DictConfig,train:bool=True,transforms = None):
        
        super().__init__()

        self.train = train
        self.transform = transforms
        self.df = pd.read_csv(cfg.data.train_files_path if self.train else cfg.data.test_files_path).iloc[:100]
        self.data_path = cfg.data.train_data_path if self.train else cfg.data.test_data_path
        self.num_class = cfg.data.num_class
    
    def __len__(self)->int:
        return self.df.shape[0] 

    def __getitem__(self,idx):
        
        sample_name,label = self.df.iloc[idx]

        img_data = torch.empty(4,512,512,dtype=torch.float32)
        for i,color in enumerate(["green","blue","red","yellow"]):

            sample_color_path = self.data_path + "/" + sample_name + f"_{color}.png"
            img_data[i] = torchvision.io.read_image(sample_color_path)
        
        if self.transform:
            img_data = self.transform(img_data)

        if self.train:

            label = torch.tensor(tuple(map(int,label.strip().split())),dtype=torch.int32)
            mhe_label = torch.zeros(self.num_class,dtype=torch.uint8).scatter_(0,label,1)
            return (img_data/255,mhe_label)
        
        else:
            return img_data/255
        

def calculate_sample_weights(cfg:DictConfig):
    
    df = pd.read_csv(cfg.data.train_files_path)
    class_count = np.zeros(cfg.data.num_class)

    for row in df["Target"].str.split():
        for label in tuple(map(int,row)):
            class_count[label]+=1

    # sample weights is calculated by the average of class weights of target ,
    # it may become problem like the sampler favouring the samples that have high cardinality of class
    # we make sure it not happen and maintain the weights

    class_weights = 1/class_count
    class_weights = (class_weights/class_weights.sum())*100
    sample_weights = np.zeros(df.shape[0])

    for idx,row in enumerate(df["Target"].str.split()):
        
        row = tuple(map(int,row))
        for label in row:
            sample_weights[idx] += class_weights[label]

        sample_weights[idx] = sample_weights[idx]/len(row)
    
    return class_weights,sample_weights


def get_train_loader(cfg:DictConfig):

    trans = transforms.Compose([v2.RandomHorizontalFlip(p=0.5),
                                 v2.RandomVerticalFlip(p=0.5),
                                 v2.RandomRotation(180),
                                 v2.RandomAffine(degrees=0,scale=(0.9,1.1))])

    train_dataset = HPADataset(cfg,train=True,transforms=trans)

    sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(dataset = train_dataset,
                                batch_size = cfg.train.batch_size,
                                sampler=sampler,
                                num_workers=cfg.train.num_workers,
                                pin_memory=True,
                                prefetch_factor=cfg.train.prefetch)
    
    return train_loader

def get_test_loader(cfg:DictConfig):

    test_dataset = HPADataset(cfg,train=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=cfg.test.batch_size,
                             shuffle=False,
                             num_workers=cfg.test.num_workers,
                             pin_memory=True,
                             prefetch_factor=cfg.test.prefetch)
    return test_loader
