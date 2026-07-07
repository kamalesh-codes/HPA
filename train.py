import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.classification import MultilabelF1Score

from model import get_model
from loaders import get_train_loader

import hydra
from omegaconf import DictConfig



def train_model(cfg: DictConfig, rank:int):
    
    #Model creation
    device = torch.device(type="cuda",index=rank)
    model = get_model().to(device)
    ddp_model = DDP(model,device_ids=[rank])

    #loss ,optimizer & metric definition
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(ddp_model.parameters(),
                                lr=cfg.train.lr,
                                weight_decay=1e-4)
    if dist.get_rank()==0:
        f1_metric = MultilabelF1Score(num_labels=cfg.data.num_class,
                                average="macro",
                                multidim_average="global")
        f1_metric = f1_metric.to(device)
    
    train_loader = get_train_loader()

    accumulation_step = cfg.train.accumulation_step

    for epoch in range(1,cfg.train.epochs+1):

        if dist.get_rank()==0:
            f1_metric.reset()
        ddp_model.train()
        epoch_running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx,(image,target) in enumerate(train_loader,start=1):

            image,label = image.to(device,non_blocking=True),label.to(device,non_blocking=True)

            output = model(image)
            loss = criterion(output,target)
            loss = loss/accumulation_step
            loss.backward()

            if dist.get_rank()==0:
                f1_metric.update(output,target)

            if (batch_idx%accumulation_step==0):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_running_loss += torch.abs_(loss.detach())
        
        epoch_running_loss/=len(train_loader)
        epoch_running_loss = epoch_running_loss.item()
        macro_f1 = f1_metric.compute().item()

    print(f"Epoch [{epoch}/{cfg.train.epochs}] |"
            f"Train Loss: {epoch_running_loss:.5f}  Macro F1: {macro_f1:.5f}")




@hydra.main(version_base=None,config_path="configs",cofig_name="config")
def main(cfg:DictConfig):

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ("LOCAL_RANK"))
    torch.cuda.set_device(local_rank)

    train_model(cfg,local_rank)

    dist.destroy_process_group()


if __name__ == "__main__":

    main()