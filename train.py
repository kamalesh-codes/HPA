import os
import torch
from tqdm import tqdm
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
    model = get_model(cfg).to(device)
    ddp_model = DDP(model,device_ids=[rank])

    #loss ,optimizer & metric definition
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(ddp_model.parameters(),
                                lr=cfg.train.lr,
                                weight_decay=1e-4)
    if dist.get_rank()==0:
        f1_metric = MultilabelF1Score(num_labels=cfg.data.num_class,
                                average="macro",
                                multidim_average="global",
                                sync_on_compute=False)
        f1_metric = f1_metric.to(device)
    
    train_loader = get_train_loader(cfg)

    accumulation_step = cfg.train.accumulation_step

    for epoch in range(1,cfg.train.epochs+1):

        if dist.get_rank()==0:
            pbar = tqdm(total = len(train_loader)*2,unit="batch")
            pbar.set_description(f"Epoch [{epoch}/{cfg.train.epochs}]")
            f1_metric.reset()

        ddp_model.train()
        global_running_loss = torch.tensor(0.0,device=device)
        global_total_samples = torch.tesnor(0,device=device)
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx,(image,target) in enumerate(train_loader,start=1):

            local_batch_loss = torch.tensor(0.0,device=device)
            local_batch_samples = torch.tesnor(0,device=device)
            image,label = image.to(device,non_blocking=True),label.to(device,non_blocking=True)

            output = model(image)
            loss = criterion(output,target)
            (loss/accumulation_step).backward()


            if (batch_idx%accumulation_step==0):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            local_batch_loss += loss.detach()*image.Size(0)
            local_batch_samples += image.Size(0)

            dist.all_reduce(local_batch_loss,op=dist.ReduceOp.SUM)
            dist.all_reduce(local_batch_samples,op=dist.ReduceOp.SUM)

            global_running_loss += local_batch_loss
            global_total_samples += local_batch_samples

            if dist.get_rank()==0:
                f1_metric.update(output,target)
                pbar.update(2)
                pbar.set_postfix(loss = (global_running_loss/global_total_samples).item())
        
        if dist.get_rank()==0:
            macro_f1 = f1_metric.compute().item()
            pbar.set_postfix(loss = (global_running_loss/global_total_samples).item(),
                         macro_f1 = macro_f1)
            pbar.close()



@hydra.main(version_base=None,config_path="configs",config_name="config")
def main(cfg:DictConfig):

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    train_model(cfg,local_rank)

    dist.destroy_process_group()


if __name__ == "__main__":

    main()