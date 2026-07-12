import os
from datetime import datetime
import torch
from tqdm import tqdm
from torchvision.ops import sigmoid_focal_loss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.classification import MultilabelF1Score

from model import get_model
from loaders import get_train_loader

import hydra
from omegaconf import DictConfig



def train_model(cfg: DictConfig, device:torch.device):
    
    #Model creation
    model = get_model(cfg).to(device)
    ddp_model = DDP(model,device_ids=[device.index])

    #optimizer & metric definition
    optimizer = torch.optim.Adam(ddp_model.parameters(),
                                lr=cfg.train.lr,
                                weight_decay=1e-4)
    f1_metric = MultilabelF1Score(num_labels=cfg.data.num_class,
                                average="macro",
                                multidim_average="global")
    f1_metric = f1_metric.to(device)
    
    train_loader = get_train_loader(cfg)

    accumulation_step = cfg.train.accumulation_step

    for epoch in range(1,cfg.train.epochs+1):

        train_loader.sampler.set_epoch(epoch)
        f1_metric.reset()
        if dist.get_rank()==0:
            pbar = tqdm(total = len(train_loader)*torch.cuda.device_count(),unit="batch",ncols=100)
            pbar.set_description(f"Epoch [{epoch}/{cfg.train.epochs}]")

        ddp_model.train()
        global_running_loss = torch.tensor(0.0,device=device)
        global_total_samples = torch.tensor(0,device=device)
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx,(image,target) in enumerate(train_loader,start=1):

            local_batch_loss = torch.tensor(0.0,device=device)
            local_batch_samples = torch.tensor(0,device=device)
            image,target = image.to(device,non_blocking=True),target.to(device,non_blocking=True).to(torch.float32)

            output = ddp_model(image)
            loss = sigmoid_focal_loss(output,target,
                                      alpha=cfg.train.alpha,
                                      gamma=cfg.train.alpha,
                                      reduction="mean")
            (loss/accumulation_step).backward()

            f1_metric.update(output,target)

            if (batch_idx%accumulation_step==0):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            local_batch_loss += loss.detach()*image.size(0)
            local_batch_samples += image.size(0)

            dist.all_reduce(local_batch_loss,op=dist.ReduceOp.SUM)
            dist.all_reduce(local_batch_samples,op=dist.ReduceOp.SUM)

            global_running_loss += local_batch_loss
            global_total_samples += local_batch_samples

            if dist.get_rank()==0:
                pbar.update(2)
                pbar.set_postfix_str(f"loss:{(global_running_loss/global_total_samples).item():.4f}")
        
        f1 = f1_metric.compute().item()
        if dist.get_rank()==0:
            pbar.set_postfix_str(f"macro-f1:{f1:.4f} loss:{(global_running_loss/global_total_samples).item():.4f}")
            pbar.close()
        break

    if dist.get_rank()==0:
        now = datetime.now()
        dt = now.strftime("%d-%m-%H:%M")
        torch.save(ddp_model.state_dict(),f"model-epoch:{cfg.train.epochs}-{dt}.pth")


@hydra.main(version_base=None,config_path="configs",config_name="config")
def main(cfg:DictConfig):

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(type="cuda",index=local_rank)
    dist.init_process_group(backend="nccl",device_id=device)

    torch.cuda.set_device(local_rank)

    train_model(cfg,device)

    dist.destroy_process_group()


if __name__ == "__main__":

    main()