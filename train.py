import torch
import wandb
from tqdm import tqdm
from torchvision.ops import sigmoid_focal_loss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.classification import MultilabelF1Score

from model import get_model
from loaders import get_train_loader
from utils import load_checkpoint, setup,cleanup,checkpoint,is_main

import hydra
from omegaconf import DictConfig



def train_model(cfg: DictConfig, device:torch.device):

    obj = load_checkpoint()


    #Model creation
    model = get_model(cfg).to(device)

    if cfg.train.run_from_checkpoint :
        model.load_state_dict(obj["model"])

    ddp_model = DDP(model,device_ids=[device.index])

    #optimizer & metric definition
    optimizer = torch.optim.Adam(ddp_model.parameters(),
                                lr=cfg.train.lr,
                                weight_decay=1e-4)
    metric = MultilabelF1Score(num_labels=cfg.data.num_class,
                                average="macro",
                                multidim_average="global")
    metric = metric.to(device)

    starting_epoch = 1

    if cfg.train.run_from_checkpoint :
        optimizer.load_state_dict(obj["optimizer"])
        starting_epoch = obj["epoch"]+1


    if is_main():
        if cfg.train.run_from_checkpoint:
            run = wandb.init(project="HPAIC",
                            config=cfg.train,
                            id=obj["run_id"],
                            resume="must")
        else:
            run = wandb.init(project="HPAIC",
                            config=cfg.train,
                            name=cfg.train.run_name,
                            resume="never")
            
    train_loader = get_train_loader(cfg)

    accumulation_step = cfg.train.accumulation_step

    for epoch in range(starting_epoch,cfg.train.epochs+1):

        if is_main():
            pbar = tqdm(total = len(train_loader)*torch.cuda.device_count(),unit="batch")
            pbar.set_description(f"Epoch [{epoch}/{cfg.train.epochs}]")

        train_loader.sampler.set_epoch(epoch)
        metric.reset()
        ddp_model.train()
        optimizer.zero_grad(set_to_none=True)

        global_running_loss = torch.tensor(0.0,device=device)
        global_total_samples = torch.tensor(0,device=device)
        
        for batch_idx,(image,target) in enumerate(train_loader,start=1):

            local_batch_loss = torch.tensor(0.0,device=device)
            local_batch_samples = torch.tensor(0,device=device)
            image,target = image.to(device,non_blocking=True),target.to(device,non_blocking=True).to(torch.float32)


            output = ddp_model(image)       #forward pass
            loss = sigmoid_focal_loss(output,target,
                                      alpha=cfg.train.alpha,    #loss calculate 
                                      gamma=cfg.train.gamma,
                                      reduction="mean")
            (loss/accumulation_step).backward() #backward pass

            if (batch_idx%accumulation_step==0):    #weights update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)


            metric.update(output,target)

            local_batch_loss += loss.detach()*image.size(0)
            local_batch_samples += image.size(0)

            dist.all_reduce(local_batch_loss,op=dist.ReduceOp.SUM)
            dist.all_reduce(local_batch_samples,op=dist.ReduceOp.SUM)

            global_running_loss += local_batch_loss
            global_total_samples += local_batch_samples


            if is_main():
                pbar.update(2)
                pbar.set_postfix_str(f"loss:{(global_running_loss/global_total_samples).item():.4f}")

        f1 = metric.compute().item()
        global_loss = (global_running_loss/global_total_samples).item()
        if is_main():
            wandb.log({
                "train/loss":global_loss,
                "train/macro-f1":f1
            })
            pbar.set_postfix_str(f"macro-f1:{f1:.4f} loss:{global_loss:.4f}")
            pbar.close()

            obj = {"model":ddp_model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "epoch":epoch,
                "loss":global_loss,
                "run_id":run.id}
        
            checkpoint(obj,cfg)

    if is_main():
        wandb.finish()


@hydra.main(version_base=None,config_path="configs",config_name="config")
def main(cfg:DictConfig):

    device = setup()

    train_model(cfg,device)

    cleanup()


if __name__ == "__main__":

    main()