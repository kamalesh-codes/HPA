import wandb
import torch

run = wandb.init(project="demo-check",name="new-run",id="new_run",resume="allow")
print(run.id,run.name)

for i in range(10):
    wandb.log({
        "loss":torch.rand(1).item()
    })

wandb.finish()