import torch
import os
import tempfile

def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(type="cuda",index=local_rank)
    torch.distributed.init_process_group(backend="nccl",device_id=device)
    torch.cuda.set_device(local_rank)
    return device

def cleanup():

        torch.distributed.destroy_process_group()


def checkpoint(cfg,obj):

    project_root = os.path.dirname(os.path.abspath("checkpoint.pth"))
    with tempfile.NamedTemporaryFile(dir=project_root) as tmp:
        torch.save(obj,tmp.name)
        os.replace(tmp.name,"checkpoint.pth")

def load_checkpoint(cfg):

    if os.path.exists("checkpoint.pth"):
        obj = torch.load("checkpoint.pth",map_location="cpu")
        return obj
    else:
         return False

def is_main():
    return torch.distributed.get_rank()==0
