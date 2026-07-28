import time
from torch import rand
from tqdm import tqdm 
import random

pbar = tqdm(total=100,unit="batch")
for i in range(100):
    time.sleep(0.001+0.1*random.random())
    pbar.update(1)
    pbar.set_postfix_str(f"              loss {random.random()}")

pbar.close()