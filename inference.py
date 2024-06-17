import torch
import torch.nn as nn
import torch.utils.data as tud
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll

from models.folkrnn import *
from models.folktune_vae import *

from dataset import *
import random

torch.multiprocessing.set_sharing_strategy("file_system")
torch.manual_seed(0)
random.seed(0)
seed = 2394

# load dataset
dataset = FolkDataset(data_file="data/data_v2", max_sequence_length=256)

torch.manual_seed(seed)
random.seed(seed)
'''
########## FOLKRNN ########## 
# load model
model = FolkRNN.load_from_checkpoint(
    "folkRNN/version_X/checkpoints/epoch=20-step=6636.ckpt"
)
model.eval()

gen = model.inference(
    dataset.sos_idx,
    dataset.eos_idx,
    max_len=256,
    mode="topp",
    temperature=1,
    PK=0.9,
)
########## ########## ########## 
'''
########## FOLKTUNE VAE ########## 

# load model
model = FolktuneVAE.load_from_checkpoint(
    "foltune_vae/version_X/checkpoints/epoch=20-step=6636.ckpt"
)
model.eval()

gen = model.inference(
    dataset.sos_idx,
    dataset.eos_idx,
    max_len=args.max_sequence_length,
    mode=args.mode,
    temperature=args.temperature,
    PK=args.pk,
    prompt=prompt,
    z=z,
)


gen = [dataset.i2w[str(g)] for g in gen[1:-1]]
print(f"X:0\n{gen[0]}\n{gen[1]}\n{''.join(gen[2:])}")
