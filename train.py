import argparse
import torch
import torch.nn as nn
import torch.utils.data as tud
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll

from models.folktune_vae import *
from models.folkrnn import *

from dataset import *


# def main(args):
torch.multiprocessing.set_sharing_strategy("file_system")
torch.manual_seed(0)

# load dataset
dataset = FolkDataset(
    data_file="./data/dataset.txt",
    max_sequence_length=256,
)

dataset_train, dataset_val = tud.random_split(dataset, [0.9, 0.1])

batch_size = 64
train_loader = tud.DataLoader(
    dataset_train, shuffle=True, batch_size=batch_size, num_workers=8
)
val_loader = tud.DataLoader(
    dataset_val, shuffle=False, batch_size=batch_size, num_workers=8
)

# load model
model = FolktuneVAE(
    input_size=dataset.vocab_size,
    embedding_size=512,
    hidden_size=256,
    num_layers=6,
    latent_size=64,
    output_size=dataset.vocab_size,
    dropout=0.2,
    pad_index=dataset.pad_idx,
)
'''
# load model
model = FolkRNN(
    vocab_size=dataset.vocab_size,
    hidden_size=512,
    num_layers=3,
    dropout=0.5,
    pad_index=dataset.pad_idx,
)
'''

# logger
logger = pll.TensorBoardLogger("tb_logs", name="bigger-folktuneVAE")

# training
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=200,
    logger=logger,
    gradient_clip_val=1,
    callbacks=[
        # Early stopping
        plc.EarlyStopping(
            monitor="val/val_loss",
            mode="min",
            patience=5,
            verbose=True,
            strict=True,
            min_delta=1e-4,
        ),
        # saves top-K checkpoints based on "val_loss" metric
        plc.ModelCheckpoint(
            save_top_k=1,
            monitor="val/val_loss",
            mode="min",
        ),
        plc.ModelCheckpoint(
            save_top_k=1,
            monitor="epoch",
            mode="max",
            filename="last",
        ),
    ],
)

trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

