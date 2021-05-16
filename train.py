import os

from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from network import WGANGP

def main(args: Namespace) -> None:
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = WGANGP(**vars(args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # If use distubuted training  PyTorch recommends to use DistributedDataParallel.
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
    trainer = Trainer(gpus=args.gpus)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help="number of GPUs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")

    hparams = parser.parse_args()

    main(hparams)
