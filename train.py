import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import MaskedCelebADataset
from model import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 8e-5
BATCH_SIZE=16
IMAGE_SIZE=128
MASK_SIZE=64
CHANNELS_IMG=3

transforms = [
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)],
    ),
]

dataset = MaskedCelebADataset("dataset/celeba", transform=transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for batch_idx, (img, img_masked, masked_part) in enumerate(dataloader):
    print(batch_idx, img.shape, img_masked.shape, masked_part.shape)
