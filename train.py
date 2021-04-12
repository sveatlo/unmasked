import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskedCelebADataset
from model import weights_init_normal
from model.generator import Generator
from model.discriminator import Discriminator

if torch.cuda.is_available():
    print("working with cuda")
else:
    print("stuck with cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10
LEARNING_RATE = 8e-5
BATCH_SIZE=16
IMAGE_SIZE=128
MASK_SIZE=64
CHANNELS_IMG=3
ADAM_B1 = 0.5
ADAM_B2 = 0.999

transforms = [
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)],
    ),
]

dataset = MaskedCelebADataset("dataset/celeba", transform=transforms)
dataset_test = MaskedCelebADataset("dataset/celeba", transform=transforms, mode="test")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

generator = Generator().to(device)
discriminator = Discriminator().to(device)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# loss
adversarial_loss = torch.nn.MSELoss().to(device)
pixelwise_loss = torch.nn.L1Loss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(ADAM_B1, ADAM_B2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(ADAM_B1, ADAM_B2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
patch_h, patch_w = int(MASK_SIZE / 2 ** 3), int(MASK_SIZE / 2 ** 3)
patch = (1, patch_h, patch_w)
sample_interval = 500
writer = SummaryWriter()
for epoch in range(NUM_EPOCHS):
    for i, (img, img_masked, masked_part) in enumerate(dataloader):
        valid = Variable(Tensor(img.shape[0], *patch).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(img.shape[0], *patch).fill_(0.0), requires_grad=False)

        # Configure input
        imgs = Variable(img.type(Tensor))
        masked_imgs = Variable(img_masked.type(Tensor))
        masked_parts = Variable(masked_part.type(Tensor))

        ## Train Generator ##
        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_parts = generator(masked_imgs)

        # Adversarial and pixelwise loss
        g_adv = adversarial_loss(discriminator(gen_parts), valid)
        g_pixel = pixelwise_loss(gen_parts, masked_parts)
        # Total loss
        g_loss = 0.001 * g_adv + 0.999 * g_pixel

        g_loss.backward()
        optimizer_G.step()

        ## Train Discriminator ##
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(masked_parts), valid)
        fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        # Generate sample at sample interval
        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {i}/{len(dataloader)} \
                  Loss D: {d_loss:.4f}, loss G: {g_loss:.4f}"
            )

            writer.add_scalar("pixel_loss", g_loss, batches_done)
            with torch.no_grad():
                test_img = None
                for i, (img, img_masked, masked_part) in enumerate(dataloader_test):
                    test_img = Variable(img_masked.type(Tensor))
                    break
                fake = generator(test_img)

                writer.add_image("fake", torchvision.utils.make_grid(fake, normalize=True), batches_done)
