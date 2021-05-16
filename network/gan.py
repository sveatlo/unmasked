import multiprocessing

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from network.generator import Generator
from network.discriminator import Discriminator
from dataset import MaskedCelebADataset


class WGANGP(LightningModule):
    def __init__(self, latent_dim: int = 100, lr: float = 0.0002, b1: float = 0.5, b2: float = 0.999, batch_size: int = 64, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size

        # networks
        mnist_shape = (3, 128, 128)
        self.generator = Generator(latent_dim=self.latent_dim, img_shape=mnist_shape)
        self.discriminator = Discriminator(img_shape=mnist_shape)

        self.validation_z = torch.randn(8, self.latent_dim)

        self.example_input_array = torch.zeros(2, self.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)

        lambda_gp = 10

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            #  sample_imgs = self.generated_imgs[:6]
            #  grid = torchvision.utils.make_grid(sample_imgs)
            #  self.logger.experiment.add_image('generated_images', grid, batch_idx)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            g_loss = -torch.mean(self.discriminator(self(z)))
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        elif optimizer_idx == 1:
            fake_imgs = self(z)

            # Real images
            real_validity = self.discriminator(imgs)
            # Fake images
            fake_validity = self.discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        n_critic = 5

        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return (
            {'optimizer': opt_g, 'frequency': 1},
            {'optimizer': opt_d, 'frequency': n_critic}
        )

    def train_dataloader(self):
        #  transform = transforms.Compose([
        #      transforms.ToTensor(),
        #      transforms.Normalize([0.5], [0.5]),
        #  ])
        #  dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        #  return DataLoader(dataset, batch_size=self.batch_size)
        t = [
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(3)], [0.5 for _ in range(3)],
            ),
        ]
        dataset = MaskedCelebADataset("dataset/celeba", transform=t)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count())

    def on_epoch_end(self):
        z = self.validation_z.to(self.device)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
