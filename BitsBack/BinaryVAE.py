# -*- coding: utf-8 -*-
"""

Code for training a VAE with a bernoulli likelihood on binarized MNIST.

Based on :

https://github.com/bits-back/bits-back/blob/master/torch_vae/tvae_binary.py

and

https://github.com/pytorch/examples/blob/master/vae/main.py


"""

__authors__ = ["Deepak Sharma"]

from typing import Tuple, List, Union, Dict
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.distributions import Normal, Bernoulli
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='VAE with a bernoulli likelihood')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--hidden-dim', type=int, default=100,
                    help='hidden dimension size of encoder and decoder')
parser.add_argument('--learning-rate', type=int, default=-3,
                    help='exponent of the learning rate e^(input) (default = -3)')
parser.add_argument('--latent-dim', type=int, default=40,
                    help='dimensions of latent space Z')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--randomize', action='store_true', default=False,
                    help='randomize pixel values according to Bernoulli(pixels) (default: False)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

class Randomise:
    def __call__(self, pic):
        return Bernoulli(pic).sample()

class Round:
    def __call__(self, pic):
        return torch.round(pic)

class BinaryVAE(nn.Module):
    def __init__(self, data_size :int = 784, n_channels :int = 1, hidden_dim: int = 100, latent_dim: int = 40):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.data_size = data_size
        self.n_channels = n_channels

        self.fc1 = nn.Linear(self.data_size, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.data_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.data_size * self.n_channels))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss(self, x):
        recon_x, mu, logvar = self.forward(x)
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.data_size * self.n_channels), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def sample(self, n = 64):
        z = torch.randn(n, self.latent_dim)
        x_recon, _, __ = self.decode(z)
        return x_recon


def train(model, epoch, data_loader, optimizer, log_interval=10):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = model.loss(data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(data_loader.dataset)))


def test(model, epoch, data_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            data = data.to(device)
            recon_x, mu, logvar = model(data)

def run():

    model = BinaryVAE(784, 1, args.hidden_dim, args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=torch.exp(args.learning_rate)).to(device)

    if args.randomize:
        binariser = Randomise()
    else:
        binariser = Round()
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(), binariser])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=False, download=True,
                       transform=transforms.Compose([transforms.ToTensor(), binariser])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    try:
        for epoch in range(1, args.epochs + 1):
            train(model, epoch, train_loader, optimizer)
            test(model, epoch, test_loader)
            sample = model.sample(epoch)
            save_image(sample.cpu().view(64, 1, 28, 28),
                         'results/sample_' + str(epoch) + '.png')
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'saved_params/torch_binary_vae_params_new')
    torch.save(model.state_dict(), 'saved_params/torch_binary_vae_params_new')

if __name__ == "__main__":
    run()
