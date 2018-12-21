#from comet_ml import Experiment
import argparse
import torch
import torch.utils.data
from torch import nn, optim, lgamma
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.distributions import Normal, Categorical, Beta, Binomial
from torchvision.utils import save_image
import numpy as np

parser = argparse.ArgumentParser(description='VAE with a beta-binomial likelihood')
parser.add_argument('--batch-size', type=int, default=100,
                    help='input batch size for training (default: 100)')
parser.add_argument('--model-filename-prefix', type=str, default="torch_new_bb",
                    help='path to save the model params  (default: torch_new_bb)')
parser.add_argument('--hidden-dim', type=int, default=100,
                    help='hidden dimension size of encoder and decoder')
parser.add_argument('--learning-rate', type=int, default=-4,
                    help='exponent of the learning rate e^(input) (default = -4)')
parser.add_argument('--latent-dim', type=int, default=20,
                    help='dimensions of latent space Z')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--seed', type=int, default=17,
                    help='random seed (default: 17)')

args = parser.parse_args()
model_filename_suffix = "{}_{}_{}_{}_{}".format(args.batch_size, args.hidden_dim, args.latent_dim, args.epochs, abs(args.learning_rate))
model_filename = "{}_{}".format(args.model_filename_prefix, model_filename_suffix)

#experiment = Experiment(api_key="Gncqbz3Rhfy3MZJBcX7xKVJoo", project_name="bits-back", workspace="deepak-sharma-mail-mcgill-ca")
#experiment.log_multiple_params(vars(args))

torch.manual_seed(args.seed)

def beta_binomial_log_pdf(k, n, alpha, beta):
    numer = lgamma(n+1) + lgamma(k + alpha) + lgamma(n - k + beta) + lgamma(alpha + beta)
    denom = lgamma(k+1) + lgamma(n - k + 1) + lgamma(n + alpha + beta) + lgamma(alpha) + lgamma(beta)

    #print("numer autograd = {}".format(numer))
    #print("numer  = {}".format(np.where(torch.isinf(numer) == 1)))
    #print("denom  = {}".format(np.where(torch.isinf(denom) == 1)))
    #print("numer - denom = {}".format(np.where(torch.isnan(numer - denom) == 1)))

    return numer - denom


class BetaBinomialVAE(nn.Module):
    def __init__(self, hidden_dim=200, latent_dim=50):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.register_buffer('prior_mean', torch.zeros(1))
        self.register_buffer('prior_std', torch.ones(1))
        self.register_buffer('n', torch.ones(100, 784) * 255.)

        self.fc1 = nn.Linear(784, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)

        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.bn21 = nn.BatchNorm1d(self.latent_dim)
        self.bn22 = nn.BatchNorm1d(self.latent_dim)

        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, 784*2)

    def encode(self, x):
        """Return mu, sigma on latent"""
        h = x / 255.  # otherwise we will have numerical issues
        h1 = self.fc1(h)
        b1 = self.bn1(h1)
        h = F.leaky_relu(self.bn1(self.fc1(h)) , negative_slope = 0.001)
        return self.bn21(self.fc21(h)), torch.exp(self.bn22(self.fc22(h)))

    def reparameterize(self, mu, std):
        if self.training:
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = F.leaky_relu(self.bn3(self.fc3(z)), negative_slope = 0.001)
        h = self.fc4(h)
        log_alpha, log_beta = torch.split(h, 784, dim=1)
        return torch.exp(log_alpha), torch.exp(log_beta)

    def loss(self, x):
        z_mu, z_std = self.encode(x.view(-1, 784))
        z = self.reparameterize(z_mu, z_std)  # sample zs

        x_alpha, x_beta = self.decode(z)
        l = beta_binomial_log_pdf(x.view(-1, 784), self.n,
                                  x_alpha, x_beta)
        l = torch.sum(l, dim=1)

        #print("value of l = {}".format(l))
        p_z = torch.sum(Normal(self.prior_mean, self.prior_std).log_prob(z), dim=1)
        q_z = torch.sum(Normal(z_mu, z_std).log_prob(z), dim=1)
        return -torch.mean(l + p_z - q_z) * np.log2(np.e) / 784.

    def sample(self, device, epoch, num=64):
        sample = torch.randn(num, self.latent_dim).to(device)
        x_alpha, x_beta = self.decode(sample)
        beta = Beta(x_alpha, x_beta)
        p = beta.sample()
        binomial = Binomial(255, p)
        x_sample = binomial.sample()
        x_sample = x_sample.float() / 255.
        save_image(x_sample.view(num, 1, 28, 28),
                   'results/epoch_{}_samples.png'.format(epoch))

    def reconstruct(self, x, device, epoch):
        x = x.view(-1, 784).float().to(device)
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)  # sample zs
        x_alpha, x_beta = self.decode(z)
        beta = Beta(x_alpha, x_beta)
        p = beta.sample()
        binomial = Binomial(255, p)
        x_recon = binomial.sample()
        x_recon = x_recon.float() / 255.
        x_with_recon = torch.cat((x, x_recon))
        save_image(x_with_recon.view(64, 1, 28, 28),
                   'results/epoch_{}_recon.png'.format(epoch))


def train(model, device, epoch, data_loader, optimizer, log_interval=10):
    model.train()
    losses = []
    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = model.loss(data)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, np.mean(losses)))


def test(model, device, epoch, data_loader):
    model.eval()
    losses = []
    for data, _ in data_loader:
        data = data.to(device)
        loss = model.loss(data)
        losses.append(loss.item())
    print('\nEpoch: {}\tTest loss: {:.6f}\n\n'.format(
        epoch, np.mean(losses)
    ))
    #experiment.log_metric("training_loss", np.mean(losses), epoch)


if __name__ == '__main__':
    epochs = args.epochs
    batch_size = args.batch_size

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    class ToInt:
        def __call__(self, pic):
            return pic * 255

    transforms.Compose([transforms.ToTensor(), ToInt()])

    model = BetaBinomialVAE(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5*np.exp(args.learning_rate))

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(), ToInt()])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=False, download=True,
                       transform=transforms.Compose([transforms.ToTensor(), ToInt()])),
        batch_size=batch_size, shuffle=True, **kwargs)

    recon_dataset = datasets.MNIST('data/mnist', train=False, download=True,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 ToInt()])).test_data[:32]

    for epoch in range(1, epochs + 1):
        train(model, device, epoch, train_loader, optimizer)
        test(model, device, epoch, test_loader)
        model.reconstruct(recon_dataset, device, epoch)
        model.sample(device, epoch)
    torch.save(model.state_dict(), 'saved_params/{}'.format(model_filename))
