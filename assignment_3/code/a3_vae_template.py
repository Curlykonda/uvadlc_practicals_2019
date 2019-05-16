import argparse
import numpy as np
from pathlib import Path
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist
import sys
sys.path.append("..")

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, in_dim=784):
        super().__init__()

        #Architecture according to Kingma & Welling
        #tanh were replaced by ReLU
        self.fc1 = nn.Linear(in_dim, hidden_dim)

        self.mean_mlp = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )


        self.sigma_mlp = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
            nn.Softplus() #assures that std is non-negative
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        #mean, std = None, None
        #raise NotImplementedError()

        mean = self.mean_mlp(input)
        std = self.sigma_mlp(input)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, out_dim=784):
        super().__init__()

        self.decode_mlp = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self.decode_mlp(input)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, data_dim=784, device='cpu'):
        super().__init__()

        self.z_dim = z_dim
        self.data_dim = data_dim
        self.eps = 1e-8 #computational stability
        self.device = device
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def compute_elbo(self, input, recon, mu, std):

        std = std + self.eps
        recon = recon + self.eps

        #reconstruction loss: neg log likelihood Bernoulli
        #See Kingma & Welling Appendix C1
        recon_loss = -1 * torch.sum(input * torch.log(recon) + (1-input) * torch.log(1 - recon), dim=1)
        #BCE = F.binary_cross_entropy(recon, input, reduction='sum')

        #regularisation loss: KL-divergence approx posterior and prior, Gaussian case
        #See Kingma & Welling Appendix B
        var = std.pow(2)
        #reg_loss = 0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var, dim=1)
        reg_loss = 0.5 * torch.sum(mu.pow(2) + var - 1 - torch.log(var), dim=1)
        return torch.mean(recon_loss + reg_loss, dim=0)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        #encode
        mean, std = self.encoder(input)

        #reparameterise
        epsilon = torch.randn_like(std).to(self.device) # noise from unit Gaussian
        z = mean + std * epsilon
        z.to(self.device)

        #decode
        recon = self.decoder(z) #shape: batch_size x data_dim

        #compute elbo
        average_negative_elbo = self.compute_elbo(input, recon, mean, std)

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        #sample from unit Gaussian
        z = torch.randn(n_samples, self.z_dim).to(self.device)

        #decode latent vectors
        sampled_ims = self.decoder(z)
        print(f"Shape sampled ims:{sampled_ims.shape}")

        #compute means
        im_means = sampled_ims.mean(dim=0)

        #reshape samples
        im_dim = int(np.sqrt(self.data_dim))
        sampled_ims = sampled_ims.view(-1,1,im_dim,im_dim)
        im_means = im_means.view(1,1,im_dim,im_dim)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = 0

    for i, mb in enumerate(data):

        #reshape mini-batch
        mb = mb.reshape(-1, model.data_dim).to(model.device)
        #forward pass
        elbo = model(mb)

        #Perform backward pass in training mode
        if model.training:
            optimizer.zero_grad()
            elbo.backward()
            optimizer.step()

        average_epoch_elbo += elbo.item()

    return average_epoch_elbo / i


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)

def save_vae_samples(model, epoch, path, n_row=5):
    with torch.no_grad():
        sample_ims, im_means = model.sample(n_row**2)
        #transform samples to grid represenation
        sample_ims = make_grid(sample_ims, nrow=n_row)

        #format sampled images
        #sample.view(n_samples, 1, 28, 28)
        samples = sample_ims.cpu().numpy().transpose(1,2,0)

        #save samples
        file_name = (f"sample_{epoch}.png")
        plt.imsave(path + "/" + file_name, samples)
        print(f"Saved {file_name}\n")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = bmnist()[:2]  # ignore test split

    #initialise VAE model
    model = VAE(z_dim=ARGS.zdim, device=device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    # make the results directory
    #res_path = Path.cwd() / 'out_vae'
    #res_path.mkdir(parents=True, exist_ok=True)
    res_path = "./out_vae"
    os.makedirs(res_path, exist_ok=True)

    #sample from VAE
    save_vae_samples(model, epoch=0, path=res_path)
    train_curve, val_curve = [], []
    print(f"Start training for {ARGS.epochs} epochs")

    for epoch in range(1, ARGS.epochs+1):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo:.4f} val_elbo: {val_elbo:.4f}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        # --------------------------------------------------------------------
        save_vae_samples(model, epoch, res_path)

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2).
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
