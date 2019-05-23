import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self, zdim=128, out_dim=784):
        super(Generator, self).__init__()

        #Receives random noise, output images by de-convolutions
        self.hidden_units = [128, 256, 512, 1024]
        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 784
        #   Output non-linearity
        self.generate = nn.Sequential(
            nn.Linear(zdim, self.hidden_units[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_units[0], self.hidden_units[1]),
            nn.BatchNorm1d(self.hidden_units[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_units[1], self.hidden_units[2]),
            nn.BatchNorm1d(self.hidden_units[2]),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_units[2], self.hidden_units[3]),
            nn.BatchNorm1d(self.hidden_units[3]),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_units[3], out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Tanh()
        )

    def forward(self, z):
        # Generate images from z
        output = self.generate(z)

        return output


class Discriminator(nn.Module):
    def __init__(self, in_dim=784):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.hidden_units = [512, 256]

        self.discriminate = nn.Sequential(
            nn.Linear(in_dim, self.hidden_units[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_units[0], self.hidden_units[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_units[1], 1),
            nn.Sigmoid() #outputs probability between 0 and 1
        )

    def forward(self, img):
        # return discriminator score for img
        score = self.discriminate(img)

        return score


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, res_path):

    #set device
    discriminator.to(device)
    generator.to(device)

    #initialise stats
    stats = {}
    stats["loss_G"] = []
    stats["loss_D"] = []
    stats["acc"] = []
    stats["score"] = []

    path_imgs = res_path + "images/"
    path_model = res_path + "models/"

    os.makedirs(path_imgs, exist_ok=True)
    os.makedirs(path_model, exist_ok=True)

    for epoch in range(args.n_epochs):

        t0 = time.time()
        acc = 0
        score = 0
        losses_G = []
        losses_D = []
        accs = []
        scores = []

        for i, (imgs, _) in enumerate(dataloader):

            #Reshape and send to device
            imgs = imgs.view(-1, 784).to(device)
            batch_size = imgs.shape[0]

            # Train Generator
            # ---------------
            for g_step in range(args.add_iter_G):
                #sample random noise
                z = torch.randn(batch_size, args.zdim).to(device)

                #forward pass
                fake_imgs = generator(z)

                #compute loss
                #adjusted objective: minimise neg expectation of log(D(G(z))
                loss_G = -1*torch.log(discriminator(fake_imgs)).sum()
                #maybe constraint: min < loss < max

                #backward pass
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()


            # Train Discriminator
            # -------------------
            #sample random noise
            z = torch.randn(batch_size, args.zdim).to(device)

            #forward pass
            score_real = discriminator(imgs)
            score_fake = discriminator(generator(z))
            m_score_fake = score_fake.mean()
            #compute loss
            #
            loss_D = -1*(torch.log(score_real) + torch.log(1-score_fake)).sum()

            #Avoid training D when it's too good
            if not(args.fix_D & (acc >= args.acc_thresh)):
                #backward pass
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

            # Save Stats
            # -----------
            real_pos = (score_real >= 0.5).sum().item() # True Positives
            fake_pos = (score_fake >= 0.5).sum().item() # False Positives
            acc = (real_pos + batch_size - fake_pos) / (2*batch_size) # Acc of D
            score = fake_pos / batch_size # Score of G

            losses_G.append(loss_G.item())
            losses_D.append(loss_D.item())
            accs.append(acc)
            scores.append(score)


            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(gen_imgs[:25],
                #            'images/{}.png'.format(batches_done),
                #            nrow=5, normalize=True)

                #save fake images
                save_image(generator(z[:25]).view(-1, 1, 28, 28),
                    path_imgs + "fake_{}.png".format(epoch),
                           nrow=5, normalize=True)

                #save real data from batch
                save_image(imgs[:25].view(-1, 1, 28, 28),
                           path_imgs + "real_{}.png".format(epoch),
                           nrow=5, normalize=True)
        #Print update
        t1 = time.time()
        stats["loss_G"].append(np.mean(losses_G))
        stats["loss_D"].append(np.mean(losses_D))
        stats["acc"].append(np.mean(accs))
        stats["score"].append(float(np.mean(scores)))

        print(f"====== Training Epoch {epoch+1}/{args.n_epochs} ======")
        print(f"Avg Generator loss: {stats['loss_G'][-1]:.4f}")
        print(f"Generator score: {stats['score'][-1]:.4f}\n")
        print(f"Avg Discriminator loss: {stats['loss_D'][-1]:.4f}")
        print(f"Discriminator acc: {stats['acc'][-1]:.4f}")
        print(f"Trained in {t1-t0:.4f}s\n")

        #save intermediate results
        torch.save(generator.state_dict(),
                   path_model + "generator.pt")

        torch.save(discriminator.state_dict(),
                   path_model + "discriminator.pt")

        np.save(path_model + "results", stats)
        print("Saved results")


    return stats

def print_setting(config):
    for key, value in vars(config).items():
        print("{0}: {1}".format(key, value))

def plot_results(results, path_plots):
    np.save(path_plots + "results", results)

    print("Results saved for plotting!")
    #

def interpolate_plot(generator, path_plots, steps=9):

    for idx in range(5):
        # sample random noise
        z = torch.randn(2, args.zdim)

        # interpolate
        interpolated = [torch.linspace(z[0][i], z[1][i], steps=steps)
                        for i in range(args.zdim)]
        interpolated = torch.stack(interpolated, dim=1).to(device)
        #print(interpolated.shape)

        # forward pass
        fake_imgs = generator(interpolated)

        save_image(fake_imgs.view(-1, 1, 28, 28),
                   path_plots + (f"interpolated_{idx}.png"),
                   nrow=steps, normalize=True
                   )

def main():
    # Create output image directory
    res_path = args.res_path
    os.makedirs(res_path, exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(zdim=args.zdim)
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2)) #betas: decay of first order momentum
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    print_setting(args)
    # Start training
    print("Start training")
    results = train(dataloader, discriminator, generator, optimizer_G, optimizer_D, res_path)
    print("Done training")

    #print results
    path_plots = res_path + "plots/"
    os.makedirs(path_plots, exist_ok=True)
    plot_results(results, path_plots)
    interpolate_plot(generator, path_plots, steps=9)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #training hyperparameters
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--zdim', type=int, default=100, help='dimensionality of the latent space')
    #optimiser
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    #Specifically for D
    parser.add_argument("--fix_D", type=bool, default=False, help="Fix D when it becomes too good")
    parser.add_argument("--acc_thresh", type=float, default=0.8, help="Acc threshold when to fix D")

    #Specifically for G
    parser.add_argument("--add_iter_G", type=int, default=1, help="Additional iteration for G training")

    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--res_path', type=str, default="./GAN/",
                        help='Path where to save results')
    args = parser.parse_args()

    main()
