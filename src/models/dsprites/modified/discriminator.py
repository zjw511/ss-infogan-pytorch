import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

'''
Discriminator Model Definition
'''


class Discriminator(nn.Module):
    '''Shared Part of Discriminator and Recognition Model'''

    def __init__(self, n_c_disc, dim_c_disc, dim_c_cont):
        super(Discriminator, self).__init__()
        self.dim_c_disc = dim_c_disc
        self.dim_c_cont = dim_c_cont
        self.n_c_disc = n_c_disc
        # Shared layers
        self.module_shared = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=1,
                                    out_channels=32,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Reshape(-1, 64*4*4),
            spectral_norm(nn.Linear(in_features=64*4*4, out_features=128)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # Layer for Disciminating
        self.module_D = nn.Sequential(
            spectral_norm(nn.Linear(in_features=128, out_features=1)),
            nn.Sigmoid()
        )

        self.module_Q = nn.Sequential(
            spectral_norm(nn.Linear(in_features=128, out_features=128)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        if self.n_c_disc != 0:
            self.latent_disc = nn.Sequential(
                spectral_norm(nn.Linear(
                    in_features=128, out_features=self.n_c_disc*self.dim_c_disc)),
                Reshape(-1, self.n_c_disc, self.dim_c_disc),
                nn.Softmax(dim=2)
            )

        self.latent_cont = spectral_norm(nn.Linear(
            in_features=128, out_features=self.dim_c_cont))

    def forward(self, z):
        out = self.module_shared(z)
        probability = self.module_D(out)
        probability = probability.squeeze()
        internal_Q = self.module_Q(out)
        c_cont = self.latent_cont(internal_Q)

        if self.n_c_disc != 0:
            c_disc_logits = self.latent_disc(internal_Q)
            return probability, c_disc_logits, c_cont
        else:
            return probability, c_cont


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
