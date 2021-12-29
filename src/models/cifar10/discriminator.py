import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def _upsample(x):
    h, w = x.shape[2:]
    return F.interpolate(x, size=(h * 2, w * 2))


def upsample_conv(x, conv):
    return conv(_upsample(x))

def _downsample(x):
    return F.avg_pool2d(x, 2)


class disBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False):
        super(disBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)) # 谱归一化
        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        self.c2 = nn.utils.spectral_norm(nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
            nn.init.xavier_uniform_(self.c_sc.weight.data)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, input):
        return self.residual(input) + self.shortcut(input)

class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation
        self.c1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad))
        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        self.c2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
        nn.init.xavier_uniform_(self.c_sc.weight.data)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, input):
        return self.residual(input) + self.shortcut(input)

class Discriminator(nn.Module):
    def __init__(self, n_c_disc, dim_c_disc, dim_c_cont, ch=128, activation=F.relu):
        super(Discriminator, self).__init__()
        self.activation = activation
        self.dim_c_disc = dim_c_disc
        self.dim_c_cont = dim_c_cont
        self.n_c_disc = n_c_disc
        self.block1 = OptimizedBlock(3, ch)
        self.block2 = disBlock(ch, ch, activation=activation, downsample=True)
        self.block3 = disBlock(ch, ch, activation=activation, downsample=False)
        self.block4 = disBlock(ch, ch, activation=activation, downsample=False)
        self.l5 = nn.utils.spectral_norm(nn.Linear(ch, 1, bias=False))
        nn.init.xavier_uniform_(self.l5.weight.data)
        self.module_Q =nn.utils.spectral_norm(nn.Linear(128, 128))
        nn.init.xavier_uniform_(self.module_Q.weight.data)
        # n_c_disc
        if self.n_c_disc != 0:
            self.l_disc = nn.Linear(in_features=128, out_features=self.n_c_disc*self.dim_c_disc)
            self.soft_disc = nn.Softmax(dim=2)
            self.latent_disc = nn.Sequential(
                nn.Linear(
                    in_features=128, out_features=self.n_c_disc*self.dim_c_disc),
                Reshape(-1, 1,self.dim_c_disc),
                nn.Softmax(dim=2))
        self.latent_cont_mu = nn.Linear(
            in_features=128, out_features=self.dim_c_cont)
        nn.init.xavier_uniform_(self.latent_cont_mu.weight.data)

        # self.latent_cont_var = nn.Linear(
        #     in_features=128, out_features=self.dim_c_cont)
        # nn.init.xavier_uniform_(self.latent_cont_var.weight.data)

        self.initial()
    
    def initial(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        self.apply(weights_init)
        
    def forward(self, input):
        h = input
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = torch.sum(h, dim=(2,3))
        # D
        probability = self.l5(h).squeeze()
        # Q
        internal_Q = self.module_Q(h)
        c_cont_mu = self.latent_cont_mu(internal_Q)
        # c_cont_var = torch.exp(self.latent_cont_var(internal_Q))
        if self.n_c_disc != 0:
            c_disc_logits = self.latent_disc(internal_Q).squeeze()

            return probability, c_disc_logits, c_cont_mu
        else: 
            return probability, c_cont_mu

def weights_init(m):
    classname = m.__class__.__name__
    print(m)
    print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
