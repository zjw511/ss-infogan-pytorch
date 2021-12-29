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


class genBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=F.relu, hidden_channels=None, ksize=3, pad=1, upsample=False):
        super(genBlock, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        self.b1 = nn.BatchNorm2d(in_channels)
        self.b2 = nn.BatchNorm2d(hidden_channels)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            nn.init.xavier_uniform_(self.c_sc.weight.data)

    def residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, input):
        return self.residual(input) + self.shortcut(input)

class Generator(nn.Module):
    def __init__(self, ch=256, dim_z=128, bottom_width=4, activation=F.relu, distribution="normal"):
        super(Generator, self).__init__()
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * ch)
        nn.init.xavier_uniform_(self.l1.weight.data)
        self.block2 = genBlock(ch, ch, activation=activation, upsample=True)
        self.block3 = genBlock(ch, ch, activation=activation, upsample=True)
        self.block4 = genBlock(ch, ch, activation=activation, upsample=True)
        self.b5 = nn.BatchNorm2d(ch)
        self.c5 = nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.c5.weight.data)
        self.initial()
    
    def initial(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('Linear') != -1:
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        self.apply(weights_init)

    def forward(self, input):
        h = input
        h0 = self.l1(h)
        h = h0.view(h0.size(0),-1,self.bottom_width,self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = torch.sigmoid(self.c5(h))
        return h

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
    def __init__(self, ch=128, activation=F.relu):
        super(Discriminator, self).__init__()
        self.activation = activation
        self.block1 = OptimizedBlock(3, ch)
        self.block2 = disBlock(ch, ch, activation=activation, downsample=True)
        self.block3 = disBlock(ch, ch, activation=activation, downsample=False)
        self.block4 = disBlock(ch, ch, activation=activation, downsample=False)
        self.l5 = nn.utils.spectral_norm(nn.Linear(ch, 1, bias=False))
        nn.init.xavier_uniform_(self.l5.weight.data)
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
        output = self.l5(h)
        return output

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

# summary(ResNetDiscriminator(), (3, 32, 32))