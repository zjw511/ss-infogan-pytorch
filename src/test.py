import os
import torch
import numpy as np
import time
import datetime
import itertools
import torchvision.utils as vutils
import torch.nn.functional as F
from utils import *
from metrics import *

torch.autograd.set_detect_anomaly(True)
import torch
import torchvision
from PIL import Image
import yaml
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn import manifold
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
from models.cifar10.discriminator import Discriminator
from config import get_config


def k_means(f, num_class):
    f_cluster = torch.chunk(f,num_class, dim=0)
    f_means = []
    for i in range(num_class):
        mean_i = torch.mean(f_cluster[i], dim=0)
        f_means.append(mean_i.view(1, -1))
    f_means = torch.cat(f_means, dim=0)
    # label_kmeans = KMeans(n_clusters=num_class, init=f_means.data.cpu().numpy(), max_iter=30000).fit_predict(f.data.cpu().numpy())
    label_kmeans = KMeans(n_clusters=num_class, random_state=0).fit_predict(f.data.cpu().numpy())
    return torch.from_numpy(label_kmeans)


def calculate_acc(label_kmeans, num, num_class):
    acc = 0
    for i in range(num_class):
        index = torch.eq(label_kmeans, i).float()
        index = torch.chunk(index, num_class ,dim=0)
        n = []
        for j in range(num_class):
            n.append(torch.sum(index[j]).view(-1, 1))
        acc += torch.max(torch.cat(n, dim=0))
    acc = acc / num
    return acc


def dataset(root, dataset='Fashion', train=False):
    # load dataset
    transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x * 2. - 1.)])
    if dataset == 'fashion':
        channels =1
        datasets = torchvision.datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)
    elif dataset == 'mnist':
        datasets = torchvision.datasets.MNIST(root=root, train=False, transform=transform, download=True)
        channels =1
    elif dataset == 'cifar10':
        datasets =  torchvision.datasets.CIFAR10(root=root, train=False, transform=transform,download=True)
        channels =3
    else:
        raise ValueError('Unsupport data')
    return datasets, channels


def draw_graph_tsne(X, y):
    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X.data.cpu())

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
#     print(X_norm.shape)
#     noise = (np.random.randn(*X_norm.shape)+1)*0.5
#     X_norm += 0.1*noise
    # plt.figure(figsize=(8, 8))
    for i in range(10):
        plt.scatter(X_norm[y==i,0], X_norm[y==i,1], alpha=0.8, label='%s' % i)
    # plt.xticks([])
    # plt.yticks([])
    plt.legend()
    plt.savefig('./samples/clusters_image.png')
    plt.show()


def get_test_dsprites(root, batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(),
        # transforms.Resize(im_size),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.ImageFolder(root, transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    return trainloader


def input_cifar10(batch_size, root='../cifar10'):
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2. - 1.)
        # transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size())),
        ])

    trainset = torchvision.datasets.CIFAR10(root=root, transform=transform, train=False,download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4, drop_last=True)
    return trainloader

class Trainer:
    def __init__(self, config, dataloader_unlabel):
        self.config = config
        self.dataset = config.dataset
        self.n_c_disc = config.n_c_disc
        self.dim_c_disc = config.dim_c_disc
        self.dim_c_cont = config.dim_c_cont
        self.dim_z = config.dim_z
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer
        self.lr_G = config.lr_G
        self.lr_D = config.lr_D
        self.lr_CR = config.lr_CR
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        # self.gpu_id = config.gpu_id
        self.num_epoch = config.num_epoch
        self.lambda_disc = config.lambda_disc
        self.lambda_cont = config.lambda_cont
        self.alpha = config.alpha
        self.log_step = config.log_step
        self.project_root = config.project_root
        self.model_name = config.model_name
        self.use_visdom = config.use_visdom

        self.dataloader_unlabel = dataloader_unlabel
        # self.dataloader_label = dataloader_label
        self.img_list = {}
        self.device = 'cuda:0'
        self.build_models()
    #     if self.use_visdom:
    #         self.visdom_log_number = config.visdom_log_number
    #         self._set_plotter(config)
    #         self._set_logger()

    # def _set_plotter(self, config):
    #     self.plotter = VisdomPlotter(config)

    # def _set_logger(self):
    #     self.logger = Logger()
    #     self.logger.create_target('step', 's', 'iterations')
    #     self.logger.create_target('Loss_G', 'G', 'Generator Loss')
    #     self.logger.create_target('Loss_D', 'D', 'Discriminator Loss')
    #     self.logger.create_target('Loss_Info', 'I', 'Info(G+L_d+L_c) Loss')
    #     if self.n_c_disc != 0:
    #         self.logger.create_target('Loss_Disc', 'I_d', 'Discrete Code Loss')
    #     self.logger.create_target('Loss_CR', 'CR', 'Contrastive Loss')
    #     self.logger.create_target(
    #         'Prob_D', 'P_d_real', 'Prob of D for real / fake sample')
    #     self.logger.create_target(
    #         'Prob_D', 'P_d_fake', 'Prob of D for real / fake sample')
    #     self.logger.create_target(
    #         'Loss_Cont', 'I_c_total', 'Continuous Code Loss')
    #     # for i in range(self.dim_c_cont):
    #     # self.logger.create_target(
    #     # 'Loss_Cont', f'I_c_{i+1}', 'Continuous Code Loss')

    #     return

    def _sample(self, batch_size):
        # Sample Z from N(0,1)
        z = torch.randn(batch_size, self.dim_z, device=self.device)

        # Sample discrete latent code from Cat(K=dim_c_disc)
        if self.n_c_disc != 0:
            idx = np.zeros((self.n_c_disc, batch_size))
            c_disc = torch.zeros(batch_size, self.n_c_disc,
                                 self.dim_c_disc, device=self.device)
            for i in range(self.n_c_disc):
                idx[i] = np.random.randint(
                    self.dim_c_disc, size=batch_size)
                c_disc[torch.arange(0, batch_size), i, idx[i]] = 1.0

        # Sample continuous latent code from Unif(-1,1)
        c_cond = torch.rand(batch_size, self.dim_c_cont,
                            device=self.device) * 2 - 1

        # Concat z, c_disc, c_cond
        if self.n_c_disc != 0:
            for i in range(self.n_c_disc):
                z = torch.cat((z, c_disc[:, i, :].squeeze()), dim=1)
        z = torch.cat((z, c_cond), dim=1)

        if self.n_c_disc != 0:
            return z, idx , c_cond
        else:
            return z

    def _sample_fixed_noise(self):

        if self.n_c_disc != 0:
            # Sample Z from N(0,1)
            fixed_z = torch.randn(self.dim_c_disc*10, self.dim_z)

            # For each discrete variable, fix other discrete variable to 0 and random sample other variables.
            idx = np.arange(self.dim_c_disc).repeat(10)

            c_disc_list = []
            for i in range(self.n_c_disc):
                zero_template = torch.zeros(
                    self.dim_c_disc*10, self.n_c_disc, self.dim_c_disc)
                # Set all the other discrete variable to Zero([1,0,...,0])
                for ii in range(self.n_c_disc):
                    if (i == ii):
                        pass
                    else:
                        zero_template[:, ii, 0] = 1.0
                for j in range(len(idx)):
                    zero_template[np.arange(self.dim_c_disc*10), i, idx] = 1.0
                c_disc_list.append(zero_template)

            c_range = torch.linspace(start=-1, end=1, steps=10)
            c_range_list = []
            for i in range(self.dim_c_disc):
                c_range_list.append(c_range)
            c_range = torch.cat(c_range_list, dim=0)

            c_cont_list = []
            for i in range(self.dim_c_cont):
                c_zero = torch.zeros(self.dim_c_disc * 10, self.dim_c_cont)
                c_zero[:, i] = c_range
                c_cont_list.append(c_zero)

            fixed_z_dict = {}
            for idx_c_disc in range(len(c_disc_list)):
                for idx_c_cont in range(len(c_cont_list)):
                    z = fixed_z.clone()
                    for j in range(self.n_c_disc):
                        z = torch.cat(
                            (z, c_disc_list[idx_c_disc][:, j, :].squeeze()), dim=1)
                    z = torch.cat((z, c_cont_list[idx_c_cont]), dim=1)
                    fixed_z_dict[(idx_c_disc, idx_c_cont)] = z
            return fixed_z_dict
        else:
            fixed_z = torch.randn(self.dim_c_cont*10, self.dim_z)
            c_range = np.linspace(start=-1, stop=1, num=10)
            template = torch.zeros((self.dim_c_cont*10, self.dim_c_cont))
            for c_dim in range(self.dim_c_cont):
                for i, v in enumerate(c_range):
                    template[10 * c_dim + i, c_dim] = v
            fixed_z = torch.cat((fixed_z, template), dim=1)
            return fixed_z

    def build_models(self):
        if self.dataset == 'cifar10':
            from models.cifar10.discriminator import Discriminator
            from models.cifar10.generator import Generator
            from models.cifar10.cr_discriminator import CRDiscriminator
            from models.cifar10.SupContrast_resnet_big import SupConResNetZ
        else:
            raise(NotImplementedError)

        # Initiate Models
        self.G = Generator(self.dim_z, self.n_c_disc, self.dim_c_disc,
                           self.dim_c_cont).to(self.device)
        self.D = Discriminator(self.n_c_disc, self.dim_c_disc,
                               self.dim_c_cont).to(self.device)
        # self.CR = CRDiscriminator(self.dim_c_disc).to(self.device)
        # self.CR = SupConResNetZ(name='resnet18', feat_dim=self.dim_c_disc).to(self.device)
        model_dir = '/root/zjw/ssinfoGAN_cifar10/results/Debugging6213/checkpoint/Epoch_250.pth'
        self.G.load_state_dict(torch.load(model_dir, map_location=self.device)['Generator'])
        # self.D.load_state_dict(torch.load(model_dir, map_location=self.device)['Discriminator'])
        # # Initialize
        # self.G.apply(weights_init_normal)
        # self.D.apply(weights_init_normal)
        # self.CR.apply(weights_init_normal)
        return

    def set_optimizer(self, param_list, lr):
        params_to_optimize = itertools.chain(*param_list)
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                params_to_optimize, lr=lr, betas=(self.beta1, self.beta2))
            return optimizer
        else:
            raise NotImplementedError

    def save_model(self, epoch):
        save_dir = os.path.join(
            self.project_root, 'results/%s/checkpoint'%self.model_name)
        os.makedirs(save_dir, exist_ok=True)

        # Save network weights.
        torch.save({
            'Generator': self.G.state_dict(),
            'Discriminator': self.D.state_dict(),
            'configuations': self.config
        }, '%s/Epoch_%s.pth'%(save_dir,epoch))

        return

    def _get_idx_fixed_z(self):
        # self.rand_ = 
        idx_fixed = torch.from_numpy(np.array([np.random.randint(0, self.dim_c_disc)
                                               for i in range(self.batch_size)])).to(self.device)
        code_fixed = np.array([np.random.rand()
                               for i in range(self.batch_size)])
        if self.n_c_disc != 0:
            latent_pair_1, _ = self._sample(self.batch_size)
            latent_pair_2, _ = self._sample(self.batch_size)
        else:
            latent_pair_1 = self._sample(self.batch_size)
            latent_pair_2 = self._sample(self.batch_size)
        for i in range(self.batch_size):
            latent_pair_1[i][-self.dim_c_disc+idx_fixed[i]] = code_fixed[i]
            latent_pair_2[i][-self.dim_c_disc+idx_fixed[i]] = code_fixed[i]
        z = torch.cat((latent_pair_1, latent_pair_2), dim=0)

        return z, idx_fixed

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] -= 2e-4/60000
        return param_group['lr']

    def train(self):
        # Set opitmizers
        # Sample fixed latent codes for comparison

        for i in range(10):
            if self.n_c_disc != 0:
                fixed_z_dict = self._sample_fixed_noise()
            else:
                fixed_z = self._sample_fixed_noise()



            epoch = 1
                # Plot and Log generated images
            print(fixed_z_dict.keys())
            for key in fixed_z_dict.keys():
                fixed_z = fixed_z_dict[key].to(self.device)
                idx_c_disc = key[0]
                idx_c_cont = key[1]

                # Generate and plot images from fixed inputs
                imgs, title = plot_generated_data(
                    self.config, self.G, fixed_z, epoch, idx_c_disc, idx_c_cont)
                torchvision.utils.save_image(imgs,  './gen_classes{}.png'.format(i), nrow=10 , normalize=True)
            # collect images for animation
            # self.img_list[(epoch, key[0], key[1])] = vutils.make_grid(
            #     imgs, nrow=10, padding=2, normalize=True)

                # # Log Image
                # if self.use_visdom:

config, unparsed = get_config()
    # save_config(config)
dataloader_unlabel = get_loader(config.batch_size, config.project_root, config.dataset)
trainer = Trainer(config, dataloader_unlabel)
trainer.train()