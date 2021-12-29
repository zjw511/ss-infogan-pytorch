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
from evaluate import *
from sample import *
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
        
        self.unsup = config.unsup
        self.sup = config.sup
        

        self.dataloader_unlabel = dataloader_unlabel
        # self.dataloader_label = dataloader_label
        self.img_list = {}
        self.device = 'cuda:0'
        self.build_models()

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
        if self.n_c_disc != 0:
            optim_G = self.set_optimizer([self.G.parameters(),], lr=self.lr_G)
        else:
            optim_G = self.set_optimizer([self.G.parameters(), self.D.module_Q.parameters(
            ), self.D.latent_cont_mu.parameters()], lr=self.lr_G)
        optim_D = self.set_optimizer(
            [self.D.parameters()], lr=self.lr_D)

        # adversarial_loss = torch.nn.BCELoss()
        categorical_loss = torch.nn.CrossEntropyLoss()
        continuous_loss = torch.nn.MSELoss()

        # Sample fixed latent codes for comparison
        if self.n_c_disc != 0:
            fixed_z_dict = self._sample_fixed_noise()
        else:
            fixed_z = self._sample_fixed_noise()

        start_time = time.time()
        # num_steps = len(self.data_loader)
        step = 0
        log_target = {}
        # real_datasets
        # sample_real()
        real_datasets = sample_real(save_root=os.path.join(self.project_root, 'results/%s'%self.model_name))#torch.load("../{}_real_{}_group.pt".format(500, 5))
        transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x * 2. - 1.)
        # transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size())),
        ])
        ''' 
        info data
        '''
        unlabelled_sampling_prob = 0
        unlabelled_percentage = 0.99
        additive_prob_growth = 0.1
        real_idx = 0
        group_num = 5
        for epoch in range(self.num_epoch):
            epoch_start_time = time.time()
            step_epoch = 0
            for i in range(500):
                    # label or unlabel
                if torch.bernoulli(unlabelled_sampling_prob*torch.ones(1)) == 0:
                    labelled_batch = True
                    data,label_ = real_datasets[real_idx % group_num]
                    data = transform(data)
                    # try:
                    #     data,label_ = next(dataloader_label_iter)
                    # except:
                    #     dataloader_label_iter =  iter(self.dataloader_label)
                    #     data,label_ = next(dataloader_label_iter)
                    real_idx+=1
                else:
                    labelled_batch = False
                    try:
                        data,label_ = next(dataloader_unlabel_iter)
                    except:
                        dataloader_unlabel_iter =  iter(self.dataloader_unlabel)
                        data,label_ = next(dataloader_unlabel_iter)
                        # if (data.size()[0] != self.batch_size):
                        #     self.batch_size = data.size()[0]

                data_real = data.to(self.device)
                label_ = label_.to(self.device)
                # Update Discriminator
                # Reset optimizer
                optim_D.zero_grad()

                # Calculate Loss D(real)
                prob_real,sup_real,_ = self.D(data_real)

                loss_D_real = torch.mean(F.relu(1. - prob_real.view(-1)))

                # Calculate gradient -> grad accums to module_shared / modue_D
                loss_D_real.backward(retain_graph=True)
                if labelled_batch:
                    # print(label_.shape,sup_real.shape)
                    loss_info_semsup_real = categorical_loss(sup_real,label_) * self.sup
                    # print(loss_info_semsup_real)
                    loss_info_semsup_real.backward(retain_graph=True)
                # calculate Loss D(fake)
                # Sample noise, latent codes
                if self.n_c_disc != 0:
                    z, idx ,c_cond= self._sample(self.batch_size)
                else:
                    z = self._sample(self.batch_size)
                data_fake = self.G(z)

                prob_fake_D,_,unsup_fake = self.D(data_fake.detach())
                loss_D_fake = torch.mean(F.relu(1. + prob_fake_D.view(-1)))

                # Calculate gradient -> grad accums to module_shared / modue_D
                loss_D_fake.backward(retain_graph=True)
                fake_unsup = c_cond
                unsup_loss = continuous_loss(unsup_fake,fake_unsup) * self.unsup
                unsup_loss.backward()
                loss_D = loss_D_real + loss_D_fake + unsup_loss + loss_info_semsup_real
                # loss_D.backward()
                # Update Parameters for D
                optim_D.step()

                # Update Generator and Q
                if i % 3 ==0:
                # if True:
                # Reset Optimizer
                    optim_G.zero_grad()

                    # Calculate loss for generator
                    # Sample noise, latent codes
                    if self.n_c_disc != 0:
                        z, idx,con_c = self._sample(2*self.batch_size)
                    else:
                        z = self._sample(2*self.batch_size)
                    data_fake = self.G(z)
                    if self.n_c_disc != 0:
                        prob_fake, disc_logits, latent_mu = self.D(data_fake)
                    else:
                        prob_fake, latent_mu, latent_var = self.D(data_fake)
                    loss_G =  - torch.mean(prob_fake.view(-1))

                    # if self.n_c_disc != 0:
                    #     # Calculate loss for discrete latent code
                    target = torch.LongTensor(idx).to(self.device).view(-1)
                    loss_G.backward(retain_graph=True)
                    # print(target.shape,disc_logits.shape)
                    semsup_fake_loss = categorical_loss(disc_logits,target) * self.sup
                    # semsup_fake_loss.backward(retain_graph=True )
                    fake_unsup = con_c
                    loss_unsup = continuous_loss(latent_mu,fake_unsup)* self.unsup
                    # loss_unsup.backward()
                    info_loss = semsup_fake_loss + loss_unsup
                    info_loss.backward()
                    loss_info = loss_G + info_loss
                    optim_G.step()


                if i % 5 == 4:
                    self.lr_G = self.adjust_learning_rate(optim_G, epoch)
                    self.lr_D = self.adjust_learning_rate(optim_D, epoch)
                    # self.lr_CR = self.adjust_learning_rate(optim_CR, epoch)
                if (step % self.log_step == 0):
                    print('==========')
                    print('Model Name: %s'%self.model_name)                    
                    if self.n_c_disc != 0:
                        print('Epoch [%d/%d], Step [%d], Elapsed Time: %s \nLoss D : %.4f,  Loss Info: %.4f\n Loss_Gen: %.4f'
                              % (epoch + 1, self.num_epoch, step_epoch, datetime.timedelta(seconds=time.time()-start_time), loss_D, loss_info.item(), loss_G.item()))
                    else:
                        print('Epoch [%d/%d], Step [%d], Elapsed Time: %s \nLoss D : %.4f, Loss_CR: %.4f, Loss Info: %.4f\n Loss_Gen: %.4f'
                              % (epoch + 1, self.num_epoch, step_epoch, datetime.timedelta(seconds=time.time()-start_time), loss_D, loss_info.item(), loss_G.item()))
                    print(
                        'Prob_real_D:%.4f, Prob_fake_D:%.4f, Prob_fake_G:%.4f'%(prob_real.mean(),prob_fake_D.mean(),prob_fake.mean()))

                step += 1
                step_epoch += 1

            if self.n_c_disc != 0:
                # Plot and Log generated images
                for key in fixed_z_dict.keys():
                    fixed_z = fixed_z_dict[key].to(self.device)
                    idx_c_disc = key[0]
                    idx_c_cont = key[1]

                    # Generate and plot images from fixed inputs
                    imgs, title = plot_generated_data(
                        self.config, self.G, fixed_z, epoch, idx_c_disc, idx_c_cont)

                    # collect images for animation
                    self.img_list[(epoch, key[0], key[1])] = vutils.make_grid(
                        imgs, nrow=10, padding=2, normalize=True)
            else:
                fixed_z = fixed_z.to(self.device)
                imgs, title = plot_generated_data_only_cont(
                    self.config, self.G, fixed_z, epoch)
                self.img_list[epoch] = vutils.make_grid(
                    imgs, nrow=10, padding=2, normalize=True)

            if unlabelled_sampling_prob < unlabelled_percentage:
                unlabelled_sampling_prob += additive_prob_growth
            if unlabelled_sampling_prob > unlabelled_percentage:
                unlabelled_sampling_prob = unlabelled_percentage
            
            if (epoch+1) % 50 == 0:
                self.save_model(epoch+1)
                self.evaluate(epoch)

    def evaluate(self,epoch):
        batch_size = 50
        num = 10000
        num_class = 10
        dataset = input_cifar10(batch_size=batch_size, root='../cifar10')

        label_kmeans = []
        labels = []
        for idx, (data, label) in enumerate(dataset):
            labels.append(label)
            logits = self.D(data.to(self.device))[1]
            # print(logits)
            label_pred = torch.argmax(logits, dim=1)
            label_kmeans.append(label_pred)

        label_kmeans = torch.cat(label_kmeans, dim=0).view(-1)
        labels = torch.cat(labels, dim=0).view(-1)
        labels, idx = torch.sort(labels.data)
        label_kmeans = label_kmeans.data[idx]

        acc = calculate_acc(label_kmeans, num, num_class)
        nmi = NMI(labels.data.cpu().numpy(), label_kmeans.data.cpu().numpy())
        ari = ARI(labels.data.cpu().numpy(), label_kmeans.data.cpu().numpy())
        print('acc: %.4f; nmi: %.4f; ari: %.4f'% (acc, nmi, ari))
        list_strings = []
        current_losses = { 'mean_acc':acc, 'mean_nmi':nmi, 'mean_ari':ari}
        print(current_losses)
        for eval_name, eval_value in current_losses.items():
            list_strings.append('%s = %.2f '%(eval_name, eval_value))
        full_string = ' '.join(list_strings)
        full_string = str(self.model_name)+'\t'+str(self.sup)+'\t'+str(self.unsup)+full_string
        # print('epoch = {} {} \n'.format(epoch, full_string))
        with open('evaluate_result_xiuzheng.txt', "a") as f:
            f.write('epoch={} {} \n'.format(epoch, full_string))
    def test(self):
        pass
