import torch
from torch._C import dtype
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import random
import os
def sample_real(root='./',save_root = '../',len_samples = 500,group = 5):
    transform = transforms.ToTensor()
    datasets =  torchvision.datasets.CIFAR10(root=root, train=True, transform=transform,download=True)
    channels = 3
    num_class = 10
    img_size = datasets[0][0].size(1)
    
    img = [[] for _ in range(num_class)]
    sample_img = [ 0 for _ in range(num_class)]
    for i,(image, label) in enumerate(datasets):
        img[label].append(image.view(-1,channels,img_size,img_size))
        # print(img.shape)
        
    for i in range(num_class):
        img[i] = torch.cat(img[i])
        # indices = random.sample(range(len(img[0])), len_samples)
        # indices = torch.tensor(indices)
        indices = torch.randperm(len(img[0]))[:len_samples]
        # print(indices)
        sample_img[i] = img[i][indices]
    group_img = []
    group_label = []
    real_datasets = []
    for i in range(group):
        group_img.append([])
        group_label.append([])
        for k in range(num_class):
            group_img[i].append(sample_img[k][i*(len_samples//group//num_class):(i+1)*(len_samples//group//num_class)])
            
            group_label[i] = group_label[i]+[k]*(len_samples//group//num_class)
        
        group_img[i] = torch.cat(group_img[i], dim=0)
        # print(group_img[i].shape)
        group_img[i] = group_img[i].view(-1, channels, img_size, img_size)
        # print(group_label[i])
        group_label[i] = torch.Tensor(group_label[i]).long()
        real_datasets.append([group_img[i],group_label[i]])
    # print(real_datasets[0][1])
    torch.save(real_datasets, os.path.join(save_root,"{}_real_{}_group.pt".format(len_samples, group)))
    return real_datasets
        
sample_real()
