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


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    model_dir = '/home/lilipan/桌面/7_4_1_new_infocr_cifar10(复件)/results/Debugging3045'
    config, unparsed = get_config()
    D = Discriminator(config.n_c_disc, config.dim_c_disc,
                            config.dim_c_cont)
    config.device = 'cuda:0'
    D.load_state_dict(torch.load(model_dir + '/checkpoint/Epoch_350.pth', map_location=config.device)['Discriminator'])
    D.to(config.device)
    D.eval()
    batch_size = 50
    num = 10000
    num_class = 10
    dataset = input_cifar10(batch_size=batch_size, root='../cifar10')

    label_kmeans = []
    labels = []
    for idx, (data, label) in enumerate(dataset):
        labels.append(label)
        logits = D(data.to(config.device))[1]
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
    # draw_graph_tsne(features, label_kmeans)
    # model.save_every_class_image(model_dir)
