from models.cifar10.generator import Generator
from config import get_config
import torch
import torchvision
from PIL import Image
import yaml
import numpy as np
def make_batches(size, batch_size):
    '''Returns a list of batch indices (tuples of indices).
    '''
    return [(i, min(size, i + batch_size)) for i in range(0, size, batch_size)]


def _sample(config, value, batch_size):
    # Sample Z from N(0,1)
    z = torch.randn(batch_size, config.dim_z, device=config.device)

    # Sample discrete latent code from Cat(K=dim_c_disc)
    if config.n_c_disc != 0:
        idx = np.zeros((config.n_c_disc, batch_size))
        c_disc = torch.zeros(batch_size, config.n_c_disc,
                                config.dim_c_disc, device=config.device)
        for i in range(config.n_c_disc):
            # idx[i] = np.random.randint(
            #     config.dim_c_disc, size=batch_size)
            idx[i] = np.ones(batch_size)*value
            c_disc[torch.arange(0, batch_size), i, idx[i]] = 1.0

    # Sample continuous latent code from Unif(-1,1)
    c_cond = torch.rand(batch_size, config.dim_c_cont,
                        device=config.device) * 2 - 1

    # Concat z, c_disc, c_cond
    if config.n_c_disc != 0:
        for i in range(config.n_c_disc):
            z = torch.cat((z, c_disc[:, i, :].squeeze()), dim=1)
    z = torch.cat((z, c_cond), dim=1)

    if config.n_c_disc != 0:
        return z, idx
    else:
        return z


if __name__ == "__main__":
    model_dir = '/root/peijun/peijun/new_infocr_cifar10/results/Debugging3442'
    config, unparsed = get_config()
    G = Generator(config.dim_z, config.n_c_disc, config.dim_c_disc, config.dim_c_cont)
    config.device = 'cuda:3'
    G.load_state_dict(torch.load(model_dir + '/checkpoint/Epoch_350.pth', map_location=config.device)['Generator'])
    G.to(config.device)
    G.eval()
    for value in range(10):
        if config.n_c_disc != 0:
            z, idx = _sample(config, value, 50000)
        else:
            z = _sample(config, value, 50000)
        batches = make_batches(50000, 25)
        h = []
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            noise_batch = z[batch_start:batch_end].to(config.device)
            out = G(noise_batch).detach().cpu().numpy()
            out = np.multiply(np.add(np.multiply(out, 0.5), 0.5), 255).astype('int32')
            #out = out[sli]
            h.append(out)
        h = np.vstack(h)
        h = np.transpose(h, (0, 2, 3, 1))
        np.save('{}/{}_lda300_1'.format(model_dir, value), h)