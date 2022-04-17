import numpy as onp
import jax.numpy as np
import matplotlib
import matplotlib.pyplot as plt
from jax import grad, jit, vmap, value_and_grad
from functools import partial
from jax.scipy.special import logsumexp
from jax.experimental import optimizers, stax
from jax.nn import selu,relu
from jax.experimental.stax import Dense, Relu, Sigmoid, Softplus, Selu, Tanh, Identity, FanOut, FanInConcat
from jax.numpy import tanh
from torch.utils.data import Dataset, DataLoader
from jax import random
import time
import pickle
import argparse
import math
from .utils import SDF_dataloader, plot_learning_curve
from .argument import args
from .nn_train import batch_forward
from numpy import sin, cos, pi



def from_spherical_to_catesian(spherical):
  theta = spherical[0]
  phi = spherical[1]
  r = spherical[2]
  x = sin(theta) * cos(phi) * r
  y = sin(theta) * sin(phi) * r
  z = cos(theta)
  catesian = np.array([x, y, z])
  return catesian

vmap_from_spherical_to_catesian = jit(vmap(from_spherical_to_catesian, in_axes = 0, out_axes = 0))




def single_seeds_forward(point, latent, params):
    in_array = np.concatenate([point, latent], 0)
    size = len(in_array)
    in_array = in_array.reshape(1, size)
    return batch_forward(params, in_array)[0]
    
  
    
def batch_seeds_forward(point, latent, params):
    shape = point.shape
    row = shape[0]
    latent = np.expand_dims(latent, 0).repeat(row, axis=0)
    in_array = np.concatenate([point, latent], 1)
    return batch_forward(params, in_array)

@jit
def get_grad(point, latent, params):
    value, grad = value_and_grad(batch_seeds_forward)(point, latent, params)
    return grad

def seeds_loss(point, latent, params):
    sdf = batch_seeds_forward(point, latent, params)
    sdf_loss = np.sum(sdf**2)
    return sdf_loss

@jit
def seeds_update(params, point, latent, opt_state):
    value, grads = value_and_grad(seeds_loss)(point, latent, params)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value


def run_refine_seeds_loop(seeds, shape, params, bound = 0.000001):
    nn = params[1]
    latent = params[0][shape]
    opt_state = opt_init(seeds)
    seeds = get_params(opt_state)
    start_time = time.time()
    for epoch in range(args.num_epochs):
        seeds, opt_state, train_loss = seeds_update(nn, seeds, latent, opt_state)
        if train_loss < bound:
            print("refined")
            break
    return seeds


def seeds_grad(seeds, params, index_master, index_slave, index_map):
    '''
    seeds should be like n*2 array(in 2d condition)
    params can be read from
    file_read = open("/gpfs/share/home/1900011026/2D_deepSDF/data/model/{}ed_params.txt".format(mode), "rb")
    params = pickle.load(file_read)
    '''
    shape_slave = index_map[index_slave]
    latent_code = params[0]
    nn = params[1]
    latent = latent_code[shape_slave]
    seeds_grad = get_grad(seeds, latent, nn)
    return seeds_grad


opt_init, opt_update, get_params = optimizers.adam(args.learning_rate)

if __name__ == '__main__':
    shape = 1
    num_seeds = 1000
    raw_seeds = onp.random.uniform( -2, 2, size = (num_seeds, 3))
#sample from a sphere
    theta = onp.random.uniform(0, 2*pi, size = num_seeds)
    arr1 = sin(theta).reshape(num_seeds, 1)
    arr2 = cos(theta).reshape(num_seeds, 1)
    #seeds = np.concatenate([arr1, arr2], 1)
    mode = 'infer'
    file_read = open("/home/ubuntu/DESKTOP/rsc/3D_deepSDF/data/model/{}ed_params.txt".format(mode), "rb")
    params = pickle.load(file_read)
    #refined_seeds = run_refine_seeds_loop(seeds, shape, params)
    batch_seeds = []
    for i in range(args.num_shape_infer):

        refined_seeds = run_refine_seeds_loop(raw_seeds, i, params)
        batch_seeds.append(refined_seeds)
    batch_seeds = np.asarray(batch_seeds)
    onp.save('/home/ubuntu/DESKTOP/rsc/3D_deepSDF/data/data_set/{}_seeds.npy'.format(mode), batch_seeds)
    '''
    plt.figure(figsize=(5, 5))
    plt.scatter(refined_seeds[:, 0], refined_seeds[:, 1], s=1)
    plt.savefig('/home/ubuntu/DESKTOP/rsc/3D_deepSDF/data/img/seeds_shape{}{}'.format(shape // 10, shape%10))
    plt.close()
    '''