import numpy as onp
import jax.numpy as np
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
from jax import grad, jit, vmap, value_and_grad
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
from .nn_train import loss, batch_forward, single_forward, forward
from .MonteCarlo_getseeds import run_dichotomy_loop

def generate_mesh(step = 0.1, bound = 3):
    x = np.arange(-bound, bound, step)
    y = np.arange(-bound, bound, step)
    z = np.arange(-bound, bound, step)
    X, Y, Z = np.meshgrid(x, y, z)
    mesh = np.stack([X, Y, Z], 3)
    return mesh
    
@jit
def get_neighbour(mesh, x_step = 0.05, y_step = 0., z_step = 0.):
    X_neighbour = mesh[:, :, :, 0] + x_step
    Y_neighbour = mesh[:, :, :, 1] + y_step
    Z_neighbour = mesh[:, :, :, 2] + z_step
    mesh_neighbour = np.stack([X_neighbour, Y_neighbour, Z_neighbour], 3)
    return mesh_neighbour
    

def single_check_line(point_A, point_B, nn, latent):
    line = np.stack([point_A, point_B], 0)
    latent_tiled = np.tile(latent, (2, 1))
    in_array = np.concatenate([line, latent_tiled], 1)
    sdf = batch_forward(nn, in_array)
    return np.where(np.sign(sdf[0]*sdf[1]) > 0, False, True)


line_check_line = vmap(single_check_line, in_axes = (0, 0, None, None), out_axes = 0)
matr_check_line = vmap(line_check_line, in_axes = (0, 0, None, None), out_axes = 0)
mesh_check_line = jit(vmap(matr_check_line, in_axes = (0, 0, None, None), out_axes = 0))

@jit
def check_neighbour(mesh, x_step, y_step, z_step, nn, latent):
    mesh_neighbour = get_neighbour(mesh, x_step, y_step, z_step)
    check = mesh_check_line(mesh, mesh_neighbour, nn, latent)
    check = np.squeeze(check, 3)
    return check, mesh_neighbour
    

def single_direction_select_line(mesh, x_step, y_step, z_step, nn, latent):
    check, mesh_neighbour = check_neighbour(mesh, x_step, y_step, z_step, nn, latent)
    point_A_batch = mesh[check]
    point_B_batch = mesh_neighbour[check]
    single_direction_lines = np.stack([point_A_batch, point_B_batch], 1)
    return single_direction_lines
    

def MarchingSquares_get_pins(step, bound, nn, latent):
    mesh = generate_mesh(step, bound)
    x_direction_lines = single_direction_select_line(mesh, step, 0., 0., nn, latent)
    y_direction_lines = single_direction_select_line(mesh, 0., step, 0., nn, latent)
    z_direction_lines = single_direction_select_line(mesh, 0., 0., step, nn, latent)
    pin_batch = np.concatenate([x_direction_lines, y_direction_lines, z_direction_lines], 0)
    return pin_batch
 
def MarchingCubes_getseeds_loop(step, bound, mode, num_seeds):
    file_read = open("/gpfs/share/home/1900011026/3D_deepSDF/data/model/{}ed_params.txt".format(mode), "rb")
    params = pickle.load(file_read)
    nn = params[1]
    latent_code = params[0]
    index = latent_code.shape[0]
    batch_seeds = []
    for i in range(index):
        pin_batch = MarchingSquares_get_pins(step, bound, nn, latent_code[i])
        all_seeds = run_dichotomy_loop(pin_batch, latent_code[i], nn, 10)
        all_num = all_seeds.shape[0]
        selector = onp.random.choice(np.arange(all_num), size = num_seeds, replace=False)
        seeds = all_seeds[selector]
        batch_seeds.append(seeds)
        print(f'{mode} shape {i} done')
    batch_seeds = np.asarray(batch_seeds)
    onp.save('/gpfs/share/home/1900011026/3D_deepSDF/data/data_set/{}_seeds.npy'.format(mode), batch_seeds)
    return batch_seeds  
    



if __name__ == '__main__':
    MarchingCubes_getseeds_loop(0.1,3,'train',400)
    #MarchingCubes_getseeds_loop(0.1,3,'infer',400)
    