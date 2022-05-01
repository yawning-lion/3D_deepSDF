import numpy as onp
from numpy import pi
import jax.numpy as np
from jax.numpy import sin, cos
import matplotlib
import matplotlib.pyplot as plt
from jax import grad, jit, vmap, value_and_grad
from jax.scipy.special import logsumexp
from jax.example_libraries import optimizers, stax
from jax.nn import selu,relu
from jax.example_libraries.stax import Dense, Relu, Sigmoid, Softplus, Selu, Tanh, Identity, FanOut, FanInConcat
from jax.numpy import tanh
from torch.utils.data import Dataset, DataLoader
from jax import random
import time
import pickle
import argparse
import math
from .utils import SDF_dataloader, plot_learning_curve
from .argument import args
from .nn_train import loss, batch_forward
from .nn_visualize import matrix_append_latent

num_pin = 2000
num_seeds = 1000
pin_length = 0.5
mode = 'train'

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


def get_middle(num_test, Max = 2, Min = 0.5):
	r_batch = onp.random.uniform(Min, Max, num_test)
	r_batch = np.array(r_batch)
	theta_batch = onp.random.uniform(0, 2*pi, num_test)
	theta_batch = np.array(theta_batch)
	middle_batch = batch_pole_transformer(r_batch, theta_batch)
	return middle_batch


def get_test(num_test, middle_batch):
	orien_batch = onp.random.uniform(0, 2*pi, (num_test,1))
	orien_batch = np.array(orien_batch)
	middle_batch = get_middle(num_test)
	test_batch = np.concatenate([middle_batch, orien_batch], 1)
	return test_batch

def single_get_end(test, lenth = 1):
	orien = test[-1]
	middle = test[:-1]
	delta = np.array([lenth*cos(orien), lenth*sin(orien)])
	head = middle + delta
	tail = middle - delta
	end = np.stack((head, tail), 0)
	return end

batch_get_end = jit(vmap(single_get_end, in_axes = (0, None), out_axes = 0))

def get_pin(num_test, lenth):
	middle_batch = get_middle(num_test)
	test_batch = get_test(num_test, middle_batch)
	end_batch = batch_get_end(test_batch, lenth)
	return end_batch


def single_find_pin_val(end, latent, nn):
	head = end[0]
	tail = end[1]
	in_head = np.concatenate([head, latent], 0)
	in_tail = np.concatenate([tail, latent], 0)
	in_array = np.stack((in_head, in_tail), 0)
	val = batch_forward(nn, in_array)
	return val

batch_find_pin_val = vmap(single_find_pin_val, in_axes = (0, None, None), out_axes = 0)

def select_pin_helper(val, pin):
	signal = np.multiply(val[:,0], val[:,1])
	#print(signal)
	selector = np.where(signal < 0, True, False).reshape(-1)
	#print(selector)
	return selector


#针的形状是（x, 2, 2)，每跟针都由两个端点表示

def get_seg(pin, val):
	a = np.abs(val[0])
	b = np.abs(val[1])
	miu_a = b / (a + b)
	miu_b = a / (a + b)
	seg = miu_a * pin[0] + miu_b * pin[1]
	return seg

@jit
def dichotomy(pin_batch, latent, nn):
    middle_batch = (pin_batch[:, 0] + pin_batch[:, 1]) / 2
    left_pin_batch = np.stack((pin_batch[:, 0], middle_batch[:]), 1)
    right_pin_batch = np.stack((middle_batch[:], pin_batch[:, 1]), 1)
    #print(left_pin_batch.shape)
    divided_pin_batch = np.concatenate([left_pin_batch, right_pin_batch], 0)
    #print(divided_pin_batch[0:2])
    val_batch = batch_find_pin_val(divided_pin_batch, latent, nn)
    selector = select_pin_helper(val_batch, divided_pin_batch)
    return selector, divided_pin_batch, val_batch


def run_dichotomy_loop(pin_batch, latent, nn, epoch = 10):
    for i in range(epoch):
        selector, divided_pin_batch, val_batch = dichotomy(pin_batch, latent, nn)
        pin_batch = divided_pin_batch[selector]
        val_batch = val_batch[selector]
    seeds_batch = (pin_batch[:, 0] + pin_batch[:, 1]) / 2
    return seeds_batch



batch_get_seg = vmap(get_seg, in_axes = (0, 0), out_axes = 0)


def Monte_Carlo_getseeds(num_pin, pin_length, mode, latent, nn):
    pin_batch = get_pin(num_pin, pin_length)
    seeds = run_dichotomy_loop(pin_batch, latent, nn, 10)
    return seeds, shape, mode





def run_get_seeds_loop(num_seeds, num_pin, pin_length, mode, index):
    file_read = open("/home/ubuntu/DESKTOP/rsc/3D_deepSDF/data/model/{}ed_params.txt".format(mode), "rb")
    params = pickle.load(file_read)
    nn = params[1]
    latent_code = params[0]
    batch_seeds = []
    for i in range(index):
        seeds, shape, mode = Monte_Carlo_getseeds(num_pin, pin_length, mode, latent_code[i], nn)
        batch_seeds.append(seeds[:num_seeds])
    batch_seeds = np.asarray(batch_seeds)
    onp.save('/home/ubuntu/DESKTOP/rsc/3D_deepSDF/data/data_set/{}_seeds.npy'.format(mode), batch_seeds)
    return batch_seeds

#如果报错了，再跑一遍。针的个数肯定大于种子的个数，可能每个形状的种子个数都不一样，为了能够统一，都取前num_seeds个。如果报错的话，是存在形状种子个数小于num_seeds了
#可以尝试调大针个数，调小种子个数，再跑一边



if __name__ == '__main__':
    '''
    file_read = open("/gpfs/share/home/1900011026/2D_deepSDF/data/model/{}ed_params.txt".format(mode), "rb")
    params = pickle.load(file_read)
    nn = params[1]
    latent_code = params[0]
    shape = 2
    batch_seg, shape, mode = Monte_Carlo_getseeds(num_pin, pin_length, shape, mode, latent_code, nn)
    '''
    batch_seeds = run_get_seeds_loop(num_seeds, num_pin, pin_length, 'train', args.num_shape_train)
    batch_seeds = run_get_seeds_loop(num_seeds, num_pin, pin_length, 'infer', args.num_shape_infer)
    '''
    step=0.01
    x = onp.arange(-3,3,step)
    y = onp.arange(-3,3,step)
    X,Y = onp.meshgrid(x,y)
    S = np.ones(X.shape) * shape
    point = np.stack([X,Y,S],axis=2)
    in_array = matrix_append_latent(latent_code, point)
    OUT = batch_forward(nn, in_array)
    OUT = OUT.reshape(X.shape)
    plt.figure(figsize=(5,5))
    contour = plt.contour(X,Y,OUT,[-0.5,0.5],colors='k')
    plt.scatter(batch_seg[:, 0], batch_seg[:,1], s = 0.5, c = 'r', marker = 'o')
    plt.savefig('/gpfs/share/home/1900011026/2D_deepSDF/data/img/seeds/{}_seeds_{}{}'.format(mode, shape // 10, shape%10))
    plt.close()
    '''
