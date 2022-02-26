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
from .nn_train import loss, batch_forward


config = {'data_path':'/home/yawnlion/Desktop/PYproject/2D_deepSDF/data/data_set/supervised_data.npy',
        'mode':'infer',
        'train_loss_record_path':'/home/yawnlion/Desktop/PYproject/2D_deepSDF/data/model/train_loss_record.npy',
        'test_loss_record_path':'/home/yawnlion/Desktop/PYproject/2D_deepSDF/data/model/test_loss_record.npy',
        'if_test':False,
        'boundary_point_path':'/home/yawnlion/Desktop/PYproject/2D_deepSDF/data/data_set/infer_boundary_point.npy'}


def append_latent(latent_code,point):
    shape=point[args.num_dim].astype(int)
    latent=np.asarray(latent_code)[shape]
    in_array=np.concatenate((point[0:-1],latent))
    return in_array

batch_append_latent=vmap(append_latent,in_axes=(None,0))

matrix_append_latent=vmap(batch_append_latent,in_axes=(None,0))

def plot_SDF(nn, latent_code):
	step=0.01
	shape=args.shape_index
	x = onp.arange(-3,3,step)
	y = onp.arange(-3,3,step)
	X,Y = onp.meshgrid(x,y)
	S = np.ones(X.shape) * shape
	point = np.stack([X,Y,S],axis=2)
	in_array = matrix_append_latent(latent_code, point)
	OUT = batch_forward(nn, in_array)
	OUT = OUT.reshape(X.shape)
	plt.figure(figsize=(5,5))
	contour = plt.contour(X,Y,OUT,[-0.5,0,0.5],colors='k')
	boundary_point = onp.load(config['boundary_point_path'])[shape]
	x_b = boundary_point[:, 0]
	y_b = boundary_point[:, 1]
	plt.scatter(x_b, y_b, s = 1, c = 'r', marker = 'o')
	plt.show()



if __name__ == '__main__':
	'''
	plot_learning_curve(config['train_loss_record_path'], config['mode'])
	if(config['if_test']):
		test_loader = SDF_dataloader(config['data_path'], 'test', args)
		test_loss_record = []
		for batch_idx, (data, target) in enumerate(test_loader):
				point = np.array(data)    
				sdf = np.array(target)
				test_loss = loss(params, point, sdf)
				test_loss_record.append(math.log(test_loss))

		onp.save(config['test_loss_record_path'], test_loss_record)
	plot_learning_curve(config['test_loss_record_path'], 'test')
	'''
	file_read = open("data/model/{}ed_params.txt".format(config['mode']), "rb")
	params = pickle.load(file_read)
	nn = params[1]
	latent_code = params[0]
	plot_SDF(nn, latent_code)