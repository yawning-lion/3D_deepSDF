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
from .nn_train import loss, batch_forward, forward


config = {'data_path':'/home/yawnlion/Desktop/PYproject/2D_deepSDF/data/data_set/infer_data.npy',
        'mode':'infer',
        'loss_record_path':'/home/yawnlion/Desktop/PYproject/2D_deepSDF/data/model/infer_loss_record.npy'}


def infer_loss(infer_latent, nn, in_array, sdf):
	params = [infer_latent, nn]
	out_put = forward(params,in_array)
	sdf_loss = np.sum((out_put - sdf) ** 2)
	latent_loss = np.linalg.norm(infer_latent)
	return sdf_loss + latent_loss / args.convariance


@jit
def infer_update(infer_latent, nn, in_array, sdf, opt_state):
    value, grads = value_and_grad(infer_loss)(infer_latent, nn, in_array, sdf)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value


def run_infer_loop(nn):
    infer_loss_record = []
    infer_latent = onp.random.rand(args.num_shape_infer, args.latent_len)
    opt_state = opt_init(infer_latent)
    infer_latent = get_params(opt_state)
    infer_loader = SDF_dataloader(config['data_path'], config['mode'], args)
    start_time = time.time()

    for epoch in range(args.num_epochs):

        for batch_idx, (data, target) in enumerate(infer_loader):
            point = np.array(data)    
            sdf = np.array(target)
            infer_latent, opt_state, infer_loss = infer_update(infer_latent, nn, point, sdf, opt_state)
            infer_loss_record.append(math.log(infer_loss))

        if((epoch+1)%32 == 0):
            epoch_time = time.time() - start_time
            print("Epoch {} | T: {:0.2f} | Train_loss: {:0.6f} ".format(epoch+1, epoch_time, infer_loss))
            start_time = time.time()
    onp.save(config['loss_record_path'], infer_loss_record)
    infered_params = [infer_latent, nn]
    file_w = open("/home/yawnlion/Desktop/PYproject/2D_deepSDF/data/model/infered_params.txt", "wb")
    pickle.dump(infered_params, file_w)




opt_init, opt_update, get_params = optimizers.adam(args.learning_rate)

if __name__ == '__main__':
	file_read = open("data/model/trained_params.txt", "rb")
	params = pickle.load(file_read)
	nn = params[1]
	run_infer_loop(nn)
	plot_learning_curve(config['loss_record_path'],config['mode'])