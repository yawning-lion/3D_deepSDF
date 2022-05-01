import numpy as onp
import jax.numpy as np
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



config = {'data_path':'/gpfs/share/home/1900011026/3D_deepSDF/data/data_set/supervised_data.npy',
        'mode':'train',
        'loss_record_path':'/gpfs/share/home/1900011026/3D_deepSDF/data/model/train_loss_record.npy'}


def get_mlp(args):
    if args.activation == 'selu':
        act_fun = Selu
    elif args.activation == 'tanh':
        act_fun = Tanh
    elif args.activation == 'relu':
        act_fun = Relu
    else:
        raise ValueError(f"Invalid activation function {args.activation}.")

    layers_hidden = []
    for _ in range(args.n_hidden):
        layers_hidden.extend([Dense(args.width_hidden), act_fun])
    layers_hidden.append(Dense(1))

    if args.skip:
        layers_skip = []
        for _ in range(args.n_skip):
            layers_skip.extend([Dense(args.width_hidden), act_fun])
        layers_skip.append(Dense(args.width_hidden - args.latent_len - args.point_dim))
        mlp = stax.serial(FanOut(2),
                        stax.parallel(Identity,
                        stax.serial(*layers_skip)),
                        FanInConcat(),
                        stax.serial(*layers_hidden))
    else:
        mlp = stax.serial(*layers_hidden)

    return mlp



def append_latent(latent_code, point):
    shape = point[args.point_dim].astype(int)
    latent = np.asarray(latent_code)[shape]#jax.errors.TracerArrayConversionError
    in_array = np.concatenate((point[0:-1], latent))
    return in_array


batch_append_latent=vmap(append_latent,in_axes=(None,0))


def forward(params, in_array):
    in_array = batch_append_latent(params[0], in_array)
    out_put = batch_forward(params[1], in_array)
    return out_put.reshape(-1)

def boundary_loss(params, boundary_input):
    out_put = forward(params, boundary_input)
    sdf_loss = np.sum(out_put ** 2)
    return sdf_loss

def loss(params, in_array, sdf, boundary_input):
    out_put = forward(params,in_array)
    sdf_loss = np.sum((out_put - sdf) ** 2)
    latent_loss = np.linalg.norm(params[0])
    return latent_loss / args.convariance + sdf_loss + boundary_loss(params, boundary_input)

def single_forward(params, point):
    in_array = append_latent(params[0], point)
    return batch_forward(params[1], in_array)[0]


@jit
def update(params, point, sdf, opt_state, boundary_input):
    value, grads = value_and_grad(loss)(params, point, sdf, boundary_input)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value


def run_training_loop():
    train_loss_record = []
    latent_code = onp.random.rand(args.num_shape_train, args.latent_len)
    params = [latent_code, net_params]
    opt_state = opt_init(params)
    params = get_params(opt_state)
    train_loader = SDF_dataloader(config['data_path'], config['mode'], args)
    start_time = time.time()
    boundary = np.load("/gpfs/share/home/1900011026/3D_deepSDF/data/data_set/train_batch_verts.npy")
    shape_boundary = boundary.shape
    boundary = boundary.reshape(shape_boundary[0]*shape_boundary[1], shape_boundary[2])
    shape_index = np.arange(args.num_shape_train)
    shape_index = np.repeat(shape_index, shape_boundary[1])
    shape_index = shape_index.reshape(shape_index.size, 1)
    boundary_input = np.concatenate([boundary, shape_index], 1)
    for epoch in range(args.num_epochs):

        for batch_idx, (data, target) in enumerate(train_loader):

            point = np.array(data)    
            sdf = np.array(target)
            params, opt_state, train_loss = update(params, point, sdf, opt_state, boundary_input)
            
        train_loss_record.append(math.log(train_loss))
        if((epoch+1)%32 == 0):
            epoch_time = time.time() - start_time
            print("Epoch {} | T: {:0.2f} | Train_loss: {:0.6f} ".format(epoch+1, epoch_time, train_loss))
            start_time = time.time()
            file_w = open("/gpfs/share/home/1900011026/3D_deepSDF/data/model/trained_params.txt", "wb")
            pickle.dump(params, file_w)
    file_w = open("/gpfs/share/home/1900011026/3D_deepSDF/data/model/trained_params.txt", "wb")
    pickle.dump(params, file_w)
    onp.save(config['loss_record_path'], train_loss_record)


key = random.PRNGKey(1)
init_params, batch_forward = get_mlp(args)
_, net_params = init_params(key,(-1, args.point_dim + args.latent_len))
opt_init, opt_update, get_params = optimizers.adam(args.learning_rate)


if __name__ == '__main__':
    run_training_loop()

