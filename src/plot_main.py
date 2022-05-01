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
from .particles_generator import get_batch_verts
import trimesh
from .test_shape import mesh_for_show
from mpl_toolkits.mplot3d import Axes3D

config = {'data_path':'/home/yawnlion/Desktop/PYproject/3D_deepSDF/data/data_set/supervised_data.npy',
        'mode':'train',
        'loss_record_path':'/home/yawnlion/Desktop/PYproject/3D_deepSDF/data/model/train_loss_record.npy'}

#plot_learning_curve(config['loss_record_path'], config['mode'])


infer_config = {
    'num_shape' : 3,
    'mesh_path':'/home/yawnlion/Desktop/PYproject/3D_deepSDF/data/data_set/sphere.msh',
    'mean' : 1
}

infer_config['num_shape'] = args.num_shape_infer

def get_infer_data(config):
    batch_verts, faces, radius = get_batch_verts(config)
    np.save('data/data_set/infer_radius.npy', radius)
    np.save('data/data_set/infer_batch_verts.npy', batch_verts)
    np.save('data/data_set/infer_faces.npy', faces)
    shape = batch_verts.shape
    size_len = shape[0] * shape[1]
    batch_verts = batch_verts.reshape(size_len, shape[2])
    #print(batch_boundary_points)
    
    shape_index = np.arange(config['num_shape'])
    shape_index = np.repeat(shape_index, shape[1])
    shape_index = shape_index.reshape(shape_index.size, 1)
    sdf = np.zeros(size_len).reshape(size_len,1)
    infer_data = np.concatenate([batch_verts, shape_index, sdf], 1)
    #disturbed_data = vmap_disturb_boundary(infer_boundary, 10)
    #shape_data = disturbed_data.shape
    #disturbed_data = disturbed_data.reshape(shape_data[0]*shape_data[1], shape_data[2])
    #infer_data = np.concatenate([disturbed_data, infer_boundary], 0)
    print(infer_data.shape)
    print(infer_data[0])
    onp.save('data/data_set/infer_data.npy', 
            infer_data)

if __name__ == '__main__':
    #get_infer_data(infer_config)
    mode = 'train'
    boundary = np.load("/home/yawnlion/Desktop/PYproject/3D_deepSDF/data/data_set/{}_batch_verts.npy".format(mode))
    seeds = np.load("/home/yawnlion/Desktop/PYproject/3D_deepSDF/data/data_set/{}_seeds.npy".format(mode))
    shape = 2
    verts1 = boundary[shape]
    seeds1 = seeds[shape]
    print(verts1.shape)
    cloud1 = trimesh.points.PointCloud(seeds1)
    #cloud1.show()
    cloud1_scene = cloud1.scene()
    cloud1_scene.show()
    faces = np.load("/home/yawnlion/Desktop/PYproject/3D_deepSDF/data/data_set/faces.npy".format(mode))
    mesh2 = mesh_for_show(boundary, faces, shape)
    cloud1_scene.add_geometry(mesh2)
    cloud1_scene.show()
    #plot_learning_curve(config['loss_record_path'], 'train')
    
    supervised_data = np.load("/home/yawnlion/Desktop/PYproject/3D_deepSDF/data/data_set/supervised_data.npy")
    query = np.load('data/data_set/test_query.npy')
    query1 = query[1, :1000]

    data1 = supervised_data[0:12800]
    point = query1[:, 0:3]
    label = data1[:, -1].reshape(-1)
    fig = plt.figure()
    ax = Axes3D(fig)
    #ax.scatter(point[:, 0], point[:,1], point[:, 2], c = [label], cmap = 'viridis', alpha = 0.2)
    #ax.scatter(point[:, 0], point[:,1], point[:, 2], alpha = 0.2)
    #ax.scatter(point[:, 0], point[:,1], point[:, 2], alpha = 0.2)
    #plt.show()
    print(supervised_data.shape)
    print(point[:, 0].max())
    print(point[:, 1].max())
    print(point[:, 2].max())
