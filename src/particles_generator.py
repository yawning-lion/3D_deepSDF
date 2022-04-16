import jax.numpy as np
import matplotlib
import matplotlib.pyplot as plt
from jax import jit
import numpy as onp
from jax import vmap, jit
from jax.numpy import pi, sin, cos
import jax
import trimesh
from jax import random
import meshio

config = {
    'mesh_path':'/home/ubuntu/DESKTOP/rsc/3D_deepSDF/data/data_set/sphere.msh',
    'num_shape':20,
    'mean':1
}


def get_mesh(config):
    mesh = meshio.read('/home/ubuntu/DESKTOP/rsc/3D_deepSDF/data/data_set/sphere.msh')
    faces = mesh.get_cells_type("triangle")
    points = mesh.points
    return np.array(points), np.array(faces)

def kernel(x1, x2):
    l = 0.5
    sigma = 0.1
    x = x1 - x2
    Range = 3.0
    return sigma**2 * np.exp(- (np.dot(x, x) / Range) / l**2)

array_kernel = vmap(kernel, in_axes = (0, 0), out_axes = (0))
matrix_kernel = jit(vmap(array_kernel, in_axes = (0, 0), out_axes = (0)))

def get_covarience_matrix(points):
    row, col = points.shape
    m1_points = np.expand_dims(points, 1).repeat(row, axis = 1)
    m2_points = np.transpose(m1_points, (1, 0, 2))
    cov = matrix_kernel(m1_points, m2_points)
    return cov

def get_mean(points, config):
    row, col = points.shape
    return np.ones(row) * config['mean']

def get_radius(config):
    points, faces = get_mesh(config)
    cov = get_covarience_matrix(points)
    mean = get_mean(points, config)
    row, col = points.shape
    radius = onp.random.default_rng().multivariate_normal(mean, cov, config['num_shape'])
    temp = onp.min(radius, axis = 1).reshape(config['num_shape'], 1)
    radius = radius + onp.abs(temp) * 1.5
    return radius, points, faces

def get_batch_verts(config):
    radius, points, faces = get_radius(config)
    points_temp = np.expand_dims(points, 0).repeat(config['num_shape'], 0)
    radius_temp = np.expand_dims(radius, 2).repeat(3, 2)
    batch_verts = np.multiply(points_temp, radius_temp)
    return batch_verts, faces, radius

def generate_particles(config):
    batch_verts, faces, radius = get_batch_verts(config)
    np.save('data/data_set/radius.npy', radius)
    np.save('data/data_set/batch_verts.npy', batch_verts)
    np.save('data/data_set/faces.npy', faces)

if __name__ == '__main__':
    generate_particles(config)