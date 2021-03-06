# -*- coding: utf-8 -*-
"""polygon_generator.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PT_hXgZTIQz7kkX1WVCXqNiuoio08anm
"""

import jax.numpy as np
import numpy as onp
from jax import vmap, jit
from jax.numpy import pi, sin, cos
import jax

!pip install trimesh

import trimesh

num_division = 10
num_shape = 4
len = num_division * (num_division - 2)

#@title 生成圆内接多面体

def get_normal_point(num_division, index, r = 1):
  dim_phi = num_division
  dim_theta = num_division - 2
  pos_phi = index % dim_phi
  pos_theta = index // dim_phi
  phi_seg = (2 * pi) / dim_phi
  theta_seg = pi / (dim_theta + 1)
  phase = np.where(pos_theta % 2, phi_seg / 2, 0)
  theta = (pos_theta + 1) * theta_seg
  phi = pos_phi * phi_seg + phase
  normal_point = np.array([theta, phi, r])
  return normal_point

get_normal_vertices = jit(vmap(get_normal_point, in_axes = (None, 0, None), out_axes = 0))

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

#根据总指标计算出theta和phi的位置，即得到了从一维返回到二维矩阵的映射
def compute_pos(pos_theta, pos_phi, num_division):
  return pos_theta * num_division + pos_phi

#按点所在位置计算出模式 找出另外两个点相连
def get_mode_faces(index, num_division, direct = 1):
  dim_phi = num_division
  dim_theta = num_division - 2
  pos_phi = index % dim_phi
  pos_theta = index // dim_phi
  mode = pos_theta % 2
  pos_phi_left = np.where(mode, pos_phi, (pos_phi - 1 + num_division) % dim_phi)
  pos_phi_right = np.where(mode, (pos_phi + 1) % dim_phi, pos_phi)
  vert_p_l = compute_pos(pos_theta + direct, pos_phi_left, num_division)
  vert_p_r = compute_pos(pos_theta + direct, pos_phi_right, num_division)
  faces = np.array([index, vert_p_l, vert_p_r])
  return faces

vmap_get_mode_faces = jit(vmap(get_mode_faces, in_axes = (0, None, None), out_axes = 0))

# 使顶点与下一层的点相连
def get_peak_faces(sign, num_division):
  len = num_division * (num_division - 2)
  p = sign + len
  offset = np.where(sign, len - num_division, 0) 
  a1 = np.ones(num_division) * p
  temp = np.arange(num_division)
  a2 = temp + offset
  a3 = (temp + 1) % num_division + offset
  faces = np.concatenate((a1.reshape(num_division, 1), a2.reshape(num_division, 1), a3.reshape(num_division, 1)), axis = 1)
  return faces

def get_normal_trimesh(num_division, r = 1):
  len = num_division * (num_division - 2)
  index_array = np.arange(len).reshape(len,)
  vertices_spherical = get_normal_vertices(num_division, index_array, r)
  index_fore = np.arange(len - num_division).reshape(len - num_division,)
  index_back = index_fore + num_division
  faces_fore = vmap_get_mode_faces(index_fore, num_division, 1)
  #jax.ops.index_update(faces_fore, np.index_exp[:, [0, 1]], faces_fore[:, [1, 0]])
  faces_back = vmap_get_mode_faces(index_back, num_division, -1)
  vert_peak = np.array([[0, 0, 1], [pi, 0, 1]])
  p1_faces = get_peak_faces(0, num_division)
  #jax.ops.index_update(p1_faces, np.index_exp[:, [0, 1]], p1_faces[:, [1, 0]])
  p2_faces = get_peak_faces(1, num_division)
  verts_spherical = np.concatenate((vertices_spherical, vert_peak), axis = 0)
  faces = np.concatenate((faces_fore, faces_back, p1_faces, p2_faces), axis = 0)
  verts = vmap_from_spherical_to_catesian(verts_spherical)
  return verts, faces

def kernel(x1, x2):
  sigma = 6
  x = x1 - x2
  k = np.exp(-np.dot(x, x) / (2 * sigma**2))
  return k

v1_kernel = vmap(kernel, in_axes = (0, 0), out_axes = 0)
m_kernel = jit(vmap(v1_kernel, in_axes = (0, 0), out_axes = 0))

def get_radius_sample(num_division, num_shape):
  verts, faces = get_normal_trimesh(num_division)
  len = num_division * (num_division -2) + 2
  mean = 1
  m1_verts = np.expand_dims(verts, 1).repeat(len, axis = 1)
  m2_verts = np.transpose(m1_verts, (1, 0, 2))
  cov_matrix = onp.array(m_kernel(m1_verts, m2_verts))
  mean_vector = onp.ones(len) * mean
  radius_samples = onp.random.multivariate_normal(mean_vector, cov_matrix, num_shape)
  assert onp.min(radius_samples) > 0, "Radius must be postive!"
  assert onp.max(radius_samples) < 2, "Radius too large!"
  return radius_samples, verts, faces

radius_samples, verts, faces = get_radius_sample(num_division, num_shape)

print(radius_samples.shape)
batch_verts = np.expand_dims(verts, 0).repeat(num_shape, 0)
print(batch_verts.shape)
radius_temp = np.expand_dims(radius_samples, 2).repeat(3, 2)
print(radius_temp.shape)
batch_verts = np.multiply(batch_verts, radius_temp)

def mesh_for_show(batch_verts, faces, shape):
  verts = batch_verts[shape]
  faces_re = faces[:, [1, 0, 2]]
  faces_ful = np.concatenate((faces, faces_re), axis = 0)
  v = verts.tolist()
  f = faces_ful.tolist()
  mesh = trimesh.Trimesh(vertices = v, faces = f)
  return mesh

def get_batch_verts(num_division, num_shape):
  radius_samples, verts, faces = get_radius_sample(num_division, num_shape)
  batch_verts = np.expand_dims(verts, 0).repeat(num_shape, 0)
  radius_temp = np.expand_dims(radius_samples, 2).repeat(3, 2)
  batch_verts = np.multiply(batch_verts, radius_temp)
  return batch_verts, faces

batch_verts, faces = get_batch_verts(12, 5)

mesh = mesh_for_show(batch_verts, faces, 0)
#mesh.show()