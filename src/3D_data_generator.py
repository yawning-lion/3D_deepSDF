import numpy as onp
import jax.numpy as np
import jax
from jax.numpy import pi, cos, sin
import matplotlib
import matplotlib.pyplot as plt
from jax import jit
from jax import vmap
import argparse
from jax.numpy.linalg import norm
from jax import random
from .argument import args

config = {
    'num_query':12800,
    'num_shape':20,
    'batch_size':5
}
config['num_shape'] = args.num_shape_train


infer_cofig = {
    'num_shape' : 3,
    'mesh_path':'/home/ubuntu/DESKTOP/rsc/3D_deepSDF/data/data_set/sphere.msh'
}

infer_cofig['num_shape'] = args.num_shape_infer

def d_to_line_seg(P, A, B):
    '''Distance of a point P to a line segment AB'''
    AB = B - A
    BP = P - B
    AP = P - A
    AB_BP = np.dot(AB, BP)
    AB_AP = np.dot(AB, AP)
    mod = np.sqrt(np.sum(AB**2))
    tmp2 = np.absolute(np.cross(AB, AP)) / mod
    tmp1 = np.where(AB_AP < 0., np.sqrt(np.sum(AP**2)), tmp2)
    return np.where(AB_BP > 0., np.sqrt(np.sum(BP**2)), tmp1)

#P,A,B in type/size :deviceArray([a,b])

d_to_line_segs = jit(vmap(d_to_line_seg, in_axes=(None, 0, 0), out_axes=0))

'''
 vmap the d_to_line_segs in axes (none,0,0):,P in type :deviceArray([a,b])
 A,B with the type deviceArray([[a,b]...])
'''

def sign_to_line_seg(P, O, A, B):
    ''' 
    If P is inside the triangle OAB, return True, otherwise return False.
    '''
    OA = A - O
    OB = B - O
    OP = P - O
    AB = B - A
    AP = P - A
    OAxOB = np.cross(OA, OB)
    OAxOP = np.cross(OA, OP)
    OBxOP = np.cross(OB, OP)
    OAxAB = np.cross(OA, AB)
    ABxAP = np.cross(AB, AP)
    tmp2 = np.where(ABxAP * OAxAB < 0., False, True)
    tmp1 = np.where(OAxOB * OBxOP > 0., False, tmp2)
    return  np.where(OAxOB * OAxOP < 0., False, tmp1)

#P,A,B with the type :deviceArray([a,b])

sign_to_line_segs = jit(vmap(sign_to_line_seg, in_axes=(None, None, 0, 0), out_axes=0))

def d_to_line_seg(P, A, B):
    '''
    Distance of a point P to a line segment AB
    '''
    BA = B - A
    PB = P - B
    PA = P - A
    tmp1 = np.where(np.dot(BA, PA) < 0., norm(PA), norm(np.cross(BA, PA)) / norm(BA))
    return np.where(np.dot(BA, PB) > 0., norm(PB), tmp1)


def d_to_triangle(P, face_verts):
    '''
    Distance of a point P to a triangle (P1, P2, P3)
    Reference: https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle
    '''
    P1 = face_verts[0]
    P2 = face_verts[1]
    P3 = face_verts[2]
    u = P2 - P1
    v = P3 - P1
    w = P3 - P2
    n = np.cross(u, v)
    r = P - P1  
    s = P - P2
    t = P - P3

    n_square = np.sum(n*n)
    c3 = np.dot(np.cross(u, r), n) / n_square
    c2 = np.dot(np.cross(r, v), n) / n_square
    c1 = 1 - c3 - c2

    d1 = d_to_line_seg(P, P2, P3)
    d2 = d_to_line_seg(P, P1, P3)
    d3 = d_to_line_seg(P, P1, P2)
    d = np.min(np.array([d1, d2, d3]))

    tmp2 = np.where(c3 < 0., d, norm(P - (c1*P1 + c2*P2 + c3*P3)))
    tmp1 = np.where(c2 < 0., d, tmp2)
    return np.where(c1 < 0., d, tmp1)

def sign_to_tetrahedron(P, face_verts):
    ''' 
    If P is inside the tetrahedron (O, D, E, F), return True, otherwise return False.
    '''
    O = np.array([0., 0., 0.])
    D = face_verts[0]
    E = face_verts[1]
    F = face_verts[2]
    DO = D - O
    EO = E - O
    FO = F - O
    ED = E - D
    FD = F - D
    OD = O - D
    PO = P - O 
    PD = P - D
    tmp3 = np.where(np.dot(np.cross(ED, FD), OD)*np.dot(np.cross(ED, FD), PD) < 0., False, True)
    tmp2 = np.where(np.dot(np.cross(EO, FO), DO)*np.dot(np.cross(EO, FO), PO) < 0., False, tmp3)
    tmp1 = np.where(np.dot(np.cross(DO, FO), EO)*np.dot(np.cross(DO, FO), PO) < 0., False, tmp2)
    return np.where(np.dot(np.cross(DO, EO), FO)*np.dot(np.cross(DO, EO), PO) < 0., False, tmp1)

d_to_triangles = vmap(d_to_triangle, in_axes = (None, 0), out_axes = 0)
sign_to_tetrahedrons = vmap(sign_to_tetrahedron, in_axes = (None, 0), out_axes = 0)

@jit
def sdf_to_polygon(P, verts, faces):
    d = d_to_triangles(P, verts[faces]).min()
    sign = np.where(np.any(sign_to_tetrahedrons(P, verts[faces])), -1, 1)
    return d * sign
# compute sdf of a point to a shape
query_to_polygon = vmap(sdf_to_polygon, in_axes = (0, None, None), out_axes = 0)
# compute sdf of a batch of point to a shape
batch_to_polygon = jit(vmap(query_to_polygon, in_axes = (0, 0, None), out_axes = 0))
# compute sdf of a batch point to  batched shapes


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

'''
def get_query(Min, Max, config):
    lower = np.where((Min - 0.3) > 0, Min - 0.3, 0)
    upper = Max + 0.3
    r_query = onp.random.uniform(lower, upper, (config['num_query'], 1))
    onp.random.shuffle(r_query)
    theta_query = onp.random.uniform(0, pi, (config['num_query'], 1))
    onp.random.shuffle(theta_query)
    phi_query = onp.random.uniform( 0, 2 * pi, (config['num_query'], 1))
    onp.random.shuffle(phi_query)
    sphere_query = np.concatenate([r_query, theta_query, phi_query], 1)
    query = vmap_from_spherical_to_catesian(sphere_query)
    return query
'''

def get_query (Min, Max, config):
    query = onp.random.uniform(-3., 3., (config['num_query'], 3))
    return query
    
vmap_get_query = vmap(get_query, in_axes = (0,0, None), out_axes = 0)

#loop, because of the memory limit
def get_supervised_data(config):
    radius = np.load('data/data_set/radius.npy')
    batch_verts = np.load('data/data_set/train_batch_verts.npy')
    faces = np.load('data/data_set/faces.npy')
    Min = np.min(radius, 1).reshape(config['num_shape'], )
    Max = np.max(radius, 1).reshape(config['num_shape'], )
    query = []
    for i in range(config['num_shape']):
        single_query = get_query(Min[i], Max[i], config)
        query.append(single_query)
        
    query = np.asarray(query)
    Len = config['num_shape'] * config['num_query']
    batch_len = config['batch_size'] * config['num_query']
    batch_query = query[0:config['batch_size']]
    batch_batch_verts = batch_verts[0:config['batch_size']]
    batch_sdf = batch_to_polygon(batch_query, batch_batch_verts, faces).reshape(batch_len, 1)
    for i in range(config['num_shape'] // config['batch_size'] - 1):
        start = (i + 1) * config['batch_size']
        end = (i + 2) * config['batch_size']
        batch_query = query[start:end]
        batch_batch_verts = batch_verts[start:end]
        sdf = batch_to_polygon(batch_query, batch_batch_verts, faces).reshape(batch_len, 1)
        batch_sdf = np.concatenate([batch_sdf, sdf], 0)
    shape_index = np.arange(config['num_shape'])
    shape_index = np.repeat(shape_index, config['num_query'])
    shape_index = shape_index.reshape(Len, 1)
    query = query.reshape(Len, 3)
    supervised_data = np.concatenate([query, shape_index, batch_sdf], 1)
    np.save('data/data_set/supervised_data.npy', supervised_data)
    print('{} entry has generated,sample here{}'.format(Len, supervised_data[0]))



def infer_data_generator(num_shape, num_division, ):
    radius_samples = generate_radius_samples(num_shape, num_division)
    batch_boundary_points = compute_boundary_points(radius_samples)
    #print(batch_boundary_points)
    onp.save('data/data_set/infer_batch_verts.npy',
            batch_boundary_points)
    shape = batch_boundary_points.shape
    size_len = shape[0] * shape[1]
    batch_boundary_points = batch_boundary_points.reshape(size_len, shape[2])
    #print(batch_boundary_points)
    
    shape_index = np.arange(num_shape)
    shape_index = np.repeat(shape_index, num_division)
    shape_index = shape_index.reshape(shape_index.size, 1)
    sdf = np.zeros(size_len).reshape(size_len,1)
    infer_boundary = np.concatenate([batch_boundary_points, shape_index, sdf], 1)
    disturbed_data = vmap_disturb_boundary(infer_boundary, 10)
    shape_data = disturbed_data.shape
    disturbed_data = disturbed_data.reshape(shape_data[0]*shape_data[1], shape_data[2])
    infer_data = np.concatenate([disturbed_data, infer_boundary], 0)
    print(infer_data.shape)
    print(infer_data[0])
    onp.save('/gpfs/share/home/1900011026/2D_deepSDF/data/data_set/infer_data.npy', 
            infer_boundary)






if __name__ == '__main__':
    get_supervised_data(config)

