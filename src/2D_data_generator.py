import numpy as onp
import jax.numpy as np
import jax
import matplotlib
import matplotlib.pyplot as plt
from jax import jit
from jax import vmap
import argparse
from .argument import args

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


def squareSDF(x, y):
    '''
    given a point as (x,y)
    return its SDF to a square
    '''
    point = np.array([x, y])
    square_corners = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
    square_corners_rolled = np.roll(square_corners, 1, axis=0)
    pointO = np.array([0., 0.])
    sign = np.where(np.any(sign_to_line_segs(point, pointO, square_corners, square_corners_rolled)), -1., 1.)
    sdf = np.min(d_to_line_segs(point,square_corners, square_corners_rolled)) * sign
    return sdf

def circleSDF(x, y):
    return np.sqrt(x**2 + y**2) - 1

def triangleSDF(x, y):
    '''
    given a point as (x,y)
    return its SDF to a triangle which is determined by assigned boundry point in corners
    '''
    point = np.array([x, y])
    triangle_corners = np.array([[1., 0.5], [-1., 0.5], [0., -1.]])
    triangle_corners_rolled = np.roll(triangle_corners, 1, axis=0)
    pointO = np.array([0., 0.])
    sign = np.where(np.any(sign_to_line_segs(point, pointO, triangle_corners, triangle_corners_rolled)), -1., 1.)
    sdf = np.min(d_to_line_segs(point,triangle_corners, triangle_corners_rolled)) * sign
    return sdf

def get_angles(num_division):
    return onp.linspace(0, 2 * onp.pi, num_division + 1)[:-1]
# angle average split

def generate_radius_samples(num_shape, num_division=64):
    '''Generate multivariate Gaussian samples.
    Each sample is a vector of radius.

    Returns
    -------
    Numpy array of shape (num_samples, num_division)
    '''

    def kernel(x1, x2):
        '''Periodic kernel
        '''
        sigma = 0.2
        l = 0.4
        p = 2 * onp.pi
        k = sigma**2 * onp.exp(-2 * onp.sin(onp.pi * onp.absolute(x1 - x2) / p)**2 / l**2)
        return k
 
    def mean(x):
        return 1.

    angles = get_angles(num_division)
 
    kernel_matrix = onp.zeros((num_division, num_division))
    mean_vector = onp.zeros(num_division)

    for i in range(num_division):
        mean_vector[i] = mean(angles[i])
        for j in range(num_division):
            kernel_matrix[i][j] = kernel(angles[i], angles[j])

    radius_samples = onp.random.multivariate_normal(mean_vector, kernel_matrix, num_shape)

    assert onp.min(radius_samples) > 0, "Radius must be postive!"
    assert onp.max(radius_samples) < 2, "Radius too large!"

    return radius_samples

def compute_boundary_points(radius_samples):
    '''For each boundary point in each radius_sample, we compute the coordinates at 
    which the boundary loss will be evaluated
    
    Returns
    -------
    Numpy array of shape (num_samples, num_division, dim)
    '''
    angles = get_angles(len(radius_samples[0]))
    x = radius_samples * onp.cos(angles)
    y = radius_samples * onp.sin(angles)
    boundary_points = onp.stack([x, y], axis=2)
    return boundary_points

def shapeSDF(x, y, boundary_points):
    '''
    give points in the form of separate x,y ,for it's convenient to apply vmap

    '''
    point = np.array([x, y])
    boundary_points = np.array(boundary_points)
    boundary_points_rolled = np.roll(boundary_points, 1, axis=0)
    pointO = np.array([0., 0.])
    sign = np.where(np.any(sign_to_line_segs(point, pointO, boundary_points, boundary_points_rolled)), -1., 1.)
    sdf = np.min(d_to_line_segs(point,boundary_points, boundary_points_rolled)) * sign
    return sdf

def supervised_point_generator(num_shape, num_point, num_division):
    '''
    generate supervised data:
    num:num_shape * num_point
    return shape: list[num , (num_dim + 11)],num
    entry formation: point, index, sdf
    '''
    radius_samples = generate_radius_samples(num_shape, num_division)
    batch_boundary_points = compute_boundary_points(radius_samples)
    onp.save('/home/yawnlion/Desktop/PYproject/2D_deepSDF/data/data_set/train_boundary_point.npy', batch_boundary_points)
    point_tmp = onp.random.uniform(-2, 2, size = (num_point, 2))
    x = point_tmp[:, 0]
    y = point_tmp[:, 1]# should be replaced by batched eikonal point
    fullshapeSDF = vmap(shapeSDF, in_axes=(0, 0, None))
    batch_fullshapeSDF = vmap(fullshapeSDF, in_axes=(None, None, 0))
    batch_sdf = batch_fullshapeSDF(x, y, batch_boundary_points)
    sdf = batch_sdf.reshape(-1)
    sdf = sdf.reshape(sdf.size,1)
    point = np.tile(point_tmp, (num_shape, 1))
    shape_index = np.arange(num_shape)
    shape_index = np.repeat(shape_index, num_point)
    shape_index = shape_index.reshape(shape_index.size, 1)
    supervised_data = np.concatenate([point, shape_index, sdf], 1)

    return supervised_data


def infer_data_generator(num_shape, num_division):
	radius_samples = generate_radius_samples(num_shape, num_division)
	batch_boundary_points = compute_boundary_points(radius_samples)
	#print(batch_boundary_points)
	onp.save('/home/yawnlion/Desktop/PYproject/2D_deepSDF/data/data_set/infer_boundary_point.npy',
    		batch_boundary_points)
	shape = batch_boundary_points.shape
	size_len = shape[0] * shape[1]
	batch_boundary_points = batch_boundary_points.reshape(size_len, shape[2])
	#print(batch_boundary_points)
	shape_index = np.arange(num_shape)
	shape_index = np.repeat(shape_index, num_division)
	shape_index = shape_index.reshape(shape_index.size, 1)
	sdf = np.zeros(size_len).reshape(size_len,1)
	infer_data = np.concatenate([batch_boundary_points, shape_index, sdf], 1)
	onp.save('/home/yawnlion/Desktop/PYproject/2D_deepSDF/data/data_set/infer_data.npy', 
			infer_data)

if __name__ == "__main__":
    supervised_data = supervised_point_generator(args.num_shape_train, args.num_point, args.num_division)
    onp.save('/home/yawnlion/Desktop/PYproject/2D_deepSDF/data/data_set/supervised_data.npy', supervised_data)
    infer_data_generator(args.num_shape_infer, args.num_division)