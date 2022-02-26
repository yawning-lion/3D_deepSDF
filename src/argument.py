import argparse


ap=argparse.ArgumentParser()
ap.add_argument('--num_dim', type = int, default = 3)
ap.add_argument('--num_shape_train', type = int, default = 20)
# the amount of shapes in train set
ap.add_argument('--num_shape_infer', type = int, default = 3)
ap.add_argument('--num_point', type = int, default = 1280)
# the amount of points in every shape , randomly sampling
ap.add_argument('--shape_index', type = int, default = 0, help = "choose from 0,1,2...default 0")
# for each shape, assign a number to characterize it ,make up a component of in array
ap.add_argument('--activation', choices = ['tanh', 'selu', 'relu'], default = 'relu')
# select activate function in console
ap.add_argument('--num_division', type=int, default = 9)
#differently from 2D situation, denote the division of phi and theta
ap.add_argument('--num_epochs', type = int, default = 256)
ap.add_argument('--convariance', type = int, default = 100)
ap.add_argument('--latent_len', type = int , default = 16)
ap.add_argument('--width_hidden', type = int, default = 256)
ap.add_argument('--out_len', type = int, default = 1)
ap.add_argument('--learning_rate', type = float, default = 0.0075)
ap.add_argument('--batch_size', type = int, default = 256)
ap.add_argument('--point_dim', type = int, default = 2)
ap.add_argument('--n_hidden', type = int, default = 2)
ap.add_argument('--skip', action = 'store_true', default = True)
ap.add_argument('--n_skip', type = int, default = 2)
ap.add_argument('--n_jobs', type = int, default = 0)
args=ap.parse_args()

if __name__ == '__main__':
	print(args.num_dim)