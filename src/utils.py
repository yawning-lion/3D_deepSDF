import numpy as onp
import jax.numpy as np
import matplotlib
import matplotlib.pyplot as plt
from jax import jit
from jax import vmap
from jax.scipy.special import logsumexp
from jax.experimental import optimizers
from torch.utils.data import Dataset, DataLoader
from .argument import args

class SDF_Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self,
                 path,
                 mode = 'train',
                 parse = args):
        self.mode = mode

        supervised_data = onp.load(path)[:]
        point = supervised_data[:, 0:-1]
        sdf = supervised_data[:, -1]


        if mode == 'infer':
            #when inference,data is boundary points, sdf = 0
            data = point[:]
            self.data = data
            self.target = sdf
        else:
            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(point)) if i % 8 != 0]
            elif mode == 'test':
                indices = [i for i in range(len(point)) if i % 8 == 0]
            

            self.data = point[indices]
            self.target = sdf[indices]

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of SDF Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        return self.data[index], self.target[index]
    
    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)

def SDF_dataloader(path, mode, args):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = SDF_Dataset(path, mode = mode, parse = args)  # Construct dataset
    #print(dataset[0])
    dataloader = DataLoader(
        dataset, args.batch_size,
        shuffle=(mode != 'test'), drop_last = False,
        num_workers = args.n_jobs, pin_memory = False)                            # Construct dataloader
    return dataloader



def plot_learning_curve(loss_record_path, mode):
    ''' Plot learning curve of your DNN (train & test & infer loss) '''
    loss = onp.load(loss_record_path)
    total_steps = len(loss)
    x_1 = range(total_steps)
    plt.figure(figsize = (6, 4))
    plt.plot(x_1, loss, c='tab:red', label = mode)
    plt.ylim(-4., 4.)
    plt.xlabel('steps')
    plt.ylabel('logged MSE loss')
    plt.title('Learning curve of {}'.format(mode))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    '''
	test_tr_set = SDF_Dataset(path = '/home/yawnlion/Desktop/PYproject/2D_deepSDF/data/data_set/supervised_data.npy',
		                mode = 'train',
		                parse = args)
    '''
    path = '/home/yawnlion/Desktop/PYproject/2D_deepSDF/data/data_set/supervised_data.npy'
    test_loader1 = SDF_dataloader(path,
		                        'train',
		                        args)
    test_loader2 = SDF_dataloader(path,
		                        'test',
		                        args)
    test_loader3 = SDF_dataloader(path,
		                        'infer',
                                args)
