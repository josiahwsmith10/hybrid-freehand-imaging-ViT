import torch
from scipy.constants import pi
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

class HFFH_ViT_Data:
    def __init__(self, args) -> None:
        self.args = args
    
    def load_data(self, filenames, matlab_var_name):
        d = {}
        num_samples = 0
        for filename in filenames:
            d[filename] = sio.loadmat(f"./matlab_data/{filename}.mat")[matlab_var_name]
            print(f"loaded {matlab_var_name} from {filename}")
            Nx = d[filename].shape[0]
            Ny = d[filename].shape[1]
            N = d[filename].shape[2]
            num_samples += N
            
        out = np.zeros((Nx, Ny, num_samples))
            
        for ind, key in enumerate(d):
            ind_start = N*ind
            ind_end = N*(ind+1)
            out[:, :, ind_start:ind_end] = np.abs(d[key])
        
        ind_shuffle = np.random.permutation(num_samples)
        out = out[:, :, ind_shuffle]
        return out
    
    def load_train_data(self, filenames):
        data_all_lr = self.load_data(filenames=filenames, matlab_var_name='radarImagesFFH')
        data_all_hr = self.load_data(filenames=filenames, matlab_var_name='idealImages')
        return data_all_lr, data_all_hr
    
    def load_test_data(self, filenames):
        data_all_FFH = self.load_data(filenames=filenames, matlab_var_name='radarImagesFFH')
        data_all_RMA = self.load_data(filenames=filenames, matlab_var_name='radarImagesRMA')
        data_all_BPA = self.load_data(filenames=filenames, matlab_var_name='radarImagesBPA')
        data_all_hr = self.load_data(filenames=filenames, matlab_var_name='idealImages')
        return data_all_FFH, data_all_RMA, data_all_BPA, data_all_hr
    
    def create_dataset(self, lr, hr, device=None):
        X = lr.transpose((2, 0, 1)).reshape((lr.shape[2], 1, lr.shape[0], lr.shape[1]))
        Y = hr.transpose((2, 0, 1)).reshape((hr.shape[2], 1, hr.shape[0], hr.shape[1]))
        
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        
        if not device is None:
            ds = torch.utils.data.TensorDataset(X.to(device, dtype=torch.float), Y.to(device, dtype=torch.float))
        else:
            ds = torch.utils.data.TensorDataset(X.to(dtype=torch.float), Y.to(dtype=torch.float))        
        return ds
    
    def create_train_dataset(self, split=(3296, 800), filenames=["hffh_ViT_solid_w_points_2048_training", "hffh_ViT_solid_w_points_2048_training_v2"]):
        assert len(split) == 2, 'split must have a length of 2'
        
        self.num_train, self.num_val = split
        
        lr, hr = self.load_train_data(filenames=filenames)
        
        assert sum(split) <= lr.shape[2] and sum(split) <= hr.shape[2], f'split cannot exceed {min(lr.shape[2], hr.shape[2])}'
        
        self.dataset_train = self.create_dataset(lr[:split[0], :, :], hr[:split[0], :, :], self.args['device'])
        self.dataset_val = self.create_dataset(lr[split[0]:(split[0]+split[1]), :, :], hr[split[0]:(split[0]+split[1]), :, :], self.args['device'])
        
    
    def create_test_dataset(self, filenames=["hffh_ViT_solid_w_points_1024_testing"]):
        ffh, rma, bpa, hr = self.load_test_data(filenames=filenames)
    
