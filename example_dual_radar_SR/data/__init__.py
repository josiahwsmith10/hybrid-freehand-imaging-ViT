import torch
from scipy.constants import pi
import numpy as np
import matplotlib.pyplot as plt
from model.common import CPLX
import scipy.io as sio

class Data:
    def __init__(self, dr, args):
        self.dr = dr
        self.args = args
        
    def create_dataset_train(self, num_train, max_targets):
        self.max_targets = max_targets
        
        self.num_train = num_train
        
        self.create_XY(num_train)
        
        self.train_X = self.X
        self.train_Y = self.Y
        self.train_Z = self.Z
        self.train_SNR = self.SNR
        
        X = self.X.reshape((num_train, 1, self.dr.Nk_fb))
        if self.args['xyz_str'].lower() == 'xy':
            Y = self.Y.reshape((num_train, 1, self.dr.Nk_fb))
        elif self.args['xyz_str'].lower() == 'xz':
            Y = self.Z.reshape((num_train, 1, self.dr.Nk_fb))
        
        X = torch.cat((torch.real(X), torch.imag(X)), axis=1).to(self.args['device'])
        Y = torch.cat((torch.real(Y), torch.imag(Y)), axis=1).to(self.args['device'])
        
        self.dataset_train = torch.utils.data.TensorDataset(X, Y)
        
    def create_dataset_val(self, num_val, max_targets):
        self.max_targets = max_targets
        
        self.num_val = num_val
        
        self.create_XY(num_val)
        
        self.val_X = self.X
        self.val_Y = self.Y
        self.val_Z = self.Z
        self.val_SNR = self.SNR
        
        X = self.X.reshape((num_val, 1, self.dr.Nk_fb))
        if self.args['xyz_str'].lower() == 'xy':
            Y = self.Y.reshape((num_val, 1, self.dr.Nk_fb))
        elif self.args['xyz_str'].lower() == 'xz':
            Y = self.Z.reshape((num_val, 1, self.dr.Nk_fb))
        
        X = torch.cat((torch.real(X), torch.imag(X)), axis=1).to(self.args['device'])
        Y = torch.cat((torch.real(Y), torch.imag(Y)), axis=1).to(self.args['device'])
        
        self.dataset_val = torch.utils.data.TensorDataset(X, Y)
        
    def create_dataset_test(self, num_test, max_targets):
        self.max_targets = max_targets
        
        self.num_test = num_test
        
        self.create_XY(num_test)
        
        self.test_X = self.X
        self.test_Y = self.Y
        self.test_Z = self.Z
        self.test_SNR = self.SNR
        
        X = self.X.reshape((num_test, 1, self.dr.Nk_fb))
        if self.args['xyz_str'].lower() =='xy':
            Y = self.Y.reshape((num_test, 1, self.dr.Nk_fb))
        elif self.args['xyz_str'].lower() == 'xz':
            Y = self.Z.reshape((num_test, 1, self.dr.Nk_fb))
        
        X = torch.cat((torch.real(X), torch.imag(X)), axis=1)
        Y = torch.cat((torch.real(Y), torch.imag(Y)), axis=1)
        
        self.dataset_test = torch.utils.data.TensorDataset(X, Y)
       
    def create_XY(self, num):
        def linear_power(x, N):
            return torch.sqrt(1/N * torch.sum(torch.abs(x)**2))
        
        X = torch.zeros((num, self.dr.Nk_fb), dtype=torch.complex64)
        Y = torch.zeros((num, self.dr.Nk_fb), dtype=torch.complex64)
        Z = torch.zeros((num, self.dr.Nk_fb), dtype=torch.complex64)
        SNR = np.zeros(num)
        
        k = self.dr.k_HR.reshape((1, self.dr.Nk_fb))
        
        for ind_num in range(num):
            # Handle noise
            SNR[ind_num] = 50*np.random.rand(1)
            n = torch.randn((2*self.dr.Nk), dtype=torch.complex64)
            n_pow = linear_power(n, 2*self.dr.Nk)
            
            # Handle signal
            num_targets = np.random.randint(1, self.max_targets+1)
            R = 0.05*self.dr.r0.range_max_m + 0.9*self.dr.r0.range_max_m*torch.rand(num_targets, 1).reshape((num_targets, 1))
            
            amps = torch.randn(num_targets, 1).reshape((num_targets, 1))
            theta = 2*pi*torch.rand(num_targets, 1).reshape((num_targets, 1))
            
            amps = (0.5 + torch.abs(amps)) * torch.exp(1j*theta)
            
            x_HR = torch.sum(amps * torch.exp(1j*2*k*R), axis=0)
            x_LR = x_HR.clone()
            x_LR[self.dr.Nk:-self.dr.Nk] = torch.zeros(self.dr.Nk_fb - 2*self.dr.Nk, dtype=torch.complex64)
            
            x_LR_pow = linear_power(x_LR, 2*self.dr.Nk)
            x_HR_pow = linear_power(x_HR, self.dr.Nk_fb)
            
            n = n/n_pow*x_LR_pow*10**(-SNR[ind_num]/10)
            
            x_LR[:self.dr.Nk] += n[:self.dr.Nk]
            x_LR[-self.dr.Nk:] += n[-self.dr.Nk:]
            
            x_LR_pow = linear_power(x_LR, 2*self.dr.Nk)
            x_HR_pow = linear_power(x_HR, self.dr.Nk_fb)
            
            #x_LR = x_LR/x_LR_pow
            #x_HR = x_HR/x_HR_pow
            
            y_LR = torch.fft.fft(x_LR, self.dr.Nk_fb, dim=0, norm="ortho")
            y_HR = torch.fft.fft(x_HR, self.dr.Nk_fb, dim=0, norm="ortho")
            
            X[ind_num] = y_LR # Frequency LR
            Y[ind_num] = y_HR # Frequency HR
            Z[ind_num] = x_HR # Time HR
            
        self.X = X
        self.Y = Y
        self.Z = Z
        self.SNR = SNR
        
    def test_net(self, args, model, data):
        ind_rand = np.random.randint(0, data.num_test)
        
        print(f"SNR = {self.test_SNR[ind_rand]}")

        (X, Y) = data.dataset_test[ind_rand]
        x_in = X.to(args['device'])

        x_in = CPLX(x_in[0, :].reshape((1, 1, -1)), x_in[1, :].reshape((1, 1, -1)))
        Y_pred = model.model(x_in)
        y_pred = (Y_pred.r + 1j*Y_pred.i).cpu().detach().numpy().reshape(-1)

        x = X[0] + 1j*X[1]
        y = Y[0] + 1j*Y[1]
        
        if args['xyz_str'].lower() == 'xz':
            y = np.fft.fft(y, norm="ortho")
            y_pred = np.fft.fft(y_pred, norm="ortho")
        else:
            y = y.numpy()

        fig = plt.figure(figsize=[24, 16])
        plt.subplot(241)
        plt.plot(np.abs(x))
        plt.title("Input " + str(ind_rand))

        plt.subplot(242)
        plt.plot(np.abs(y_pred))
        plt.title("Output")

        plt.subplot(243)
        plt.plot(np.abs(y))
        plt.title("Ground Truth")

        plt.subplot(244)
        err = np.abs(y - y_pred)
        plt.plot(err)
        plt.title("Error")

        x = np.fft.ifft(x, norm="ortho")
        y = np.fft.ifft(y, norm="ortho")
        y_pred = np.fft.ifft(y_pred, norm="ortho")            
            
        plt.subplot(245)
        plt.plot(np.real(x))
        plt.title("Input")

        plt.subplot(246)
        plt.plot(np.real(y_pred))
        plt.title("Output")

        plt.subplot(247)
        plt.plot(np.real(y))
        plt.title("Ground Truth")

        plt.subplot(248)
        err = y - y_pred
        plt.plot(np.real(err))
        plt.title("Error")

        plt.show()
        
class MATLAB_Data():
    def __init__(self, dr, args):
        self.dr = dr
        self.args = args
        
    def load_matlab_data(
        self,
        filenames=["cutout1", "cutout2", "circle", "diamond", "square", "star", "triangle", "knife"]
        ):
        d = {}
        num_samples = 0
        for filename in filenames:
            d[filename] = sio.loadmat(f"./matlab_data/data/{filename}.mat")['sarData']
            print(f"loaded {filename}")
            N = d[filename].shape[0]
            Nk = d[filename].shape[1]
            num_samples += N
            
        data_all = np.zeros((num_samples, Nk), dtype=np.complex64)
            
        for ind, key in enumerate(d):
            ind_start = N*ind
            ind_end = N*(ind+1)
            data_all[ind_start:ind_end, :] = d[key]
            
        del(d)
            
        return data_all, num_samples

    def create_datasets_matlab(
        self,
        is_noise=True,
        split=[200000, 1000, 1000],
        filenames=["cutout1", "cutout2", "circle", "diamond", "square", "star", "triangle", "knife"]
        ):
        def linear_power(x, N):
            return torch.sqrt(1/N * torch.sum(torch.abs(x)**2))
        
        data_all, num_samples = self.load_matlab_data(filenames)
            
        # Shuffle and select data
        print("shuffling and selecting data...")
        ind_shuffle = np.random.permutation(num_samples)
        
        num_samples = (split[0]+split[1]+split[2])
        data_all = data_all[ind_shuffle, :][:num_samples, :]
        
        x_HR = torch.from_numpy(data_all)
        x_LR = x_HR.clone()
        x_LR[:,self.dr.Nk:-self.dr.Nk] = torch.zeros((num_samples, self.dr.Nk_fb - 2*self.dr.Nk), dtype=torch.complex64)
        
        # Normalize power of all HR samples
        print("normalizing HR...")
        for ind, hr in enumerate(x_HR):
            hr_pow = linear_power(hr, self.dr.Nk_fb)
            x_HR[ind, :] = hr/hr_pow
            
        # Normalize power of all LR samples
        print("normalizing LR...")
        for ind, lr in enumerate(x_LR):
            lr_pow = linear_power(lr, 2*self.dr.Nk)
            x_LR[ind, :] = lr/lr_pow
        
        # Add Guassian noise to LR samples
        if is_noise:
            print("adding noise...")
            SNR = np.zeros(num_samples)
            for ind, lr in enumerate(x_LR):
                # Select SNR from 0 dB - 50 dB
                SNR[ind] = 50*np.random.rand(1)
                
                # Generate noise
                n = torch.randn((2*self.dr.Nk), dtype=torch.complex64)
                n_pow = linear_power(n, 2*self.dr.Nk)
                
                # Scale noise power
                n = n/n_pow*10**(-SNR[ind]/10)
                
                # Add noise to signal
                lr[:self.dr.Nk] += n[:self.dr.Nk]
                lr[-self.dr.Nk:] += n[-self.dr.Nk:]
                
                # Set signal back
                x_LR[ind, :] = lr
                
        print("taking X FFT...")
        X = torch.fft.fft(x_LR, self.dr.Nk_fb, dim=-1, norm="ortho").reshape((num_samples, 1, self.dr.Nk_fb))
        Y = x_HR
        
        # Get either xy or xz:
        print("getting Y...")
        if self.args['xyz_str'].lower() =='xy':
            Y = torch.fft.fft(Y, self.dr.Nk_fb, dim=-1, norm="ortho")
            Y = Y.reshape((num_samples, 1, self.dr.Nk_fb))
        elif self.args['xyz_str'].lower() == 'xz':
            Y = Y.reshape((num_samples, 1, self.dr.Nk_fb))

        # Reshape (num_samples x 2 x Nk_fb)
        X = torch.cat((torch.real(X), torch.imag(X)), axis=1)
        Y = torch.cat((torch.real(Y), torch.imag(Y)), axis=1)
        
        # Split data
        print("creating datasets")
        self.train_X = X[:split[0], :, :]
        self.train_Y = Y[:split[0], :, :]
        self.train_SNR = SNR[:split[0]]
        self.dataset_train = torch.utils.data.TensorDataset(self.train_X.to(self.args['device']), self.train_Y.to(self.args['device']))
        
        self.val_X = X[split[0]:(split[0]+split[1]), :, :]
        self.val_Y = Y[split[0]:(split[0]+split[1]), :, :]
        self.val_SNR = SNR[split[0]:(split[0]+split[1])]
        self.dataset_val = torch.utils.data.TensorDataset(self.val_X.to(self.args['device']), self.val_Y.to(self.args['device']))
        
        self.test_X = X[(split[0]+split[1]):(split[0]+split[1]+split[2]), :, :]
        self.test_Y = Y[(split[0]+split[1]):(split[0]+split[1]+split[2]), :, :]
        self.test_SNR = SNR[(split[0]+split[1]):(split[0]+split[1]+split[2])]
        self.dataset_test = torch.utils.data.TensorDataset(self.test_X, self.test_Y)
        
        self.num_train, self.num_val, self.num_test = split[0], split[1], split[2]
        
    def test_net(self, args, model, data):
        ind_rand = np.random.randint(0, data.num_test)
        
        print(f"SNR = {self.test_SNR[ind_rand]}")

        (X, Y) = data.dataset_test[ind_rand]
        x_in = X.to(args['device'])

        x_in = CPLX(x_in[0, :].reshape((1, 1, -1)), x_in[1, :].reshape((1, 1, -1)))
        Y_pred = model.model(x_in)
        y_pred = (Y_pred.r + 1j*Y_pred.i).cpu().detach().numpy().reshape(-1)

        x = X[0] + 1j*X[1]
        y = Y[0] + 1j*Y[1]
        
        if args['xyz_str'].lower() == 'xz':
            y = np.fft.fft(y, norm="ortho")
            y_pred = np.fft.fft(y_pred, norm="ortho")
        else:
            y = y.numpy()

        fig = plt.figure(figsize=[24, 16])
        plt.subplot(241)
        plt.plot(np.abs(x))
        plt.title("Input " + str(ind_rand))

        plt.subplot(242)
        plt.plot(np.abs(y_pred))
        plt.title("Output")

        plt.subplot(243)
        plt.plot(np.abs(y))
        plt.title("Ground Truth")

        plt.subplot(244)
        err = np.abs(y - y_pred)
        plt.plot(err)
        plt.title("Error")

        x = np.fft.ifft(x, norm="ortho")
        y = np.fft.ifft(y, norm="ortho")
        y_pred = np.fft.ifft(y_pred, norm="ortho")            
            
        plt.subplot(245)
        plt.plot(np.real(x))
        plt.title("Input")

        plt.subplot(246)
        plt.plot(np.real(y_pred))
        plt.title("Output")

        plt.subplot(247)
        plt.plot(np.real(y))
        plt.title("Ground Truth")

        plt.subplot(248)
        err = y - y_pred
        plt.plot(np.real(err))
        plt.title("Error")

        plt.show()