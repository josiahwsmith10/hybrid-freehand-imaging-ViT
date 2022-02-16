import torch
from scipy.constants import pi
import numpy as np
import matplotlib.pyplot as plt
from model.common import CPLX

class Data:
    def __init__(self):
        self.X = None
        
    def create_dataset_train(self, num_train, N_LR, N_HR, max_freqs, xyz_str, is_phi=False):
        self.N_LR = N_LR
        self.N_HR = N_HR
        self.max_freqs = max_freqs
        self.is_phi = is_phi
        
        self.num_train = num_train
        
        self.create_XY(num_train)
        
        self.train_X = self.X
        self.train_Y = self.Y
        
        X = self.X.reshape((num_train, 1, self.N_HR))
        if xyz_str.lower() == 'xy':
            Y = self.Y.reshape((num_train, 1, self.N_HR))
        elif xyz_str.lower() == 'xz':
            Y = self.Z.reshape((num_train, 1, self.N_HR))
        
        X = np.concatenate((np.real(X), np.imag(X)), axis=1)
        Y = np.concatenate((np.real(Y), np.imag(Y)), axis=1)
        
        tensor_X = torch.Tensor(list(X))
        tensor_Y = torch.Tensor(list(Y))
        
        self.dataset_train = torch.utils.data.TensorDataset(tensor_X, tensor_Y)
        
    def create_dataset_test(self, num_test, N_LR, N_HR, max_freqs, xyz_str, is_phi=False):
        self.N_LR = N_LR
        self.N_HR = N_HR
        self.max_freqs = max_freqs
        self.is_phi = is_phi
        
        self.num_test = num_test
        
        self.create_XY(num_test)
        
        self.test_X = self.X
        self.test_Y = self.Y
        
        X = self.X.reshape((num_test, 1, self.N_HR))
        if xyz_str =='xy':
            Y = self.Y.reshape((num_test, 1, self.N_HR))
        elif xyz_str == 'xz':
            Y = self.Z.reshape((num_test, 1, self.N_HR))
        
        X = np.concatenate((np.real(X), np.imag(X)), axis=1)
        Y = np.concatenate((np.real(Y), np.imag(Y)), axis=1)
        
        tensor_X = torch.Tensor(list(X))
        tensor_Y = torch.Tensor(list(Y))
        
        self.dataset_test = torch.utils.data.TensorDataset(tensor_X, tensor_Y)
        
    def create_XY_old(self, num):
        X = np.zeros((num, self.N_HR), dtype=complex)
        Y = np.zeros((num, self.N_HR), dtype=complex)
        
        t_LR = np.arange(self.N_LR).reshape((1, self.N_LR))
        t_HR = np.arange(self.N_HR).reshape((1, self.N_HR))
        
        for ind_num in range(num):
            num_freqs = np.random.randint(1, self.max_freqs+1)
            freqs = -pi + 2*pi*np.random.rand(num_freqs, 1).reshape((num_freqs, 1))
            amps = 0.5 + 0.5*np.random.rand(num_freqs, 1).reshape((num_freqs, 1))
            
            temp = np.sum(amps * np.exp(1j*freqs*t_LR), axis=0)
            #temp = np.concatenate((temp, temp), axis=0)
            x_LR = torch.from_numpy(temp)
            x_HR = torch.from_numpy(np.sum(amps * np.exp(1j*freqs*t_HR), axis=0))
            
            y_LR = torch.fft.fft(x_LR, self.N_HR, dim=0, norm="ortho").cpu()
            y_HR = torch.fft.fft(x_HR, self.N_HR, dim=0, norm="ortho").cpu()
            
            X[ind_num] = y_LR.numpy()
            Y[ind_num] = y_HR.numpy()
            
        self.X = X
        self.Y = Y
        
    def create_XY(self, num):
        X = np.zeros((num, self.N_HR), dtype=complex)
        Y = np.zeros((num, self.N_HR), dtype=complex)
        Z = np.zeros((num, self.N_HR), dtype=complex)
        n_HR = np.arange(self.N_HR).reshape((1, self.N_HR))
        
        for ind_num in range(num):
            num_freqs = np.random.randint(1, self.max_freqs+1)
            freqs = -pi + 2*pi*np.random.rand(num_freqs, 1).reshape((num_freqs, 1))
            amps = 0.5 + 0.5*np.random.rand(num_freqs, 1).reshape((num_freqs, 1))
            if self.is_phi:
                phi = -pi + 2*pi*np.random.rand(num_freqs, 1).reshape((num_freqs, 1))
            else:
                phi = 0
            
            N_LR = np.random.randint(self.N_LR, self.N_HR*3//4)
            x_HR = torch.from_numpy(np.sum(amps * np.exp(1j*(freqs*n_HR + phi)), axis=0))
            x_LR = x_HR[0:N_LR]
            
            y_LR = torch.fft.fft(x_LR, self.N_HR, dim=0, norm="ortho").cpu()
            y_HR = torch.fft.fft(x_HR, self.N_HR, dim=0, norm="ortho").cpu()
            
            X[ind_num] = y_LR.numpy()
            Y[ind_num] = y_HR.numpy()
            Z[ind_num] = x_HR.cpu().numpy()
            
        self.X = X
        self.Y = Y
        self.Z = Z
        
    def test_net(self, args, model, data):
        ind_rand = np.random.randint(0, data.num_test)

        (X, Y) = data.dataset_test[ind_rand]
        x_in = X.to(args.device)

        x_in = CPLX(x_in[0, :].reshape((1, 1, -1)), x_in[1, :].reshape((1, 1, -1)))
        Y_pred = model.model(x_in)
        y_pred = (Y_pred.r + 1j*Y_pred.i).cpu().detach().numpy().reshape(-1)

        x = X[0] + 1j*X[1]
        y = Y[0] + 1j*Y[1]
        
        if args.xyz_str.lower() == 'xz':
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