import torch
from scipy.constants import pi
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms

_parula_data = [[0.2081, 0.1663, 0.5292], 
                [0.2116238095, 0.1897809524, 0.5776761905], 
                [0.212252381, 0.2137714286, 0.6269714286], 
                [0.2081, 0.2386, 0.6770857143], 
                [0.1959047619, 0.2644571429, 0.7279], 
                [0.1707285714, 0.2919380952, 0.779247619], 
                [0.1252714286, 0.3242428571, 0.8302714286], 
                [0.0591333333, 0.3598333333, 0.8683333333], 
                [0.0116952381, 0.3875095238, 0.8819571429], 
                [0.0059571429, 0.4086142857, 0.8828428571], 
                [0.0165142857, 0.4266, 0.8786333333], 
                [0.032852381, 0.4430428571, 0.8719571429], 
                [0.0498142857, 0.4585714286, 0.8640571429], 
                [0.0629333333, 0.4736904762, 0.8554380952], 
                [0.0722666667, 0.4886666667, 0.8467], 
                [0.0779428571, 0.5039857143, 0.8383714286], 
                [0.079347619, 0.5200238095, 0.8311809524], 
                [0.0749428571, 0.5375428571, 0.8262714286], 
                [0.0640571429, 0.5569857143, 0.8239571429], 
                [0.0487714286, 0.5772238095, 0.8228285714], 
                [0.0343428571, 0.5965809524, 0.819852381], 
                [0.0265, 0.6137, 0.8135], 
                [0.0238904762, 0.6286619048, 0.8037619048], 
                [0.0230904762, 0.6417857143, 0.7912666667], 
                [0.0227714286, 0.6534857143, 0.7767571429], 
                [0.0266619048, 0.6641952381, 0.7607190476], 
                [0.0383714286, 0.6742714286, 0.743552381], 
                [0.0589714286, 0.6837571429, 0.7253857143], 
                [0.0843, 0.6928333333, 0.7061666667], 
                [0.1132952381, 0.7015, 0.6858571429], 
                [0.1452714286, 0.7097571429, 0.6646285714], 
                [0.1801333333, 0.7176571429, 0.6424333333], 
                [0.2178285714, 0.7250428571, 0.6192619048], 
                [0.2586428571, 0.7317142857, 0.5954285714], 
                [0.3021714286, 0.7376047619, 0.5711857143], 
                [0.3481666667, 0.7424333333, 0.5472666667], 
                [0.3952571429, 0.7459, 0.5244428571], 
                [0.4420095238, 0.7480809524, 0.5033142857], 
                [0.4871238095, 0.7490619048, 0.4839761905], 
                [0.5300285714, 0.7491142857, 0.4661142857], 
                [0.5708571429, 0.7485190476, 0.4493904762],
                [0.609852381, 0.7473142857, 0.4336857143], 
                [0.6473, 0.7456, 0.4188], 
                [0.6834190476, 0.7434761905, 0.4044333333], 
                [0.7184095238, 0.7411333333, 0.3904761905], 
                [0.7524857143, 0.7384, 0.3768142857], 
                [0.7858428571, 0.7355666667, 0.3632714286], 
                [0.8185047619, 0.7327333333, 0.3497904762], 
                [0.8506571429, 0.7299, 0.3360285714], 
                [0.8824333333, 0.7274333333, 0.3217], 
                [0.9139333333, 0.7257857143, 0.3062761905], 
                [0.9449571429, 0.7261142857, 0.2886428571], 
                [0.9738952381, 0.7313952381, 0.266647619], 
                [0.9937714286, 0.7454571429, 0.240347619], 
                [0.9990428571, 0.7653142857, 0.2164142857], 
                [0.9955333333, 0.7860571429, 0.196652381], 
                [0.988, 0.8066, 0.1793666667], 
                [0.9788571429, 0.8271428571, 0.1633142857], 
                [0.9697, 0.8481380952, 0.147452381], 
                [0.9625857143, 0.8705142857, 0.1309], 
                [0.9588714286, 0.8949, 0.1132428571], 
                [0.9598238095, 0.9218333333, 0.0948380952], 
                [0.9661, 0.9514428571, 0.0755333333], 
                [0.9763, 0.9831, 0.0538]]

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
        
        ds = torch.utils.data.TensorDataset(X.to(dtype=torch.float), Y.to(dtype=torch.float))        
        return ds
    
    def create_train_dataset(self, split=(3296, 800), filenames=["hffh_ViT_solid_w_points_2048_training", "hffh_ViT_solid_w_points_2048_training_v2"]):
        assert len(split) == 2, 'split must have a length of 2'
        
        self.num_train, self.num_val = split
        
        lr, hr = self.load_train_data(filenames=filenames)
        
        assert sum(split) <= lr.shape[2] and sum(split) <= hr.shape[2], f'split cannot exceed {min(lr.shape[2], hr.shape[2])}'
        
        self.dataset_train = self.create_dataset(lr[:, :, :split[0]], hr[:, :, :split[0]], self.args['device'])
        self.dataset_val = self.create_dataset(lr[:, :, split[0]:(split[0]+split[1])], hr[:, :, split[0]:(split[0]+split[1])], self.args['device'])
        
    
    def create_test_dataset(self, filenames=["hffh_ViT_solid_w_points_1024_testing"]):
        ffh, rma, bpa, hr = self.load_test_data(filenames=filenames)
        
        self.test = {}
        self.test['ffh'] = ffh.transpose((2, 0, 1))
        self.test['rma'] = rma.transpose((2, 0, 1))
        self.test['bpa'] = bpa.transpose((2, 0, 1))
        self.test['hr'] = hr.transpose((2, 0, 1))
        
    def test_net_quant(self, model, args):
        def normalize(x):
            return (x-x.min())/(x.max()-x.min())

        def PSNR(y_true, y_pred, MAXp=1.):
            if not isinstance(MAXp, torch.Tensor):
                MAXp = torch.tensor(MAXp)
            if y_true.max() != 1.:
                MAXp = torch.tensor(y_true.max())
            batch_psnr = 20*torch.log10(MAXp)-10*torch.log10(torch.mean(torch.square(y_true-y_pred), dim=(-1,-2)))
            return batch_psnr

        def RMSE(y_true, y_pred, eps=1e-6):
            rmse = torch.sqrt(torch.mean(torch.square(y_true-y_pred) + eps, dim=(-1,-2)))
            return rmse
        
        def test_one(x, model, args):
            return model(x.reshape((1, 1, 200, 200)).to(device=args['device'], dtype=torch.float))[0][0]
        
        def preprocess(x, args, shape=(256, 256)):
            p = transforms.Resize(shape)
            return p(torch.tensor(x).reshape((1, x.shape[0], x.shape[1]))).to(device=args['device'], dtype=torch.float)[0]

        # Calculate PSNR and RMSE

        psnrs_ffh = []
        rmses_ffh = []
        psnrs_ours = []
        rmses_ours = []
        psnrs_bpa = []
        rmses_bpa = []
        psnrs_rma = []
        rmses_rma = []

        for i in range(self.test['ffh'].shape[0]):            
            xx_ffh = preprocess(self.test['ffh'][i], args, shape=(200, 200))
            xx_bpa = preprocess(self.test['bpa'][i], args)
            xx_rma = preprocess(self.test['rma'][i], args)
            yy_ffh = preprocess(self.test['hr'][i], args)
            yy_bpa = preprocess(self.test['hr'][i], args)
            yy_rma = preprocess(self.test['hr'][i], args)
            
            # Generate
            generated = preprocess(test_one(xx_ffh, model, args), args)
            xx_ffh = preprocess(xx_ffh, args)
            
            # Normalize ffh data
            z = torch.concat([xx_ffh, yy_ffh, generated])
            z = normalize(z)
            xx_ffh = z[:z.shape[0]//3]
            yy_ffh = z[z.shape[0]//3 : 2*(z.shape[0]//3)]
            generated = z[2*(z.shape[0]//3):]

            # Normalize bpa data
            xx_bpa = normalize(xx_bpa)
            
            # Normalize rma data
            xx_rma = normalize(xx_rma)
            psnrs_ffh.append(PSNR(y_true=yy_ffh, y_pred=xx_ffh).item())
            rmses_ffh.append(RMSE(y_true=yy_ffh, y_pred=xx_ffh).item())
            psnrs_ours.append(PSNR(y_true=yy_ffh, y_pred=generated).item())
            rmses_ours.append(RMSE(y_true=yy_ffh, y_pred=generated).item())

            psnrs_bpa.append(PSNR(y_true=yy_bpa, y_pred=xx_bpa).item())
            rmses_bpa.append(RMSE(y_true=yy_bpa, y_pred=xx_bpa).item())
            psnrs_rma.append(PSNR(y_true=yy_rma, y_pred=xx_rma).item())
            rmses_rma.append(RMSE(y_true=yy_rma, y_pred=xx_rma).item())
            
            print(f"{i} / 1024", end="\r")

        print(f"FFH  PSNR: {np.mean(psnrs_ffh):.3f}")
        print(f"FFH  RMSE: {np.mean(rmses_ffh):.3f}")
        print(f"OURS PSNR: {np.mean(psnrs_ours):.3f}")
        print(f"OURS RMSE: {np.mean(rmses_ours):.3f}")
        print(f"BPA  PSNR: {np.mean(psnrs_bpa):.3f}")
        print(f"BPA  RMSE: {np.mean(rmses_bpa):.3f}")
        print(f"RMA  PSNR: {np.mean(psnrs_rma):.3f}")
        print(f"RMA  RMSE: {np.mean(rmses_rma):.3f}")
    
    def Save(self, PATH):
        data_save = {
            'args': self.args,
            'dataset_train': self.dataset_train,
            'dataset_val': self.dataset_val,
            'test': self.test
        }
        torch.save(data_save, PATH)
        
    def Load(self, PATH):
        data_save = torch.load(PATH)
        
        self.dataset_train = data_save['dataset_train']
        self.dataset_val = data_save['dataset_val']
        self.test = data_save['test']
        
    def plot_single(self, im, title='Plot'):
        parula_map = LinearSegmentedColormap.from_list('parula', _parula_data)
        im = im / im.max()
        np_im_grid = parula_map(im)

        plt.figure(figsize=(4, 4), dpi=100)
        #print(np_im_grid.shape)
        c = plt.imshow(np_im_grid, interpolation="nearest", origin='lower', vmin=0, vmax=im.max())
        ax = plt.gca()
        ax.axis('off')
        #plt.colorbar(c)
        ax.set_xticks(np.arange(0,np_im_grid.shape[0]))
        ax.set_yticks(np.arange(0,np_im_grid.shape[1]))
        ax.grid(color='black', linestyle='-', linewidth=2)
        plt.title(title, fontweight="bold")
        plt.show()
        
    def preview(self):
        ind = np.random.randint(len(self.dataset_val))
        
        lr = self.dataset_val.tensors[0][ind][0]
        hr = self.dataset_val.tensors[1][ind][0]
        
        self.plot_single(lr, title=f"LR #{ind}")
        self.plot_single(hr, title=f"HR #{ind}")

    def test_net(self, model, args, ind=None):
        if ind is None:
            ind = np.random.randint(len(self.dataset_val))
        
        lr = self.dataset_val.tensors[0][ind][0]
        hr = self.dataset_val.tensors[1][ind][0]
        sr = model(self.dataset_val.tensors[0][ind].reshape((1, 1, 200, 200)).to(args['device'])).cpu().detach().numpy()[0][0]
        
        self.plot_single(lr, title=f"LR #{ind}")
        self.plot_single(hr, title=f"HR #{ind}")
        self.plot_single(sr, title=f"SR #{ind}")