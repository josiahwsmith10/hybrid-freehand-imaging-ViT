import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, c
from model.common import CPLX
from scipy.interpolate import interpn
from scipy.spatial.distance import cdist
from radar.functions import plot_image_2d, plot_image_2d_dB

class FMCW():
    def __init__(self, Nk=64, B=4e9, fS=2000e3, f0=77e9):
        self.Nk = Nk
        self.B = B
        self.S = fS
        self.f0 = f0
        self.K = fS*B/(Nk-1)
        self.c = c
        
        self.f = f0 + np.arange(self.Nk)*self.K/self.fS
        self.k = 2*pi*self.f/self.c
        
        self.range_max_m = self.fS*self.c/(2*self.K)
        self.range_res_m = self.c/(2*self.B)
        self.lambda_m = self.c/(self.f0 + self.B/2)        

class Radar():
    def __init__(self, Nk=64, B=4e9, fS=2000e3, f0 = 77e9):
        self.fmcw = FMCW(Nk, B, fS, f0)
        
class TIRadar():
    def __init__(self, f0=77e9, K=124.996e12, Nk=64, fS=2000e3):
        self.f0 = f0
        self.K = K
        self.Nk = Nk
        self.fS = fS
        self.Compute()
        
    def __str__(self):
        return f"TI Radar with f0={self.f0*1e-9} GHz, K={self.K*1e-12} MHz/us, Nk={self.Nk}, fS={self.fS*1e-3} ksps"
        
    def Compute(self):
        self.f = self.f0 + self.K/self.fS*(torch.arange(self.Nk))
        self.k = 2*pi/c*self.f
        self.range_max_m = self.fS*c/(2*self.K)
        
class DualRadar():
    def __init__(self, f0=[60e9, 76.999456e9], K=124.996e12, Nk=64, Nk_fb=336, fS=2000e3):
        self.f0 = f0
        self.K = K
        self.Nk = Nk
        self.Nk_fb = Nk_fb
        self.fS = fS
        self.Compute()
        
    def __str__(self):
        return f"Dual Radar with \nradar1: \t{self.r0.__str__()}, \nradar2: \t{self.r1.__str__()} \nfull-band: \t{self.r_fb.__str__()}"
    
    def Compute(self):
        self.r0 = TIRadar(f0=self.f0[0])
        self.r1 = TIRadar(f0=self.f0[1])
        self.r_fb = TIRadar(f0=self.f0[0],Nk=self.Nk_fb)
        
        self.f_LR = torch.zeros(self.Nk_fb)
        self.f_LR[:self.Nk] = self.r0.f
        self.f_LR[-self.Nk:] = self.r1.f
        self.k_LR = 2*pi/c*self.f_LR
        
        self.f_HR = self.r_fb.f
        self.k_HR = self.r_fb.k

class DualRadarSAR:
    def __init__(self, dr):
        self.dr = dr

    def create_array_1d(self, Nx, dx):
        x_m = np.arange(-Nx / 2, Nx / 2) * dx
        y_m = 0
        z_m = 0
        self.z0_m = 0
        self.Nx = Nx
        self.dx = dx

        X, Y, Z = np.meshgrid(x_m, y_m, z_m, indexing="ij")
        self.array_xyz_m = np.concatenate((X, Y, Z), axis=2).reshape((-1, 3))

    def create_target(self, xyz_m, amp=1):
        self.target_xyz_m = xyz_m.reshape((-1, 3))
        self.amp = amp

    def create_target_rand_XZ(self, num, z=[0.2, 0.3]):
        x_m = -self.Nx * self.dx / 2 + 1e-3 + (self.Nx * self.dx - 2e-3) * np.random.rand(
            num, 1
        ).reshape((num, 1))
        y_m = np.zeros((num, 1)).reshape((num, 1))
        z_m = z[0] + (z[1] - z[0]) * np.random.rand(num, 1).reshape((num, 1))
        amp = 0.5 + 0.5 * np.random.rand(num, 1).reshape((1, num))
        amp = 1
        X, Y, Z = np.meshgrid(x_m, y_m, z_m, indexing="ij")
        self.target_xyz_m = np.concatenate((x_m, y_m, z_m), axis=1)
        self.amp = amp
        print(amp)

    def Compute_HR(self):
        R = cdist(self.array_xyz_m, self.target_xyz_m)

        self.sar_data = np.zeros(
            (self.array_xyz_m.shape[0], self.dr.Nk_fb), dtype=complex
        )

        for indK in range(0, self.dr.Nk_fb):
            self.sar_data[:, indK] = np.sum(
                self.amp * np.exp(1j * 2 * self.dr.k_HR[indK].numpy() * R), axis=1
            )
            
    def Compute_LR(self):
        self.Compute_HR()
        self.sar_data[:,self.dr.Nk:-self.dr.Nk] = np.zeros((self.Nx, self.dr.Nk_fb - 2*self.dr.Nk))
        

    def reconstruct_x_sar_xz_rma(
        self, nFFTx, nFFTz, sar_data=None, ti_radar=None, im_x=None, im_z=None, title="Reconstructed Image"
    ):
        if sar_data is None:
            sar_data = self.sar_data
            ti_radar = self.dr.r_fb

        zero_pad = np.zeros(((nFFTx - sar_data.shape[0]) // 2, ti_radar.Nk), dtype=complex)
        sar_data_padded = np.concatenate((zero_pad, sar_data), axis=0)

        k = ti_radar.k.reshape((1, -1)).numpy()
        L_x = nFFTx * self.dx
        dkX = 2 * pi / L_x
        kX = dkX * np.arange(-nFFTx / 2, nFFTx / 2).reshape((nFFTx, 1))

        kZU = np.linspace(0, 2 * np.max(k) - 2 * np.max(k) / nFFTz, nFFTz)
        dkZU = kZU[1] - kZU[0]

        kXU = np.tile(kX, (1, nFFTz))
        kU = 1 / 2 * np.sqrt(kX ** 2 + kZU ** 2)
        kZ = np.sqrt((4 * k ** 2 - kX ** 2) * (4 * k ** 2 > kX ** 2))

        focusing_filter = np.exp(-1j * kZ * self.z0_m) * (4 * k ** 2 > kX ** 2)

        sar_data_fft = np.fft.fftshift(
            np.fft.fft(np.conj(sar_data_padded), n=nFFTx, axis=0, norm="ortho"), axes=0
        )

        sar_image_fft = interpn(
            (kX.reshape(-1), k.reshape(-1)),
            sar_data_fft,
            (kXU, kU),
            method="nearest",
            bounds_error=False,
            fill_value=0,
        )

        self.sar_image = np.fft.ifft2(sar_image_fft, norm="ortho")

        x_m = np.arange(-(nFFTx - 1) / 2, (nFFTx - 1) / 2 + 1) * self.dx
        z_m = 2 * pi / (dkZU * nFFTz) * np.arange(nFFTz)

        if im_x is None:
            im = np.abs(self.sar_image)
            X, Z = np.meshgrid(x_m, z_m, indexing="ij")
        else:
            X, Z = np.meshgrid(im_x, im_z, indexing="ij")
            im = interpn(
                (x_m, z_m),
                np.abs(self.sar_image),
                (X, Z),
                method="linear",
                bounds_error=False,
                fill_value=0,
            )
            x_m, z_m = im_x, im_z
        
        plot_image_2d(im, x_m, z_m, xyz_str="xy", title=title)

    def test_model(self, args, model, nFFTx, nFFTz, im_x=None, im_z=None):
        self.Compute_LR()
        sar_data = self.sar_data
        
        s = torch.from_numpy(sar_data).reshape((self.Nx, 1, self.dr.Nk_fb))
        s = torch.fft.fft(s, n=self.dr.Nk_fb, dim=2, norm="ortho").cpu().numpy()

        sr = torch.Tensor(np.real(s)).to(args['device'])
        si = torch.Tensor(np.imag(s)).to(args['device'])
        c = CPLX(sr, si)
        c_pred = model.model(c)
        c_pred = (
            (c_pred.r + 1j * c_pred.i).cpu().detach().numpy().reshape((self.Nx, self.dr.Nk_fb))
        )
        if args['xyz_str'].lower() == "xy":
            c_pred = np.fft.fft(c_pred, axis=1, norm="ortho")

        xf = s[self.Nx // 4, 0, :].reshape(-1)
        xt = np.fft.ifft(xf, norm="ortho")
        yt = c_pred[self.Nx // 4, :].reshape(-1)
        yf = np.fft.fft(yt, norm="ortho")
        fig = plt.figure(figsize=[12, 8])
        plt.subplot(221)
        plt.plot(np.real(xt))
        plt.title("Input(t)")

        plt.subplot(222)
        plt.plot(np.real(yt))
        plt.title("Output(t)")

        plt.subplot(223)
        plt.plot(np.abs(xf))
        plt.title("Input(f)")

        plt.subplot(224)
        plt.plot(np.abs(yf))
        plt.title("Output(f)")
        
        self.reconstruct_x_sar_xz_rma(nFFTx, nFFTz, c_pred, self.dr.r_fb, im_x=im_x, im_z=im_z, title="SR - Hybrid Learning")