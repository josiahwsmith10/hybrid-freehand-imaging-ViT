import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, c

class FMCW():
    def __init__(self, Nk=64, B=4e9, fs=2000e3, f0=77e9):
        self.Nk = Nk
        self.B = B
        self.fs = fs
        self.f0 = f0
        self.K = fs*B/(Nk-1)
        self.c = c
        
        self.f = f0 + np.arange(self.Nk)*self.K/self.fs
        self.k = 2*pi*self.f/self.c
        
        self.range_max_m = self.fs*self.c/(2*self.K)
        self.range_res_m = self.c/(2*self.B)
        self.lambda_m = self.c/(self.f0 + self.B/2)        

class Radar():
    def __init__(self, Nk=64, B=4e9, fs=2000e3, f0 = 77e9):
        self.fmcw = FMCW(Nk, B, fs, f0)
        
        
        