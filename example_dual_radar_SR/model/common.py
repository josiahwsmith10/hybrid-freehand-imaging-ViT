import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CPLX:
    def __init__(self, r, i):
        self.r = r
        self.i = i
        
    def mag(self):
        return torch.sqrt(self.r.pow(2) + self.i.pow(2))
    
    def size(self):
        return self.r.size(), self.i.size()
    
    def to_complex(self):
        out = self.r + 1j*self.i
        return out
    
    def add(self, other_CPLX):
        self.r += other_CPLX.r 
        self.i += other_CPLX.i
    
    def size(self):
        return (self.r.shape, self.i.shape)
    
    def mul(self, other):
        self.r *= other
        self.i *= other
        return self
    
    def reshape(self, shape):
        r = self.r.reshape(shape)
        i = self.i.reshape(shape)
        return CPLX(r, i)
    
    def fft(self, n=None, dim=-1, norm="ortho"):
        c = self.r + 1j*self.i
        out = torch.fft.fft(c, n, dim, norm)
        self.r = torch.real(out)
        self.i = torch.imag(out)
        
    def ifft(self, n=None, dim=-1, norm="ortho"):
        c = self.r + 1j*self.i
        out = torch.fft.ifft(c, n, dim, norm)
        self.r = torch.real(out)
        self.i = torch.imag(out)

class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                dilation=1, groups=1, bias=True):
        super(ComplexConv1d, self).__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    
    def forward(self, x):
        # ComplexConv1d forward
        r = self.conv_r(x.r) - self.conv_i(x.i)
        i = self.conv_r(x.i) + self.conv_i(x.r)
        
        out = CPLX(r, i)
        return out

class CReLU(nn.Module):
    def __init__(self, inplace=False):
        super(CReLU, self).__init__()
        self.relu_r = nn.ReLU(inplace)
        self.relu_i = nn.ReLU(inplace)
        
    def forward(self, x):
        return CPLX(self.relu_r(x.r), self.relu_i(x.i))
    
class CTanh(nn.Module):
    def __init__(self):
        super(CTanh, self).__init__()
        self.tanh_r = nn.Tanh()
        self.tanh_i = nn.Tanh()
        
    def forward(self, x):
        return CPLX(self.tanh_r(x.r), self.tanh_i(x.i))

def complex_default_conv(in_channels, out_channels, kernel_size, bias=True):
    return ComplexConv1d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
    
class ComplexResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=CTanh(), res_scale=1):

        super(ComplexResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        # Complex residual block forward
        res = self.body(x).mul(self.res_scale)
        res.add(x)

        return res
    
class FFTResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=CTanh(), res_scale=1):

        super(ComplexResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        # Complex residual block forward
        res = self.body(x).mul(self.res_scale)
        res.add(x)

        return res