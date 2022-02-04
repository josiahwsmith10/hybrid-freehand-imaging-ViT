import torch
import torch.nn as nn
from model.common import complex_default_conv, CReLU, CTanh, ComplexResBlock

class ComplexModel(nn.Module):
    def __init__(self, args):
        super(ComplexModel, self).__init__()
        
        self.device = args['device']
        
        if args['model'].lower() == 'cvedsr':
            print('Making CV-EDSR model...')
            self.model = ComplexEDSR(args).to(self.device)
        elif args['model'].lower() == 'fftnet':
            print('Making FFT-Net model...')
            self.model = FFTNet(args).to(self.device)
        
        if args['precision'].lower() == 'half':
            self.model.half()
            
    def forward(self, x):        
        if self.training:
            return self.model(x)
        else:
            return self.model.forward
                
class ComplexEDSR(nn.Module):
    def __init__(self, args, conv=complex_default_conv):
        super(ComplexEDSR, self).__init__()
        
        if args['act'].lower() == 'crelu':
            self.act = CReLU(True)
        elif args['act'].lower() == 'ctanh':
            self.act = CTanh()        
        
        # define head module
        m_head = [conv(args['in_channels'], args['n_feats'], args['kernel_size'])]
        
        # define body module
        m_body = [
            ComplexResBlock(
                conv, args['n_feats'], args['kernel_size'], act=self.act, 
                res_scale=args['res_scale']
            ) for _ in range(args['n_res_blocks'])
        ]
        m_body.append(conv(args['n_feats'], args['n_feats'], args['kernel_size']))
        
        # define tail module (not upsampler if fft size is same for LR and HR)
        m_tail = [conv(args['n_feats'], args['out_channels'], args['kernel_size'])]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        
    def forward(self, x):
        # could subtract mean here
        x = self.head(x)
        
        res = self.body(x)
        res.add(x)
        
        x = self.tail(res)
        # could add the mean here
        return x
    
class FFTNet(nn.Module):
    def __init__(self, args, conv=complex_default_conv):
        super(FFTNet, self).__init__()
        
        self.args = args
        
        if args['act'].lower() == 'crelu':
            self.act = CReLU(True)
        elif args['act'].lower() == 'ctanh':
            self.act = CTanh()        
        
        # define head module
        m_head = [conv(args['in_channels'], args['n_feats'], args['kernel_size'])]
        
        # define body module
        m_body1 = [
            ComplexResBlock(
                conv, args['n_feats'], args['kernel_size'], act=self.act, 
                res_scale=args['res_scale']
            ) for _ in range(args['n_res_blocks']//4)
        ]
        m_body1.append(conv(args['n_feats'], args['n_feats'], args['kernel_size']))
        
        m_body2 = [
            ComplexResBlock(
                conv, args['n_feats'], args['kernel_size'], act=self.act, 
                res_scale=args['res_scale']
            ) for _ in range(args['n_res_blocks']//4)
        ]
        m_body2.append(conv(args['n_feats'], args['n_feats'], args['kernel_size']))
        
        m_body3 = [
            ComplexResBlock(
                conv, args['n_feats'], args['kernel_size'], act=self.act, 
                res_scale=args['res_scale']
            ) for _ in range(args['n_res_blocks']//4)
        ]
        m_body3.append(conv(args['n_feats'], args['n_feats'], args['kernel_size']))
        
        m_body4 = [
            ComplexResBlock(
                conv, args['n_feats'], args['kernel_size'], act=self.act, 
                res_scale=args['res_scale']
            ) for _ in range(args['n_res_blocks']//4)
        ]
        m_body4.append(conv(args['n_feats'], args['n_feats'], args['kernel_size']))
        
        # define tail module (not upsampler if fft size is same for LR and HR)
        m_tail = [conv(args['n_feats'], args['out_channels'], args['kernel_size'])]
        
        self.head = nn.Sequential(*m_head)
        self.body1 = nn.Sequential(*m_body1)
        self.body2 = nn.Sequential(*m_body2)
        self.body3 = nn.Sequential(*m_body3)
        self.body4 = nn.Sequential(*m_body4)
        self.tail = nn.Sequential(*m_tail)
        
        if args['xyz_str'].lower() == 'xz':
            m_body5 = [
                ComplexResBlock(
                    conv, args['n_feats'], args['kernel_size'], act=self.act, 
                    res_scale=args['res_scale']
                ) for _ in range(args['n_res_blocks']//4)
            ]
            m_body5.append(conv(args['n_feats'], args['n_feats'], args['kernel_size']))
            self.body5 = nn.Sequential(*m_body5)
        
    def forward(self, x):
        # could subtract mean here
        x = self.head(x)
        
        res = self.body1(x)
        res.ifft()
        
        res = self.body2(res)
        res.fft()
        
        res = self.body3(res)
        res.ifft()
        
        res = self.body4(res)
        res.fft()
        
        res.add(x)
        
        if self.args['xyz_str'].lower() == 'xz':
            res.ifft()
            res = self.body5(res)
        
        x = self.tail(res)
        # could add the mean here
        return x