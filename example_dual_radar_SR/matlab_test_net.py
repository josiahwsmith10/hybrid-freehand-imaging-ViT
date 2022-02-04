import numpy as np
import torch
from model import ComplexModel
from model.common import CPLX
from saved import Saver

def test_net(x_in, net_path="./saved/fftnet5.tar"):
    """Applies dual-radar-SR network to data from MATLAB

    Args:
        input (MATLAB): N x Nk complex array from MATLAB, where N is the number of samples and NK is the number of ADC samples
    """
    
    x_in = np.array(x_in).astype(dtype=np.complex64)
    N, Nk = x_in.shape
    x_in = x_in.reshape((N, 1, Nk))
    x_in = torch.from_numpy(x_in)
    
    s = Saver()
    args, m = s.LoadModel(ComplexModel, net_path)
    
    x_in = x_in.to(args['device'])
    x_in = CPLX(torch.real(x_in), torch.imag(x_in))
    x_pred = m.model(x_in)
    x_out = (x_pred.r + 1j*x_pred.i).cpu().detach().numpy().reshape((N, Nk))
    
    return x_out

#x_in = np.random.randn(100, 336) 
#net_path = "./saved/fftnet5.tar"
x_out = test_net(x_in, net_path)