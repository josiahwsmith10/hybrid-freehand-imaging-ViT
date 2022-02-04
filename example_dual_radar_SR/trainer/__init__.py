import numpy as np
import torch
from model.common import CPLX
from trainer.common import timer, make_optimizer
from livelossplot import PlotLosses

class Trainer():
    def __init__(self, args, data, model, loss):
        print('Making the trainer...')
        self.args = args
        
        self.loader_train = torch.utils.data.DataLoader(data.dataset_train, 
                                                    batch_size=self.args['batch_size'], 
                                                    shuffle=True, 
                                                    pin_memory=False,
                                                    num_workers=0)
        
        self.loader_val = torch.utils.data.DataLoader(data.dataset_val, 
                                                    batch_size=self.args['batch_size'], 
                                                    shuffle=True, 
                                                    pin_memory=False,
                                                    num_workers=0)
        
        self.batches_per_epoch = len(self.loader_train)
        self.model = model.to(args['device'])
        self.loss = loss
        self.optimizer = make_optimizer(args, self.model)
        
        self.error_last = 1e8
        
        self.logs = {}
        self.liveloss = PlotLosses()
        
        
    def train(self):
        """
        Trains one epoch
        """
        
        # Training phase
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
        
        self.loss.start_log()
        self.model.train()
        
        running_loss = 0.0
        
        timer_data, timer_model = timer(), timer()
            
        for batch, (lr, hr) in enumerate(self.loader_train):
            # lr - low resolution (feature) (batch_size x 2 x N_HR)
            # hr - high resolution (label) (batch_size x 2 x N_HR)
            
            # Convert (batch_size x 2 x N_HR) tensor to CPLX
            lr = CPLX(lr[:, 0, :].reshape(self.args['batch_size'], 1, -1), 
                      lr[:, 1, :].reshape(self.args['batch_size'], 1, -1))
                      
            hr = CPLX(hr[:, 0, :].reshape(self.args['batch_size'], 1, -1), 
                      hr[:, 1, :].reshape(self.args['batch_size'], 1, -1))
            
            lr, hr = self.prepare(lr, hr) # accepts CPLX
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()
            sr = self.model(lr) # accepts CPLX
            loss = self.loss(sr, hr) # accepts CPLX
            loss.backward()
            self.optimizer.step()
            
            timer_model.hold()
            
            if self.args['print_every'] != 0 and (batch + 1) % self.args['print_every'] == 0:
                print('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args['batch_size'],
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
                
            running_loss += loss
                
            timer_data.tic()
            
        self.logs['log loss'] = running_loss.cpu().detach().numpy()
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        #print(f'epoch {self.optimizer.get_last_epoch()} finished')
        
        # Validation phase
        
        with torch.no_grad():
            # Somehow breaks everything
            # self.model.eval()
            running_loss = 0.0
                
            for batch, (lr, hr) in enumerate(self.loader_train):
                # lr - low resolution (feature) (batch_size x 2 x N_HR)
                # hr - high resolution (label) (batch_size x 2 x N_HR)
                
                # Convert (batch_size x 2 x N_HR) tensor to CPLX
                lr = CPLX(lr[:, 0, :].reshape(self.args['batch_size'], 1, -1), 
                        lr[:, 1, :].reshape(self.args['batch_size'], 1, -1))
                        
                hr = CPLX(hr[:, 0, :].reshape(self.args['batch_size'], 1, -1), 
                        hr[:, 1, :].reshape(self.args['batch_size'], 1, -1))
                
                lr, hr = self.prepare(lr, hr) # accepts CPLX
                
                sr = self.model(lr) # accepts CPLX
                loss = self.loss(sr, hr) # accepts CPLX
                running_loss += loss
                
            self.logs['val_log loss'] = running_loss.cpu().detach().numpy()
        
        self.liveloss.update(self.logs)
        self.liveloss.send()
        
    def prepare(self, *args):
        device = self.args['device']
        def _prepare(tensor):
            if self.args['precision'] == 'half': 
                tensor.r = tensor.r.half()
                tensor.i = tensor.i.half()
            
            r = tensor.r.to(device)
            i = tensor.i.to(device)
            return CPLX(r, i)
        
        return [_prepare(a) for a in args]
    
    def terminate(self):
        epoch = self.optimizer.get_last_epoch() + 1
        return epoch >= self.args['epochs']