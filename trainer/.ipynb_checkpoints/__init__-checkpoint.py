import numpy as np
import torch
from model.common import CPLX
from trainer.common import timer, make_optimizer

class Trainer():
    def __init__(self, args, my_data, my_model, my_loss):
        print('Making the trainer...')
        self.args = args
        self.scale = args.scale
        
        self.loader_train = torch.utils.data.DataLoader(my_data.dataset_train, 
                                                    batch_size=self.args.batch_size, 
                                                    shuffle=True, 
                                                    pin_memory=True,
                                                    num_workers=self.args.n_threads)
        #self.loader_test = torch.utils.data.DataLoader(my_data.dataset_test, 
        #                                            batch_size=self.args.batch_size, 
        #                                            shuffle=False, 
        #                                            pin_memory=True,
        #                                            num_workers=self.args.n_threads)
        self.batches_per_epoch = len(self.loader_train)
        self.model = my_model
        self.loss = my_loss
        self.optimizer = make_optimizer(args, self.model)
        
        self.error_last = 1e8
        
        
    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
        
        self.loss.start_log()
        self.model.train()
        
        timer_data, timer_model = timer(), timer()
        
        # set the scale to 0? Not sure what this is for, must be for a specific type of dataset
        #   only call the set_scale() function if the dataset has that attribute
        # if hasattr(self.loader_train.dataset, 'set_scale'): self.loader_train.dataset.set_scale(0)
            
        for batch, (lr, hr) in enumerate(self.loader_train):
            # lr - low resolution (feature) (batch_size x 2 x N_HR)
            # hr - high resolution (label) (batch_size x 2 x N_HR)
            
            # Convert (batch_size x 2 x N_HR) tensor to CPLX
            lr = CPLX(lr[:, 0, :].reshape(self.args.batch_size, 1, -1), 
                      lr[:, 1, :].reshape(self.args.batch_size, 1, -1))
                      
            hr = CPLX(hr[:, 0, :].reshape(self.args.batch_size, 1, -1), 
                      hr[:, 1, :].reshape(self.args.batch_size, 1, -1))
            
            lr, hr = self.prepare(lr, hr) # accepts CPLX
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()
            sr = self.model(lr) # accepts CPLX
            loss = self.loss(sr, hr) # accepts CPLX
            loss.backward()
            self.optimizer.step()
            
            timer_model.hold()
            
            if (batch + 1) % self.args.print_every == 0:
                print('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
                
            timer_data.tic()
            
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        print('epoch {} finished'.format(self.optimizer.get_last_epoch()))
        
    def test(self):
        print("WARNING: test() IS NOT IMPLEMENTED!!!!!!!!")
        
    def prepare(self, *args):
        device = self.args.device
        def _prepare(tensor):
            if self.args.precision == 'half': 
                tensor.r = tensor.r.half()
                tensor.i = tensor.i.half()
            
            r = tensor.r.to(device)
            i = tensor.i.to(device)
            return CPLX(r, i)
        
        return [_prepare(a) for a in args]
    
    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs