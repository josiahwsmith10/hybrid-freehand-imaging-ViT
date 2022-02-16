import torch

class Saver():
    def __init__(self):
        pass
    
    def Save(self, args, model, loss, trainer, PATH):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': loss,
            'args': args
        }
        
        torch.save(checkpoint, PATH)
        print(f"Saved model to: {PATH}")
    
    def Load(self, data, ModelClass, LossClass, TrainerClass, PATH):
        class Args:
            def __init__(self):
                self.act = None
                
        checkpoint = torch.load(PATH)
        
        args = checkpoint['args']
        model = ModelClass(args)
        loss = checkpoint['loss']
        trainer = TrainerClass(args, data, model, loss)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        model.eval()
        print(f"Loaded model from: {PATH}")
        return args, model, loss, trainer
    
    def LoadModel(self, ModelClass, PATH):
        checkpoint = torch.load(PATH)
        
        args = checkpoint['args']
        model = ModelClass(args)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        print(f"Loaded model from: {PATH}")
        return args, model