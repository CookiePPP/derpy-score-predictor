import math
import os

import torch
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, outdir: str, run_name: str, smoothing=0.99):
        self.outdir = outdir
        self.run_name = run_name
        self.run_dir = os.path.join(outdir, run_name)
        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "logdir"))
        self.smoothing = smoothing
        self.expavg_loss_dict = {}
    
    def state_dict(self):
        return {
            "expavg_loss_dict": self.expavg_loss_dict,
        }
    
    def load_state_dict(self, d):
        self.expavg_loss_dict = d["expavg_loss_dict"]
    
    def close(self):
        self.writer.close()
    
    def log(self, model, optimizer, loss_dict, dataset_name):
        # extract vars from objects
        step = model.iteration.item()
        epoch = model.epoch.item()
        
        # log the loss_dict to tensorboard
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                v = v.item()
            
            # calc exponential moving average of loss
            if math.isfinite(v):
                if k not in self.expavg_loss_dict:
                    self.expavg_loss_dict[k] = v
                else:
                    self.expavg_loss_dict[k] = self.smoothing * self.expavg_loss_dict[k] + (1 - self.smoothing) * v
            
            self.writer.add_scalar(f"{dataset_name}/{k}", self.expavg_loss_dict[k], step)
            self.writer.add_scalar(f"{dataset_name}_raw/{k}", v, step)
        
        # log epoch and lr
        if dataset_name == "train":
            self.writer.add_scalar("misc/epoch", epoch, step)
            
            lr = optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("misc/lr"   , lr   , step)