import torch


# Class written by CookieGalaxy / CookiePPP
# This class will calculate moving percentiles for large lists without taking forever to call.
# Taken from https://github.com/CookiePPP/TSHP/blob/2858c0ae826462993ab18155a0140e7a11257c60/TSHP/modules/train_optim.py#L200-L270
class MovingPercentile:
    def __init__(self, percentile: float, max_len=None):
        assert 0.0 <= percentile <= 1.0, 'percentile must be between 0. and 1.'
        self.percentile = percentile
        self.max_len = max_len
        
        self.a_max = None
        self.a_list = []
        self.b_min = None
        self.b_list = []
        
        self.all_list = []
    
    def len(self):
        return len(self.a_list) + len(self.b_list)
    
    def state_dict(self):
        return {
            'a_max': self.a_max,
            'a_list': self.a_list,
            'b_min': self.b_min,
            'b_list': self.b_list,
            'all_list': self.all_list,
        }
    
    def load_state_dict(self, d):
        self.a_max = d['a_max']
        self.a_list = d['a_list']
        self.b_min = d['b_min']
        self.b_list = d['b_list']
        self.all_list = d['all_list']
    
    def __call__(self, new_value):
        if self.max_len is not None:
            self.all_list.append(new_value)
        a_list_max = max(self.a_list) if self.a_list else -float('inf')
        if new_value <= a_list_max:
            self.a_list.append(new_value)
        else:
            self.b_list.append(new_value)
        
        if self.max_len is not None and len(
                self.all_list) > self.max_len:  # if max_len and all_list is longer than max_len
            oldest_val = self.all_list.pop(0)  # remove oldest entry from all_list and a_list/b_list
            try:
                a_list_old_index = self.a_list.index(oldest_val)
            except ValueError as ex:
                a_list_old_index = None
            if a_list_old_index is not None:
                self.a_list.pop(a_list_old_index)
            else:
                self.b_list.pop(self.b_list.index(oldest_val))
        
        a_len = len(self.a_list)
        total_len = a_len + len(self.b_list)
        a_max = None
        b_min = None
        if a_len / total_len > self.percentile:  # if a_list (under percentile) is too large
            # move max element from a_list to b_list
            a_max = a_max if a_max is not None else max(self.a_list)
            self.b_list.append(self.a_list.pop(self.a_list.index(a_max)))
            b_min = a_max
            a_max = None
        elif a_len / total_len < self.percentile:  # elif b_list (over percentile) is too larger
            # move min element from b_list to a_list
            b_min = b_min if b_min is not None else min(self.b_list)
            self.a_list.append(self.b_list.pop(self.b_list.index(b_min)))
            a_max = b_min
            b_min = None
        
        if len(self.b_list) > len(self.a_list):
            quantile = b_min if b_min is not None else min(self.b_list)
        else:
            quantile = a_max if a_max is not None else max(self.a_list)
        return quantile

class AutoClip:
    def __init__(self, percentile: float, max_len=None):
        self.moving_percentile = MovingPercentile(percentile, max_len)
    
    def state_dict(self):
        return self.moving_percentile.state_dict()
    
    def load_state_dict(self, d):
        self.moving_percentile.load_state_dict(d)
    
    def __call__(self, model):
        # get current grad norm value
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
        # calculate 10% percentile of grad norm values
        grad_clip_val = self.moving_percentile(grad_norm)
        if self.moving_percentile.len() > 10:
            # clip grad norm to 10% percentile value
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (list, tuple)):
        return [to_device(xi, device) for xi in x]
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        return x

import collections
def get_duplicates(lst):
    return [item for item, count in collections.Counter(lst).items() if count > 1]