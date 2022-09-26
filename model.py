# Written by CookieGalaxy / CookiePPP
# This file will contain all the Model/Loss Function code.
import os

# This model will take
# - FloatTensor[B, n_tags] where each element is the probability of a tag being present.
# -  LongTensor[B,      1] where value is the year starting from 2010.
# -  LongTensor[B,      1] where value is the week in the year.
# -  LongTensor[B,      1] where value is the day of the week.
# -  LongTensor[B,      1] where value is the hour of the day.

# The model will return a dict with parameters of a Gaussian distribution.
# {
#   "wilson_score_mu"   : FloatTensor[B, 1],
#   "wilson_score_sigma": FloatTensor[B, 1],
#   "score_mu"          : FloatTensor[B, 1],
#   "score_sigma"       : FloatTensor[B, 1],
#   "upvotes_mu"        : FloatTensor[B, 1],
#   "upvotes_sigma"     : FloatTensor[B, 1],
#   "downvotes_mu"      : FloatTensor[B, 1],
#   "downvotes_sigma"   : FloatTensor[B, 1],
# }

import torch
import torch.nn as nn
from torch.distributions import Normal

from utils import to_device


class ResNetBlock(nn.Module):
    """
    ResNetBlock takes FloatTensor[B, D] and returns FloatTensor[B, D].
    """
    def __init__(self, hidden_size, n_layers, dropout, batch_norm, act_func, rezero: bool = False):
        super(ResNetBlock, self).__init__()
        self.rezero = rezero
        self.act_func = act_func
        
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList() if batch_norm else None
        for i in range(n_layers):
            self.lins.append(nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden_size))
        self.dropout = nn.Dropout(dropout)
        if rezero:
            self.rescale = nn.Parameter(torch.ones(1) * 1e-3)

    def forward(self, x):
        y = x
        for i, lin in enumerate(self.lins):
            y = lin(y)
            if self.bns is not None:
                y = self.bns[i](y)
            y = self.act_func(y)
            y = self.dropout(y)
        if self.rezero:
            y = y * self.rescale
        return x + y

class SmoothedBatchNorm(nn.Module):
    """
    BatchNorm that uses exponential moving average of the mean and variance.
    
    Input Shape: FloatTensor[B, D]
    """
    def __init__(self, hidden_size, momentum=0.1, affine=False, track_running_stats=True):
        super(SmoothedBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(hidden_size, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    
    def forward(self, x):
        """Normalize the input."""
        if self.training:
            with torch.no_grad():
                self.bn(x)
            self.bn.eval()
            x = self.bn(x)
            self.bn.train()
        else:
            self.bn.eval()
            x = self.bn(x)
        return x
    
    def reverse(self, x):
        """Reverse the normalization."""
        weight = getattr(self.bn, 'weight', 1.0)
        bias = getattr(self.bn, 'bias', 0.0)
        running_mean = getattr(self.bn, 'running_mean', 0.0)
        running_var = getattr(self.bn, 'running_var', 1.0)
        eps = getattr(self.bn, 'eps', 1e-5)
        x = (x - bias) / weight
        x = x * torch.sqrt(running_var + eps) + running_mean
        return x

class Model(nn.Module):
    def __init__(self, tags: list[str], n_blocks: int, n_layers: int, hidden_size: int, dropout: float, batch_norm: bool, act_func: nn.ReLU(), rezero: bool = False):
        super().__init__()
        self.tags = tags # list of tags, their position is their index in the embedding
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.act_func = act_func
        self.rezero = rezero
        
        # Register buffer for iteration, epoch and val loss
        self.register_buffer('iteration', torch.zeros(1, dtype=torch.long   ))
        self.register_buffer('epoch'    , torch.zeros(1, dtype=torch.long   ))
        self.register_buffer('val_loss' , torch.zeros(1, dtype=torch.float32))
        
        # Loss function targets are normalized while the model trains.
        self.ws_norm = SmoothedBatchNorm(1, momentum=0.01, affine=False) # wilson_score
        self.sc_norm = SmoothedBatchNorm(1, momentum=0.01, affine=False) # score
        self.up_norm = SmoothedBatchNorm(1, momentum=0.01, affine=False) # upvotes
        self.dn_norm = SmoothedBatchNorm(1, momentum=0.01, affine=False) # downvotes
        
        # Tags will use linear prenet
        self.pre = nn.Linear(len(self.tags), hidden_size)
        
        # Time will use embedding
        self.year_emb = nn.Embedding(16, hidden_size)
        self.week_emb = nn.Embedding(64, hidden_size)
        self. day_emb = nn.Embedding( 7, hidden_size)
        self.hour_emb = nn.Embedding(24, hidden_size)
        
        # ResNetBlocks
        self.resnet = nn.ModuleList()
        for i in range(n_layers):
            self.resnet.append(ResNetBlock(hidden_size, n_layers, dropout, batch_norm, act_func, rezero=rezero))
        
        # Output
        self.pos = nn.Linear(hidden_size, 8)
    
    def get_dtype(self):
        """Return the dtype of the model."""
        return next(self.parameters()).dtype
    
    def get_device(self):
        """Return the device of the model."""
        return next(self.parameters()).device
    
    def forward(self, input: dict):
        # Prenet
        x = self.pre(input["tags"]) # [B, n_tags] -> [B, hidden_size]
        B, hidden_size = x.shape
        
        # Time Embeddings
        x += self.year_emb(input["year"]).view(B, hidden_size)
        x += self.week_emb(input["week"]).view(B, hidden_size)
        x += self. day_emb(input[ "day"]).view(B, hidden_size)
        x += self.hour_emb(input["hour"]).view(B, hidden_size)
        
        # ResNetBlocks
        for i in range(self.n_blocks):
            x = self.resnet[i](x)
        
        # Postnet
        x = self.pos(x)
        
        # Extract the parameters Mean and Variance of the Gaussian distribution.
        mu, sigma = x[:, :4], x[:, 4:]
        sigma = torch.exp(sigma)
        
        # Return the parameters of the Gaussian distributions
        return {
            "wilson_score_mu"   : mu[:, 0:1],
            "wilson_score_sigma": sigma[:, 0:1],
            "score_mu"          : mu[:, 1:2],
            "score_sigma"       : sigma[:, 1:2],
            "upvotes_mu"        : mu[:, 2:3],
            "upvotes_sigma"     : sigma[:, 2:3],
            "downvotes_mu"      : mu[:, 3:4],
            "downvotes_sigma"   : sigma[:, 3:4],
        }
    
    def step(self, batch):
        # Transfer to device
        batch = to_device(batch, self.get_device())
        
        # Forward pass
        output = self(batch['input'])
        
        # Normalize the targets
        target = {
            "wilson_score": self.ws_norm(batch['target']['wilson_score']),
            "score"       : self.sc_norm(batch['target']['score'       ]),
            "upvotes"     : self.up_norm(batch['target']['upvotes'     ]),
            "downvotes"   : self.dn_norm(batch['target']['downvotes'   ]),
        }
        
        # Compute the loss
        loss_dict = self.loss(output, target)
        return loss_dict
    
    def loss(self, output: dict, target: dict):
        # Compute the loss for each term using nn.GaussianNLLLoss()
        loss_dict = {}
        for key in output:
            if key.endswith("sigma"): continue
            key = key[:-3] # remove "_mu" -> "wilson_score"
            var = output[key+"_sigma"]**2
            loss_dict[key] = nn.GaussianNLLLoss(reduction="mean")(
                output[key+"_mu"], # predicted mean
                target[key      ], # target mean
                var,               # predicted variance
            )
        loss_dict["total"] = sum(loss_dict.values())
        return loss_dict
    
    def infer_score(self, input: dict):
        """Take input dict and return score."""
        # Forward pass
        output = self(input)
        
        # Reverse the normalization
        score = self.target_norm.reverse(output["score_mu"])
        
        # Return the score
        return score
    
    def infer_distribution(self, input: dict):
        """Take input dict and return parameters for all normalized distributions."""
        return self(input)
    
    def infer_percentile(self, input: dict, scores: dict):
        """
        Input tags+datetime and scores and the model will return the predicted percentile of the score.
        
        Args:
            input: dict
            scores: dict
                {'wilson_score': float, 'score': int, 'upvotes': int, 'downvotes': int}
        
        Return:
            scores_percentiles: dict
                {'wilson_score': float, 'score': float, 'upvotes': float, 'downvotes': float}
        """
        train = self.training
        self.eval()
        
        # Get the distribution parameters
        output = self(input)
        
        # Convert img_data to tensor
        dtype = self.get_dtype()
        wilson_score = torch.tensor(scores["wilson_score"], dtype=dtype).view(-1, 1) # [B]
        score        = torch.tensor(scores["score"       ], dtype=dtype).view(-1, 1) # [B]
        upvotes      = torch.tensor(scores["upvotes"     ], dtype=dtype).view(-1, 1) # [B]
        downvotes    = torch.tensor(scores["downvotes"   ], dtype=dtype).view(-1, 1) # [B]
        
        # Normalize the scores
        wilson_score = self.ws_norm(wilson_score)
        score        = self.sc_norm(score)
        upvotes      = self.up_norm(upvotes)
        downvotes    = self.dn_norm(downvotes)
        
        # Compute the percentiles using pytorch Normal distribution
        scores_percentiles = {}
        scores_percentiles["wilson_score"] = Normal(output["wilson_score_mu"], output["wilson_score_sigma"]).cdf(wilson_score)
        scores_percentiles["score"       ] = Normal(output["score_mu"       ], output["score_sigma"       ]).cdf(score       )
        scores_percentiles["upvotes"     ] = Normal(output["upvotes_mu"     ], output["upvotes_sigma"     ]).cdf(upvotes     )
        scores_percentiles["downvotes"   ] = Normal(output["downvotes_mu"   ], output["downvotes_sigma"   ]).cdf(downvotes   )
        
        # Convert to float
        scores_percentiles = {key: val.item() for key, val in scores_percentiles.items()}
        
        # Return the percentiles
        self.train(train)
        return scores_percentiles
    
    def get_kwargs(self):
        return {
            "tags": self.tags,
            "n_blocks": self.n_blocks,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "act_func": self.act_func,
            "rezero": self.rezero,
        }
    
    def save(self, path: str):
        """Save the model."""
        d = {}
        d['state_dict'] = self.state_dict()
        d['tags'] = self.tags
        d['kwargs'] = self.get_kwargs()
        
        # save to tmp file first, then move to avoid partial file
        tmp_path = path + '.tmp'
        torch.save(d, tmp_path)
        os.rename(tmp_path, path)
    
    def load(self, path: str):
        """Load the model."""
        d = torch.load(path, map_location=self.device)
        self.load_state_dict(d['state_dict'])
        self.tags = d['tags']
    
    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = "cpu"):
        """Load the model from a checkpoint."""
        d = torch.load(path, map_location=device)
        model = cls(**d['kwargs']).to(device)
        model.load_state_dict(d['state_dict'])
        model.tags = d['tags']
        return model