# Written by CookieGalaxy / CookiePPP
# This file will contain all the training and evaluation code for this model.
import os
import random

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
import torch.optim as optim
import torch.utils.data as data
import argparse
import pickle as pkl

from data import Dataset, collate_func
from model import Model
from logger import Logger
from utils import AutoClip, to_device
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training.")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help="Weight decay for optimizer.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout used between layers.")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="Hidden size of layers in the model.")
    parser.add_argument("--n_blocks", type=int, default=2,
                        help="Number of resudual blocks in the model.")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of layers in the model.")
    parser.add_argument("--n_tags", type=int, default=1000,
                        help="Number of tags for training.")
    parser.add_argument("--batch_norm", action="store_true",
                        help="Use batch norm for training.")
    parser.add_argument("--rezero", action="store_true",
                        help="Use rezero for training.")
    parser.add_argument("--metadata", type=str, default="data/metadata.p",
                        help="Path to metadata for training.")
    parser.add_argument("--tagsdata", type=str, default="data/tagsdata.p",
                        help="Path to tagsdata for training.")
    parser.add_argument("--outdir", type=str, default="runs",
                        help="Path to output directory for training.")
    parser.add_argument("--run_name", type=str, default="001",
                        help="Name of the training run.")
    return parser.parse_args()


def find_lr(model: Model, optimizer, train_loader, val_loader, logger: Logger):
    # Find the optimal learning rate for the model by exponentially increasing the learning rate.
    # Save the model and optimizer state to RAM every time the training loss decreases.
    initial_lr = 1e-7
    n_iters_to_10x = 1000
    
    best_train_loss = float("inf")
    best_loss_iter = 0
    
    # run model till loss explodes
    model.train()
    optimizer.param_groups[0]["lr"] = initial_lr
    loss_is_finite = True
    while loss_is_finite:
        for batch in tqdm(train_loader, desc="Finding LR"):
            # calc LR for this step
            lr = initial_lr * (10 ** (model.iteration.item() / n_iters_to_10x))
            optimizer.param_groups[0]["lr"] = lr
            
            # update model
            optimizer.zero_grad()
            loss_dict = model.step(batch)
            loss_dict['total'].backward()
            autoclip(model) # auto-clip the models gradients
            optimizer.step()
            model.iteration += 1
            
            # log results
            logger.log(model, optimizer, loss_dict, dataset_name="train")
            if loss_dict['total'].item() < best_train_loss:
                best_train_loss = loss_dict['total'].item()
                best_loss_iter = model.iteration.item()
                best_model_state = to_device(model.state_dict(), 'cpu')
                best_optimizer_state = to_device(optimizer.state_dict(), 'cpu')
                best_autoclip_state = autoclip.state_dict()
                best_logger_state = logger.state_dict()
            
            # exit if loss explodes
            if not torch.isfinite(loss_dict['total']):
                loss_is_finite = False
                break
        model.epoch += 1
    
    # calculate LR of best model
    best_lr = initial_lr * (10 ** (best_loss_iter / n_iters_to_10x))
    optimizer.param_groups[0]["lr"] = best_lr
    
    # load best model
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)
    autoclip.load_state_dict(best_autoclip_state)
    logger.load_state_dict(best_logger_state)
    
    return best_lr

def train(model: Model, optimizer, autoclip, train_loader, val_loader, logger: Logger):
    # track the best validation model
    best_val_loss = float("inf")
    best_val_loss_epoch = 0
    best_model_state = None
    best_optimizer_state = None
    best_autoclip_state = None
    best_logger_state = None
    
    # scheduler args
    patience = 3 # if validation loss does not decrease for this many epochs, reduce LR
    factor = 0.5 # reduce LR by this factor
    
    # train model
    model.train()
    passes = 0
    while optimizer.param_groups[0]["lr"] > 1e-5:
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train|Dataset Pass {passes}"):
            # update model
            optimizer.zero_grad()
            loss_dict = model.step(batch)
            loss_dict['total'].backward()
            autoclip(model) # auto-clip the models gradients
            if model.iteration.item() > 3000 and loss_dict['total'].item() > 1e3:
                # don't update model if loss is exploding
                [p.grad.zero_() for p in model.parameters()]
                loss_dict['total'].data.zero_()
            optimizer.step()
            model.iteration += 1
            
            # log results
            logger.log(model, optimizer, loss_dict, dataset_name="train")
            train_loss += loss_dict['total'].item()
        model.epoch += 1
        train_loss /= len(train_loader)
        
        # validate model
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in tqdm(val_loader, desc=f"Val|Dataset Pass {passes}"):
                loss_dict = model.step(batch)
                logger.log(model, optimizer, loss_dict, dataset_name="val_steps")
                val_loss += loss_dict['total'].item()
            val_loss /= len(val_loader)
            
            logger.log(model, optimizer, {'total': val_loss}, dataset_name="val")
            tqdm.write(f"ITER: {model.iteration.item()} EPOCH: {model.epoch.item()} TRAIN LOSS: {train_loss:.4f} VAL LOSS: {val_loss:.4f}")
            
            # if validation loss decreases, save model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = model.epoch.item()
                best_model_state = to_device(model.state_dict(), 'cpu')
                best_optimizer_state = to_device(optimizer.state_dict(), 'cpu')
                best_autoclip_state = autoclip.state_dict()
                best_logger_state = logger.state_dict()
            # if validation loss does not decrease for patience epochs, load best model and reduce LR
            if model.epoch.item() - best_val_loss_epoch > patience:
                # load best model
                model.load_state_dict(best_model_state)
                optimizer.load_state_dict(best_optimizer_state)
                autoclip.load_state_dict(best_autoclip_state)
                logger.load_state_dict(best_logger_state)
                
                # reduce LR
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= factor
                    tqdm.write(f"LR reduced to {param_group['lr']:.1e}")
                
                # update best_optimizer_state with new LR
                best_optimizer_state = to_device(optimizer.state_dict(), 'cpu')
            
        model.train()
        passes += 1
    
    # load best model
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)
    autoclip.load_state_dict(best_autoclip_state)
    logger.load_state_dict(best_logger_state)
    
    # save to disk
    model.save(os.path.join(logger.run_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(logger.run_dir, "optimizer.pt"))
    torch.save(autoclip.state_dict(), os.path.join(logger.run_dir, "autoclip.pt"))
    # write text file with best validation loss
    with open(os.path.join(logger.run_dir, "best_val_loss.txt"), "w") as f:
        f.write(f"{best_val_loss:.5f}")

if __name__ == "__main__":
    args = get_args()
    print(args)
    
    # Init seeds
    print("Initializing seeds...")
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    
    # Load data
    print("Loading data...")
    metadata = pkl.load(open(args.metadata, "rb"))
    tagsdata = pkl.load(open(args.tagsdata, "rb"))
    metadata = list(metadata.values()) # oops, the file is meant to be a list
    
    # Filter out any meta without the 'created_at' field
    metadata = [meta for meta in metadata if 'created_at' in meta]
    
    # Get list of tags ordered by frequency descending
    tags = sorted(tagsdata.keys(), key=lambda x: tagsdata[x]['images'], reverse=True)
    tags = tags[:args.n_tags]
    
    # Init model
    print("Initializing model...")
    model = Model(tags, args.n_blocks, args.n_layers, args.hidden_size, args.dropout, args.batch_norm, nn.ReLU(), args.rezero)
    model.cuda()
    
    # Split metadata list into train, val
    random.shuffle(metadata)
    train_metadata = metadata[:int(0.8 * len(metadata)) ]
    val_metadata   = metadata[ int(0.8 * len(metadata)):]
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = Dataset(train_metadata, model)
    val_dataset   = Dataset(  val_metadata, model)
    
    # Create dataloaders
    print("Creating dataloaders...")
    common_kwargs = {"batch_size": args.batch_size, "num_workers": 3, "collate_fn": collate_func, "persistent_workers": True}
    train_loader = data.DataLoader(train_dataset, **common_kwargs, shuffle=True )
    val_loader   = data.DataLoader(  val_dataset, **common_kwargs, shuffle=False)
    
    # Init optimizer
    print("Initializing optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=0.0, weight_decay=args.weight_decay)
    
    # Init logger
    print("Initializing logger...")
    logger = Logger(args.outdir, args.run_name)
    
    # Init GradNorm tracker
    autoclip = AutoClip(0.1, 1024)
    
    # Init custom scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    
    # Find optimal learning rate
    print("Finding optimal learning rate...")
    best_lr = find_lr(model, optimizer, train_loader, val_loader, logger)
    
    # Train model
    print("Training model...")
    train(model, optimizer, autoclip, train_loader, val_loader, logger)