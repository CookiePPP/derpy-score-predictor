# Written by CookieGalaxy / CookiePPP
# This file will contain all the training and evaluation code for this model.
import os
import random
import traceback
from copy import deepcopy
from typing import Optional

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
from test import test_model
from utils import AutoClip, to_device
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for training.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for optimizer.")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout used between layers.")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="Size of latents between blocks in the model. Can be used like a bottleneck dim.")
    parser.add_argument("--widen_factor", type=int, default=2,
                        help="Increase the size of hidden layers inside blocks by this factor.")
    parser.add_argument("--n_blocks", type=int, default=8,
                        help="Number of resudual blocks in the model.")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of layers in the model.")
    parser.add_argument("--n_tags", type=int, default=256,
                        help="Number of tags for training.")
    parser.add_argument("--batch_norm", action="store_true",
                        help="Use batch norm for training.")
    parser.add_argument("--rezero", action="store_true",
                        help="Use rezero for training.")
    
    parser.add_argument("--metadata", type=str, default="data/metadata.p",
                        help="Path to metadata for training.")
    parser.add_argument("--tagsdata", type=str, default="data/tagsdata.p",
                        help="Path to tagsdata for training.")
    parser.add_argument("--test_data", type=str, default="data/ordered_ids.txt",
                        help="Optional ordered list json file of image IDs.")
    
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
    expavg_train_loss = 10.0
    
    # run model till loss explodes
    model.train()
    optimizer.param_groups[0]["lr"] = initial_lr
    loss_is_normal = True
    while loss_is_normal:
        for batch in tqdm(train_loader, desc="Finding LR"):
            # calc LR for this step
            lr = initial_lr * (10 ** (model.iteration.item() / n_iters_to_10x))
            optimizer.param_groups[0]["lr"] = lr
            
            # update model
            optimizer.zero_grad()
            loss_dict = model.step(batch)
            loss_dict['total'].backward()
            loss = loss_dict['total'].item()
            expavg_train_loss = 0.98*expavg_train_loss + 0.02*loss

            # exit if loss is high
            if model.iteration.item() > n_iters_to_10x*2 and loss_dict['total'].item() > 10*expavg_train_loss:
                optimizer.zero_grad()
                loss_is_normal = False
                break
            
            autoclip(model) # auto-clip the models gradients
            optimizer.step()
            model.iteration += 1
            
            # log results
            logger.log(model, optimizer, loss_dict, dataset_name="train")
            if loss_dict['total'].item() < best_train_loss:
                best_train_loss = loss_dict['total'].item()
                best_loss_iter = model.iteration.item()
                best_model_state     = to_device(model    .state_dict(), 'cpu')
                best_optimizer_state = to_device(optimizer.state_dict(), 'cpu')
                best_autoclip_state = deepcopy(autoclip.state_dict())
                best_logger_state   = deepcopy(logger  .state_dict())
        model.epoch += 1
    
    # calculate LR of best model
    best_lr = initial_lr * (10 ** (best_loss_iter / n_iters_to_10x))
    optimizer.param_groups[0]["lr"] = best_lr
    
    # load best model
    model    .load_state_dict(best_model_state    )
    optimizer.load_state_dict(best_optimizer_state)
    autoclip .load_state_dict(best_autoclip_state )
    logger   .load_state_dict(best_logger_state   )
    
    return best_lr

def train(model: Model, optimizer, autoclip, train_loader, val_loader, logger: Logger, metadata, ordered_ids: Optional[list[str]]):
    # track the best validation model
    best_val_loss = float("inf")
    best_val_loss_epoch = 0
    best_val_model_state = None
    best_val_optimizer_state = None
    best_val_autoclip_state = None
    best_val_logger_state = None
    
    best_test_loss = float("inf")
    best_test_model_state = None
    best_test_optimizer_state = None
    best_test_autoclip_state = None
    best_test_logger_state = None
    
    # scheduler args
    patience = 2 # if validation loss does not decrease for this many epochs, reduce LR
    factor = 0.1**0.5 # reduce LR by this factor
    
    # train model
    model.train()
    passes = 0
    while optimizer.param_groups[0]["lr"] > 1e-6:
        # train model for one epoch
        model.train()
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
                loss_dict['total'].data.fill_(100.0)
            optimizer.step()
            model.iteration += 1
            
            # log results
            logger.log(model, optimizer, loss_dict, dataset_name="train")
            train_loss += loss_dict['total'].item()
        model.epoch += 1
        train_loss /= len(train_loader)
        
        with torch.no_grad():
            # validate model
            val_loss = 0.0
            n_loss = 0
            for i, batch in tqdm(enumerate(val_loader), desc=f"  Val|Dataset Pass {passes}"):
                with torch.random.fork_rng():
                    torch.random.manual_seed(i)
                    loss_dict = model.step(batch)
                if model.iteration.item() > 3000 and loss_dict['total'].item() > 1e3:
                    continue
                val_loss += loss_dict['total'].item()
                n_loss += 1
            val_loss = val_loss/n_loss if n_loss else 100.0
            
            # log results
            logger.log(model, optimizer, {'total': val_loss}, dataset_name="val")
            tqdm.write(f"ITER: {model.iteration.item()} EPOCH: {model.epoch.item()} TRAIN LOSS: {train_loss:.4f} VAL LOSS: {val_loss:.4f}")
            
            # if validation loss decreases, save model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = model.epoch.item()
                best_val_model_state     = to_device(model    .state_dict(), 'cpu')
                best_val_optimizer_state = to_device(optimizer.state_dict(), 'cpu')
                best_val_autoclip_state  = deepcopy(autoclip.state_dict())
                best_val_logger_state    = deepcopy(logger  .state_dict())
            
            # if validation loss does not decrease for patience epochs, load best model and reduce LR
            if model.epoch.item() - best_val_loss_epoch > patience:
                # load best model
                model    .load_state_dict(best_val_model_state    )
                optimizer.load_state_dict(best_val_optimizer_state)
                autoclip .load_state_dict(best_val_autoclip_state )
                logger   .load_state_dict(best_val_logger_state   )
                
                # reduce LR
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= factor
                    tqdm.write(f"LR reduced to {param_group['lr']:.1e}")
                    tqdm.write(f"Restored model from epoch {best_val_loss_epoch}")
                
                # update best_val_optimizer_state with new LR
                best_val_optimizer_state = to_device(optimizer.state_dict(), 'cpu')
            
            # maybe do test eval with sorted ids list
            if ordered_ids is not None:
                try:
                    test_loss = test_model(model, logger, metadata, ordered_ids)['best_inaccuracy']
                except Exception as ex:
                    # print stack trace and continue
                    print(ex)
                    traceback.print_exc()
                    test_loss = float("inf")
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_test_model_state     = to_device(model    .state_dict(), 'cpu')
                    best_test_optimizer_state = to_device(optimizer.state_dict(), 'cpu')
                    best_test_autoclip_state  = deepcopy(autoclip.state_dict())
                    best_test_logger_state    = deepcopy(logger  .state_dict())

        model.eval()
        model.train()
        passes += 1
        tqdm.write("") # newline
    
    # load val best model
    model    .load_state_dict(best_val_model_state    )
    optimizer.load_state_dict(best_val_optimizer_state)
    autoclip .load_state_dict(best_val_autoclip_state )
    logger   .load_state_dict(best_val_logger_state   )
    
    # save to disk
    prefix = "best_val_"
    save_to_disk(autoclip, best_val_loss, logger, model, optimizer, prefix)
    
    # maybe save best test model
    if ordered_ids is not None:
        model    .load_state_dict(best_test_model_state    )
        optimizer.load_state_dict(best_test_optimizer_state)
        autoclip .load_state_dict(best_test_autoclip_state )
        logger   .load_state_dict(best_test_logger_state   )
        
        save_to_disk(autoclip, best_test_loss, logger, model, optimizer, "best_test_")


def save_to_disk(autoclip, best_val_loss, logger, model, optimizer, prefix):
    model.save(os.path.join(logger.run_dir, f"{prefix}model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(logger.run_dir, f"{prefix}optimizer.pt"))
    torch.save(autoclip.state_dict(), os.path.join(logger.run_dir, f"{prefix}autoclip.pt"))
    # write text file with best validation loss
    with open(os.path.join(logger.run_dir, f"{prefix}loss.txt"), "w") as f:
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

    ordered_ids = eval(open(args.test_data).read()) if os.path.exists(args.test_data) else None
    if ordered_ids is None:
        print("No test list found. Ignoring test set.")
    
    # Filter out any meta without the 'created_at' field
    metadata = [meta for meta in metadata if 'created_at' in meta]
    
    # Get list of tags ordered by frequency descending
    tags = sorted(tagsdata.keys(), key=lambda x: tagsdata[x]['images'], reverse=True)
    # ignore any artist tags
    #blacklist = ['commission', 'source needed', 'cropped', 'image macro', 'high res', 'dead source', 'artist needed', 'edited screencap', 'alternate version', 'edit', 'screencap']
    whitelist = open('whitelist.txt').read().splitlines()
    whitelist = [tag.strip().lower() for tag in whitelist]
    whitelist = [tag for tag in whitelist if tag in tags]
    tags = [tag for tag in tags if tag in whitelist]
    tags = tags[:args.n_tags]
    assert len(tags) == args.n_tags, f"Expected {args.n_tags} tags, got {len(tags)}"
    
    # Init model
    print("Initializing model...")
    model = Model(
        tags,
        n_blocks=args.n_blocks, n_layers    =args.n_layers    , hidden_size=args.hidden_size,
        dropout =args.dropout , batch_norm  =args.batch_norm  , act_func   =nn.SiLU(),
        rezero  =args.rezero  , widen_factor=args.widen_factor,
    )
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
    common_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "collate_fn": collate_func, "persistent_workers": True}
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
    
    # Find optimal learning rate
    print("Finding optimal learning rate...")
    best_lr = find_lr(model, optimizer, train_loader, val_loader, logger)
    
    # Train model
    print("Training model...")
    train(model, optimizer, autoclip, train_loader, val_loader, logger, metadata, ordered_ids)