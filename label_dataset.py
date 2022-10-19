# Written by CookieGalaxy / CookiePPP
# This file will fake a model and a dataset and save a file with the score percentiles of each image.
import json

import torch
from tqdm import tqdm

from model import Model
from data import Dataset, collate_func
import torch.utils.data as data
import pickle as pkl

# disable gradient calculation
torch.set_grad_enabled(False)


if __name__ == '__main__':
    checkpoint_path = "runs/v3_8nb_2nl_512hs_0.5do_512bs_256tags_2patience_2wf/best_val_model.pt"
    metadata = "data/metadata.p"
    batch_size = 2048
    version = '_v3' # used in exported filename
    
    # get model + dataset
    model = Model.load_from_checkpoint(checkpoint_path).cuda()
    
    metadata = pkl.load(open(metadata, "rb"))
    metadata = list(metadata.values()) # oops, the file is meant to be a list
    metadata = [meta for meta in metadata if 'created_at' in meta] # Filter out any meta without the 'created_at' field
    dataset = Dataset(metadata, model)

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": 3,
        "collate_fn": collate_func,
        "persistent_workers": True
    }
    dataloader = data.DataLoader(dataset, **common_kwargs, shuffle=False)
    
    # calculate percentile
    scores = {}
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            output = model.infer_percentile(batch['input'], batch['target'], mc_dropout_sampling=100)
            for i, id in enumerate(batch['ids']):
                scores[id] = output[i]
    # sort by score descending
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1]['score'], reverse=True)}
    
    # save as pkl
    pkl.dump(scores, open(f"data/scores{version}.p", "wb"))
    # and as json
    json.dump(scores, open(f"data/scores{version}.json", "w"))
    # and as csv
    with open(f"data/scores{version}.csv", "w") as f:
        f.write("id,percentile\n")
        for id, score in scores.items():
            f.write(f"{id},{score['score']:.3f}\n")
    
    print("Done!")