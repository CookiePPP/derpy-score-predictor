# Written by CookieGalaxy / CookiePPP
# This file will contain all the dataloading code for the model.

# This model will take
# - FloatTensor[B, n_tags] where each element is the probability of a tag being present.
# -  LongTensor[B,      1] where value is the year starting from 2010.
# -  LongTensor[B,      1] where value is the week in the year.
# -  LongTensor[B,      1] where value is the day of the week.
# -  LongTensor[B,      1] where value is the hour of the day.

# imports
import torch
import torch.nn as nn
import torch.utils.data as data
import datetime
import random

# create preprocessing class
class Preprocess(data.Dataset):
    def __init__(self, model):
        self.tags = model.tags # [tag0, tag1, tag2, ...]
        self.tags_to_idx = {tag: idx for idx, tag in enumerate(self.tags)} # {tag0: 0, tag1: 1, tag2: 2, ...}
    
    def __call__(self, tag_str, datetime_str):
        """
        Turn date+time+tags into a model compatible format
        Example Args:
            tag_str = "safe, artist:marminatoror, edit, rainbow dash, scootaloo, pegasus, pony"
            datetime_str = "2019-01-01T00:00:00Z"
        Example Return:
            {
                "tags" : FloatTensor[1, n_tags],
                "year" :  LongTensor[1,      1],
                "week" :  LongTensor[1,      1],
                "day"  :  LongTensor[1,      1],
                "hour" :  LongTensor[1,      1],
            }
        """
        # split tags
        tags = tag_str.split(", ") # get list of tags
        tags = [tag for tag in tags if tag in self.tags] # remove tags not in model
        tags = [self.tags_to_idx[tag] for tag in tags] # convert to indices
        tags_tensor = torch.zeros(1, len(self.tags)) # create zeros tensor [B, n_tags]
        tags_tensor[0, tags] = 1 # set tag indexes to 1 (one-hot encoding)
        
        # load into datetime
        datetime_obj = datetime.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%SZ")
        # get year, week, day, hour
        year = datetime_obj.year - 2010 # 0 = 2010, 1 = 2011, ...
        week = datetime_obj.isocalendar()[1] - 1 # 0 = week 1, 1 = week 2, ...
        day = datetime_obj.weekday() # 0 = Monday, 6 = Sunday
        hour = datetime_obj.hour # 0-23
        
        # get weeks since 2010
        abwk = int((datetime_obj.year - 2010) * 52.1429) + datetime_obj.isocalendar()[1] - 1
        
        # convert to tensors
        year = torch.tensor(year, dtype=torch.long).view(1, 1) # [B, 1]
        week = torch.tensor(week, dtype=torch.long).view(1, 1) # [B, 1]
        day  = torch.tensor( day, dtype=torch.long).view(1, 1) # [B, 1]
        hour = torch.tensor(hour, dtype=torch.long).view(1, 1) # [B, 1]
        abwk = torch.tensor(abwk, dtype=torch.long).view(1, 1) # [B, 1]
        
        
        # return
        return {
            "tags" : tags_tensor,
            "year" : year,
            "week" : week,
            "day"  : day,
            "hour" : hour,
            "abwk" : abwk,
        }

# create dataset class
class Dataset(data.Dataset):
    def __init__(self, metadata: list[dict], model: nn.Module):
        self.metadata = metadata
        self.preprocess = Preprocess(model)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        d = self.metadata[index]
        # get input tensors
        tag_str = ", ".join(d['rating']+d['characters']+d['ocs']+d['tags']) # does not include artist, art pack, series, etc.
        tag_str = tag_str.lower()
        datetime_str = d['created_at']
        tensors_dict = self.preprocess(tag_str, datetime_str)
        
        # get target tensors
        wilson_score = torch.tensor(d['wilson_score'], dtype=torch.float).view(1, 1) # [B, 1]
        score        = torch.tensor(d['score'       ], dtype=torch.float).view(1, 1) # [B, 1]
        upvotes      = torch.tensor(d['upvotes'     ], dtype=torch.float).view(1, 1) # [B, 1]
        downvotes    = torch.tensor(d['downvotes'   ], dtype=torch.float).view(1, 1) # [B, 1]
        
        # add random [-0.5, 0.5] noise to score, upvotes, downvotes
        random_obj = random.Random(index)
        score     += random_obj.uniform(-0.5, 0.5)
        upvotes   += random_obj.uniform(-0.5, 0.5)
        downvotes += random_obj.uniform(-0.5, 0.5)
        
        # return
        return {
            "id": d['id'],
            "input" : tensors_dict,
            "target": {
                "wilson_score": wilson_score,
                "score"       : score,
                "upvotes"     : upvotes,
                "downvotes"   : downvotes,
            }
        }

def collate_func(batch):
    """Tasks list of {id, input, target} dicts into {input, target} tensors"""
    # get ids
    ids = [d['id'] for d in batch]
    # get input tensors
    tags = torch.cat([d['input']['tags'] for d in batch], dim=0) # FloatTensor[B, n_tags]
    year = torch.cat([d['input']['year'] for d in batch], dim=0) #  LongTensor[B,      1]
    week = torch.cat([d['input']['week'] for d in batch], dim=0) #  LongTensor[B,      1]
    day  = torch.cat([d['input']['day' ] for d in batch], dim=0) #  LongTensor[B,      1]
    hour = torch.cat([d['input']['hour'] for d in batch], dim=0) #  LongTensor[B,      1]
    abwk = torch.cat([d['input']['abwk'] for d in batch], dim=0) #  LongTensor[B,      1]
    
    # get target tensors
    wilson_score = torch.cat([d['target']['wilson_score'] for d in batch], dim=0) # FloatTensor[B, 1]
    score        = torch.cat([d['target']['score'       ] for d in batch], dim=0) # FloatTensor[B, 1]
    upvotes      = torch.cat([d['target']['upvotes'     ] for d in batch], dim=0) # FloatTensor[B, 1]
    downvotes    = torch.cat([d['target']['downvotes'   ] for d in batch], dim=0) # FloatTensor[B, 1]
    # return
    return {
        "ids": ids,
        "input" : {
            "tags": tags,
            "year": year,
            "week": week,
            "day" : day,
            "hour": hour,
            "abwk": abwk,
        },
        "target": {
            "wilson_score": wilson_score,
            "score"       : score,
            "upvotes"     : upvotes,
            "downvotes"   : downvotes,
        }
    }