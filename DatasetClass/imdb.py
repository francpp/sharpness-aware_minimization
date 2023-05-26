import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torchtext import data, datasets
from torch.utils.data import DataLoader, Dataset
import random

from utilities_cifar.cutout import Cutout

def split_train_valid_and_select(train_data, percentage):
    # split train and validation set, and select a given percentage
    total = len(train_data)
    pos_idx = range(0, int(total/2))
    print(pos_idx)
    neg_idx = range(int(total/2), int(total))
    print(neg_idx)
    
    train_pos_idx = pos_idx[:int(percentage*0.75*total/2)]
    print(train_pos_idx)
    valid_pos_idx = pos_idx[int(0.75*total/2):int(total/2*(0.75+percentage*0.25))]
    print(valid_pos_idx)
    train_neg_idx = neg_idx[:int(percentage*0.75*total/2)]
    print(train_neg_idx)
    valid_neg_idx = neg_idx[int(0.75*total/2):int(total/2*(0.75+percentage*0.25))]
    print(valid_neg_idx)
    
    train_idx = list(train_pos_idx)+list(train_neg_idx)
    valid_idx = list(valid_pos_idx)+list(valid_neg_idx)
    
    return train_idx, valid_idx

class Imdb:
    def __init__(self, percentage, batch_size, threads, device):
        dataset = ImdbSubset(percentage=percentage, batch_size=batch_size, root='./DatasetClass/imdb', seed=42, device=device)

        self.train_set = dataset.train_data
        self.valid_set = dataset.valid_data
        self.test_set = dataset.test_data
        
        self.TEXT = dataset.TEXT
        self.LABEL = dataset.LABEL
        
        self.classes = ('pos', 'neg')
        
        self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
                                                                        (self.train_set, self.valid_set, self.test_set), 
                                                                        batch_size=batch_size, 
                                                                        device=device)    


class ImdbSubset(Dataset):
    def __init__(self, percentage, batch_size, root, seed, device):
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.TEXT = data.Field() # tokenize('spacy')
        self.LABEL = data.LabelField(dtype=torch.float)

        self.train_data, self.test_data = datasets.IMDB.splits(self.TEXT, self.LABEL)
        self.train_data, self.valid_data = self.train_data.split(random_state=random.seed(seed))

        self.TEXT.build_vocab(self.train_data, max_size=len(self.train_data), vectors="glove.6B.300d")
        self.LABEL.build_vocab(self.train_data)
    
    
    def __len__(self):
        return len(self.train_data + self.valid_data + self.test_data)
    
    def __getitem__(self, index):
        
        x = self.train_data[index]
        return x