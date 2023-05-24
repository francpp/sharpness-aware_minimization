import torch.nn as nn
import torch.optim as optim
import torch

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from tqdm import tqdm

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)

    return acc


def attn_visualization(model, iterator, TEXT, multiple_flag=False):
    """
    Visualize self-attention weights with input captions.
    """

    if multiple_flag is False:
        with torch.no_grad():
            batch = next(iter(iterator))
            _, attention = model(batch.text)

            # in torchtext, batch_size is placed in dim=1. dim=0 is used for sentence length
            text = batch.text.transpose(0, 1)
            # print(attention.size())
            attention_weight = attention.cpu().numpy()

            itos = []
            for text_element in text:
                itos_element = []
                for index in text_element:
                    # print(f'{TEXT.vocab.itos[index]} ')
                    itos_element.append(TEXT.vocab.itos[index])
                itos.append(itos_element)

            plt.figure(figsize = (16, 5))
            sns.heatmap(attention_weight, annot=np.asarray(itos), fmt='', cmap='Blues')
            plt.savefig('attention.png')

    elif multiple_flag is not False:
        with torch.no_grad():
            batch_count = 0
            for batch in iterator:
                _, attention = model(batch.text)
                text = batch.text.transpose(0, 1)
                attention_weight = attention.cpu().numpy()
                
                itos = []
                for text_element in text:
                    itos_element = []
                    for index in text_element:
                        itos_element.append(TEXT.vocab.itos[index])
                    itos.append(itos_element)
                
                fig_size = len(batch.text) + 1 # for changing fig_size dynamically
                plt.figure(figsize = (fig_size, 7))
                sns.heatmap(attention_weight, annot=np.asarray(itos), fmt='', cmap='Blues')
                plt.savefig('./fig/attention_' + str(batch_count) + '.png')
                plt.close()

                if batch_count == 10:
                    break
                batch_count += 1
                
"""
import torch
from torchtext import data, datasets
import random

def load_data(BATCH_SIZE=16):
    SEED = 1234

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    TEXT = data.Field() # tokenize='spacy
    LABEL = data.LabelField(dtype=torch.float)
    
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, '.')
    # train_data contains an attribute .__dict__ that has 2 fields:
    # - text
    # - label
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))
    print('train_data: ', len(train_data))
    print('valid_data: ', len(valid_data))
    print('test_data: ', len(test_data))
    
    # train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # print('train_dataloader: ', len(train_dataloader))
    # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    # print('test_dataloader: ', len(test_dataloader))

    TEXT.build_vocab(train_data, max_size=len(train_data), vectors="glove.6B.300d")
    LABEL.build_vocab(train_data)
    print('TEXT.vocab: ', TEXT.vocab)
    print('LABEL.vocab: ', LABEL.vocab)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size=BATCH_SIZE, 
        device=device)
    print('train_iterator: ', len(train_iterator))
    print('valid_iterator: ', len(valid_iterator))
    print('test_iterator: ', len(test_iterator)) 
    
    return TEXT, LABEL, train_iterator, valid_iterator, test_iterator, train_data, valid_data, test_data
    
"""