import os
import torch, torchvision

import sys; sys.path.append("..")
from models import wide_res_net, attention_gru, gcn

# map between model name and function
models = {
    'WideResNet'                  : wide_res_net.WideResNet, 
    'AttentionGru'                : attention_gru.AttentionGru,
    'GCN'                         : gcn.GCN,
}

def load(dataset_name, model_name, model_file, args, DATASET = None, data_parallel=False):
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if dataset_name == 'cifar10' and model_name == 'WideResNet':
        net = models[model_name](args.depth, args.width_factor, args.dropout, in_channels=3, labels=10)
        if data_parallel: # the model is saved in data paralle mode
            net = torch.nn.DataParallel(net)

        if model_file:
            assert os.path.exists(model_file), model_file + " does not exist."
            stored = torch.load(model_file, map_location=lambda storage, loc: storage)
            if 'state_dict' in stored.keys():
                net.load_state_dict(stored['state_dict'])
            else:
                net.load_state_dict(stored)

        if data_parallel: # convert the model back to the single GPU version
            net = net.module
            
    if dataset_name == 'imdb' and model_name == 'AttentionGru':
        ATTN_FLAG = True
        vocab_dim = len(DATASET.TEXT.vocab)
        net = models[model_name](vocab_dim, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers, d_rate=args.dropout)
        if data_parallel: # the model is saved in data paralle mode
            net = torch.nn.DataParallel(net)

        if model_file:
            assert os.path.exists(model_file), model_file + " does not exist."
            stored = torch.load(model_file, map_location=lambda storage, loc: storage)
            if 'state_dict' in stored.keys():
                net.load_state_dict(stored['state_dict'])
            else:
                net.load_state_dict(stored)

        if data_parallel: # convert the model back to the single GPU version
            net = net.module

        net.eval()

    if model_name == 'GCN':
        Graphs = DATASET
        net = models[model_name](args.hidden_channels, Graphs.dataset.num_node_features, Graphs.dataset.num_classes).to(device)

        if data_parallel: # the model is saved in data paralle mode
            net = torch.nn.DataParallel(net)

        if model_file:
            assert os.path.exists(model_file), model_file + " does not exist."
            stored = torch.load(model_file, map_location=lambda storage, loc: storage)
            if 'state_dict' in stored.keys():
                net.load_state_dict(stored['state_dict'])
            else:
                net.load_state_dict(stored)

        if data_parallel: # convert the model back to the single GPU version
            net = net.module
        
        
    return net
