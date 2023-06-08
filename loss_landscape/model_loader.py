import os
import torch, torchvision

import sys; sys.path.append("..")
from models import wide_res_net, gcn
from models.transformer.model.transformer import Transformer 


# map between model name and function
models = {
    'WideResNet'                  : wide_res_net.WideResNet, 
    'Transformer'                 : Transformer,
    'GCN'                         : gcn.GCN,
}

def load(dataset_name, model_name, model_file, args, DATASET = None, data_parallel=False):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    net.to(device)
    net.eval()
    
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
            
    if dataset_name == 'mitbih' and model_name == 'Transformer':
        net = Transformer(d_model=args.d_model, n_head=args.n_head, max_len=args.max_len, seq_len=args.sequence_len, ffn_hidden=args.ffn_hidden, n_layers=args.n_layer, drop_prob=args.dropout, details=False, device=device).to(device=device)
        
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
