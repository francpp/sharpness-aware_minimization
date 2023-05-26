import os
import torch, torchvision

import sys; sys.path.append("..")
from models import wide_res_net, attention_gru

# map between model name and function
models = {
    'WideResNet'                  : wide_res_net.WideResNet, 
    'AttentionGru'                : attention_gru.AttentionGru,
}

def load(dataset, model_name, model_file, args, data_parallel=False):
    if dataset == 'cifar10':
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
            
#     if dataset == 'imdb':
#         net = models[model_name](...change...)
#         if data_parallel: # the model is saved in data paralle mode
#             net = torch.nn.DataParallel(net)

#         if model_file:
#             assert os.path.exists(model_file), model_file + " does not exist."
#             stored = torch.load(model_file, map_location=lambda storage, loc: storage)
#             if 'state_dict' in stored.keys():
#                 net.load_state_dict(stored['state_dict'])
#             else:
#                 net.load_state_dict(stored)

#         if data_parallel: # convert the model back to the single GPU version
#             net = net.module

#         net.eval()
        
    return net
