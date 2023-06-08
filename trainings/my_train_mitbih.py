import argparse
import torch
from torchinfo import summary
import numpy as np 

import sys; 
sys.path.append(".")

import copy
from torch import nn, optim 
from torchinfo import summary 
from models.transformer.model.transformer import Transformer 

from models.smooth_cross_entropy import smooth_crossentropy
from DatasetClass.mitbih import MitBih

from utilities.log import Log
from utilities.initialize import initialize
from utilities.step_lr import StepLR
from utilities.bypass_bn import enable_running_stats, disable_running_stats

from sam import SAM

#-------------------------------------------------------------------------------------------------------------------------------------------
def cross_entropy_loss(pred, target):

    criterion = nn.CrossEntropyLoss()
    lossClass = criterion(pred, target)

    return lossClass


def calc_loss_and_score(pred, target): 
    
    softmax = nn.Softmax(dim=1)
    pred = pred.squeeze(-1)
    target = target.squeeze(-1)
    
    ce_loss = cross_entropy_loss(pred, target)
    print(ce_loss)
    pred = softmax(pred )
    
    _,pred = torch.max(pred, dim=1)
    
    return ce_loss
 
#-------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=50, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=1, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=2, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--optimizer", default='SGD', type=str, help="SGD or SAM")
    parser.add_argument("--max_len", default=5000, type=int, help="Max time series sequence length")
    parser.add_argument("--sequence_len", default=187, type=int, help="Sequence length of time series")
    parser.add_argument("--n_head", default=2, type=int, help="Number of attention head")
    parser.add_argument("--n_layer", default=1, type=int, help="Number of encoder layers")
    parser.add_argument("--d_model", default=200, type=int, help="Dimension (for positional embedding)")
    parser.add_argument("--ffn_hidden", default=128, type=int, help="Size of hidden layer before classification")
    parser.add_argument("--feature", default=200, type=int, help="For univariate time series (1D)")
    
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Import the dataset and choose the percentage of images to extract randomly from MitBih. The subset extracted is balanced.
    dataset = MitBih(args.batch_size, args.threads)
    print('MitBih Training set: ', dataset.train_set.__len__())
    print('MitBih Test set: ', dataset.test_set.__len__())

    # Import the model
    model = Transformer(d_model=args.d_model, n_head=args.n_head, max_len=args.max_len, seq_len=args.sequence_len, ffn_hidden=args.ffn_hidden, n_layers=args.n_layer, drop_prob=args.dropout, details=False, device=device).to(device=device)
    summary(model)
    
    log = Log(log_each=10, optimizer=args.optimizer, rho=args.rho, test_case='mitbih')
    
    # select the optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'SAM':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        
    else:
        raise NotImplementedError
    
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for inputs, targets in dataset.train:
            inputs = inputs.to(device=device, dtype=torch.float)
            targets = targets.to(device=device, dtype=torch.long).squeeze(1)
            
            if args.optimizer == 'SGD':
                predictions = model(inputs)
                # loss = calc_loss_and_score(predictions, targets) 
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing) # torch.Size([batch_size])
                loss.mean().backward()
                optimizer.step()
                optimizer.zero_grad()
            
            elif args.optimizer == 'SAM':
                # first forward-backward step
                enable_running_stats(model)
                predictions = model(inputs)
                # loss = calc_loss_and_score(predictions, targets) 
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward step
                disable_running_stats(model)
                # calc_loss_and_score(model(inputs), targets)
                smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
                optimizer.second_step(zero_grad=True)
            
            with torch.no_grad():
                correct = predictions.max(dim=1).indices == targets # torch.Size([128])
                # print(loss)
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)
                
        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for inputs, targets in dataset.test:
                inputs = inputs.to(device=device, dtype=torch.float)
                targets = targets.to(device=device, dtype=torch.long).squeeze(1)

                predictions = model(inputs)
                # loss = calc_loss_and_score(predictions, targets)
                loss = smooth_crossentropy(predictions, targets)
                correct = predictions.max(dim=1).indices == targets
                log(model, loss.cpu(), correct.cpu())
        
        if epoch==int(args.epochs/2) and args.optimizer=='SAM':  
            acc = log.final__accuracy()
            state_half = {
                    'acc': acc,
                    'state_dict': model.state_dict(),
                }

            torch.save(state_half, 'to_plot/model_mitbih_half_' + args.optimizer + '_rho' + str(args.rho) + '.pt')
        
    log.flush()
    acc = log.final__accuracy()
    
    state = {
                'acc': acc,
                'state_dict': model.state_dict(),
            }
    if args.optimizer == 'SAM':
        torch.save(state, 'to_plot/model_mitbih_' + args.optimizer + '_rho' + str(args.rho) + '.pt')
    if args.optimizer == 'SGD':
        torch.save(state, 'to_plot/model_mitbih_' + args.optimizer + '.pt')

