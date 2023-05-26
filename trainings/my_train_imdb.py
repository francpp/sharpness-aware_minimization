import argparse
import torch

import sys; 
sys.path.append(".")

from models.attention_gru import AttentionGru
from models.smooth_cross_entropy import smooth_crossentropy
from DatasetClass.imdb import Imdb

from utilities_cifar.log import Log
from utilities_cifar.initialize import initialize
from utilities_cifar.step_lr import StepLR
from utilities_cifar.bypass_bn import enable_running_stats, disable_running_stats

from utilities_cifar.helpers import *

from sam import SAM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--dropout", default=0.4, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=5, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--percentage", default=0.05, type=float, help="Percentage to extract from the Imdb Dataset")
    parser.add_argument("--optimizer", default='SGD', type=str, help="SGD or SAM")
    parser.add_argument("--embedding_dim", default=300, type=int, help="embedding dimension of the vocabulary")
    parser.add_argument("--hidden_dim", default=32, type=int, help="hidden dimension of the GRU layer")
    parser.add_argument("--output_dim", default=2, type=int, help="output dimension (number of classes)")
    parser.add_argument("--num_layers", default=2, type=int, help="number of layers of the GRU layer")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Import the dataset and choose the percentage of images to extract randomly from CIFAR10. The subset extracted is balanced.
    dataset = Imdb(args.percentage, args.batch_size, args.threads, device)
    print('Imdb Training set: ', dataset.train_set.__len__())
    print('Imdb Validation set: ', dataset.valid_set.__len__())
    
    # select the model
    ATTN_FLAG = True
    vocab_dim = len(dataset.TEXT.vocab)
    model = AttentionGru(vocab_dim, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers, d_rate=args.dropout)
    
    # initialize the logger
    log = Log(log_each=10, optimizer=args.optimizer, rho=args.rho, test_case = 'imdb')
    
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
        log.train(len_dataset=len(dataset.train_set))

        for index, batch in enumerate(dataset.train_iterator):
            if index > int(len(dataset.train_iterator)*args.percentage):
                continue
            inputs = batch.text.to(device)
            targets = batch.label.to(device).long()
            
            if args.optimizer == 'SGD':
                predictions, _ = model(inputs) # torch.Size([batch_size, num_classes])
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing) # torch.Size([batch_size])
                loss.mean().backward()
                optimizer.step()
                optimizer.zero_grad()          
            
            elif args.optimizer == 'SAM':
                # first forward-backward step
                enable_running_stats(model)
                predictions, _ = model(inputs)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)
                
                # second forward-backward step
                disable_running_stats(model)
                smooth_crossentropy(model(inputs)[0], targets, smoothing=args.label_smoothing).mean().backward()
                optimizer.second_step(zero_grad=True)
                
            with torch.no_grad():
                correct = predictions.max(dim=1).indices == targets # torch.Size([batch_size])
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)
                
        model.eval()
        log.eval(len_dataset=len(dataset.valid_set))

        with torch.no_grad():
            for index, batch in enumerate(dataset.valid_iterator):
                if index > int(len(dataset.valid_iterator)*args.percentage):
                    continue
                inputs = batch.text.to(device)
                targets = batch.label.to(device).long()

                predictions, _ = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = predictions.max(dim=1).indices == targets
                log(model, loss.cpu(), correct.cpu())
        
        if epoch==int(args.epochs/2) and args.optimizer=='SAM':  
            acc = log.final__accuracy()
            state_half = {
                    'acc': acc,
                    'state_dict': model.state_dict(),
                }

            torch.save(state_half, 'to_plot/model_imdb_half_' + args.optimizer + '_rho' + str(args.rho) + '.pt')
        
    log.flush()
    acc = log.final__accuracy()
    
    state = {
                'acc': acc,
                'state_dict': model.state_dict(),
            }
    if args.optimizer == 'SAM':
        torch.save(state, 'to_plot/model_imdb_' + args.optimizer + '_rho' + str(args.rho) + '.pt')
    if args.optimizer == 'SGD':
        torch.save(state, 'to_plot/model_imdb_' + args.optimizer + '.pt')

