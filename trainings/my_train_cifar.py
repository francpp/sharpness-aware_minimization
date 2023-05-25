import argparse
import torch

import sys; 
sys.path.append(".")

from models.wide_res_net import WideResNet
from models.smooth_cross_entropy import smooth_crossentropy
from cifar.cifar import Cifar

from utilities_cifar.log import Log
# from utility.plots import load_data, record_stats, plot_loss, plot_accuracy
from utilities_cifar.initialize import initialize
from utilities_cifar.step_lr import StepLR
from utilities_cifar.bypass_bn import enable_running_stats, disable_running_stats


from sam import SAM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=8, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=50, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=1, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=2, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--percentage", default=0.05, type=float, help="Percentage to extract from the Cifar Dataset")
    parser.add_argument("--optimizer", default='SGD', type=str, help="SGD or SAM")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Import the dataset and choose the percentage of images to extract randomly from CIFAR10. The subset extracted is balanced.
    dataset = Cifar(args.percentage, args.batch_size, args.threads)
    print('Cifar Training set: ', dataset.train_set.__len__())
    print('Cifar Test set: ', dataset.test_set.__len__())

    # Import the model
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
        
    log = Log(log_each=10, optimizer=args.optimizer, rho=args.rho)
    
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

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)
            
            if args.optimizer == 'SGD':
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing) # torch.Size([batch_size])
                loss.mean().backward()
                optimizer.step()
                optimizer.zero_grad()
            
            elif args.optimizer == 'SAM':
                # first forward-backward step
                enable_running_stats(model)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward step
                disable_running_stats(model)
                smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
                optimizer.second_step(zero_grad=True)
            
            with torch.no_grad():
                correct = predictions.max(dim=1).indices == targets # torch.Size([128])
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)
                
        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = predictions.max(dim=1).indices == targets
                log(model, loss.cpu(), correct.cpu())
        
    log.flush()
    acc = log.final_flush()
    
    state = {
                'acc': acc,
                'state_dict': model.state_dict(),
            }
         
    torch.save(state, 'to_plot/model_cifar_' + args.optimizer + '.pt')

