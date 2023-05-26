import argparse
import torch

import sys; 
sys.path.append("../")

from models.gcn import GCN
from models.smooth_cross_entropy import smooth_crossentropy

from utilities_gcn.log import Log
# from utility.plots import load_data, record_stats, plot_loss, plot_accuracy
from utilities_gcn.initialize import initialize
from utilities_gcn.step_lr import StepLR
from utilities_gcn.bypass_bn import enable_running_stats, disable_running_stats
from DatasetClass.TUD import GraphDataset

from sam import SAM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=50, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--rho", default=0.3, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--optimizer", default='ADAM', type=str, help="ADAM or SAM")
    parser.add_argument("--train_rate", default=70, type=int, help="Train rate, [0,100]")
    parser.add_argument("--hidden-channels", default=64, type=int, help="Hidden channels of convolutional layers")
    args = parser.parse_args()

    initialize(args, seed=42)
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #################################### Import the dataset ##########################################
    '''
    from torch_geometric.datasets import TUDataset
    
    dataset = TUDataset(root='data/TUDataset', name='Mutagenicity')

    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    
    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    
    size_train = int(len(dataset)*args.train_rate/100)
    train_dataset = dataset[:size_train]
    test_dataset = dataset[size_train:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    
    from torch_geometric.loader import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    '''
    name='Mutagenicity'
    Graphs = GraphDataset(name, args.train_rate, args.batch_size)
    print(len(Graphs))
    Graphs.GlobalStats()
    Graphs.LocalStats()
    
    ########################### Import the model ####################################################
    from torchinfo import summary
    model = GCN(args.hidden_channels, Graphs.dataset.num_node_features, Graphs.dataset.num_classes).to(device)
    summary(model)
        
    log = Log(log_each=10, optimizer=args.optimizer, rho=args.rho)
    
    ########################## Train the model ################################################
    if args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'SAM':
        base_optimizer = torch.optim.Adam
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, weight_decay=args.weight_decay)
        
    else:
        raise NotImplementedError
    
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(Graphs.train_dataset))

        for data in Graphs.train_loader:
            input_x = data.x.to(device)
            input_edge_index = data.edge_index.to(device)
            input_batch = data.batch.to(device)
            targets = data.y.to(device)
            
            if args.optimizer == 'ADAM':
                predictions = model(input_x, input_edge_index, input_batch)
                #print('gcn predicitions: ', predictions.shape) # shape [batch_size, num_classes]
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing) # torch.Size([batch_size])
                loss.mean().backward()
                optimizer.step()
                optimizer.zero_grad()
            
            elif args.optimizer == 'SAM':
                # first forward-backward step
                enable_running_stats(model)
                predictions = model(input_x, input_edge_index, input_batch)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward step
                disable_running_stats(model)
                smooth_crossentropy(model(input_x, input_edge_index, input_batch), targets, smoothing=args.label_smoothing).mean().backward()
                optimizer.second_step(zero_grad=True)
            
            with torch.no_grad():
                correct = predictions.max(dim=1).indices == targets # torch.Size([64])
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)
                
        model.eval()
        log.eval(len_dataset=len(Graphs.test_dataset))

        with torch.no_grad():
            for data in Graphs.test_loader:
                input_x = data.x.to(device)
                input_edge_index = data.edge_index.to(device)
                input_batch = data.batch.to(device)
                targets = data.y.to(device)

                predictions = model(input_x, input_edge_index, input_batch)
                loss = smooth_crossentropy(predictions, targets)
                correct = predictions.max(dim=1).indices == targets
                log(model, loss.cpu(), correct.cpu())
        
    log.flush()
    acc = log.final_flush()
    
    state = {
                'acc': acc,
                'state_dict': model.state_dict(),
            }
         
    #torch.save(state, 'to_plot/model_gcn_' + args.optimizer + '.pt')

    