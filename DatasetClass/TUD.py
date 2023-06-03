import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

class GraphDataset:
    def __init__(self, name, train_rate, batch_size):
        torch.manual_seed(12345)
        try:
            self.dataset = TUDataset(root='DatasetClass/TUDataset', name=name).shuffle()
        except:
            self.dataset = TUDataset(root='../DatasetClass/TUDataset', name=name).shuffle()

        self.batch_size = batch_size
        self.size_train = int(len(self.dataset)*train_rate/100)
        self.train_dataset = self.dataset[:self.size_train]
        self.test_dataset = self.dataset[self.size_train:]
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
    def __len__(self):
        return len(self.dataset)  # number of graphs
    
    def __getitem__(self, index):
        data = self.dataset[index]
        input_x = data.x
        input_edge_index = data.edge_index
        input_batch = data.batch
        targets = data.y
        inputs = [input_x, input_edge_index, input_batch]
        outputs = [targets]
        return inputs, outputs
    
    def GlobalStats(self):
        print()
        print(f'Dataset: {self.dataset}:')
        print('====================')
        print(f'Number of graphs: {len(self)}')
        print(f'Number of features: {self.dataset.num_features}')
        print(f'Number of classes: {self.dataset.num_classes}')
        
    def LocalStats(self):
        data = self.dataset[0]  # Get the first graph object.
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

        
        

