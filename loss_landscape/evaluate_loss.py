import torch

def eval_loss_transformer(net,criterion,dataloader, percentage):
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    except:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    net.eval()
    loss = []
    acc = []
    with torch.no_grad():
        for inputs,targets in dataloader:
            inputs = inputs.to(device=device, dtype=torch.float)
            targets = targets.to(device=device, dtype=torch.long).squeeze(1)

            predictions = net(inputs)
            loss.append(criterion(predictions, targets).mean())
            correct = predictions.max(dim=1).indices == targets
            acc.append(sum(correct)/len(inputs))
            
    return sum(loss)/len(loss), sum(acc)/len(acc)


def eval_loss_gcn(net,criterion,dataloader):
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    except:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    net.eval()
    loss = []
    acc = []
    
    with torch.no_grad():
        for data in dataloader:
            input_x = data.x.to(device)
            input_edge_index = data.edge_index.to(device)
            input_batch = data.batch.to(device)
            targets = data.y.to(device)

            predictions = net(input_x, input_edge_index, input_batch)
            loss.append(criterion(predictions, targets).mean())
            correct = predictions.max(dim=1).indices == targets
            acc.append(sum(correct)/len(data))
    
    return sum(loss)/len(loss), sum(acc)/len(acc)

