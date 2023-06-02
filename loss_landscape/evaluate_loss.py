import torch

def eval_loss_attgru(net,criterion,dataloader, percentage):
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    except:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    net.eval()
    loss = []
    acc = []
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if index > int(len(dataloader)*percentage):
                continue
            inputs = batch.text.to(device)
            targets = batch.label.to(device).long()

            predictions, _ = net(inputs)
            loss.append(criterion(predictions, targets).mean())
            correct = predictions.max(dim=1).indices == targets
            acc.append(len(correct)/len(batch))
            
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
            acc.append(len(correct)/len(data))
    
    return sum(loss)/len(loss), sum(acc)/len(acc)

