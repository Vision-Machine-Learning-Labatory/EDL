import torch

def evaluate(model, loader, config, loss_fn, edl, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            if edl:
                pred = torch.argmax(output['probability'], dim=1)
                loss = loss_fn(output['alpha'], y, config['model']['num_classes'])
            else:
                pred = torch.argmax(output, dim=1)
                loss = loss_fn(output, y)
            correct += (pred == y).sum().item()
            total_loss += loss.item()

    accuracy = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)
    return accuracy, avg_loss