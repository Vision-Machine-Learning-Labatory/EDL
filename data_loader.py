import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def mnist_loader(config, train=True):
    resize = tuple(config['dataset']['resize'])
    transform_list = [transforms.Resize(resize), transforms.ToTensor()]
    transform = transforms.Compose(transform_list)

    dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=config['dataset']['batch_size'], shuffle=train)
    return loader

def get_data_loader(config, train):
    if config['dataset']['name'] == "MNIST":
        return mnist_loader(config=config, train=train)
    else:
        raise ValueError(f"Unknown data type: {config['dataset']['name']}")