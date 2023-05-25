import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from random import randint

from utilities_cifar.cutout import Cutout

class Cifar:
    def __init__(self, percentage, batch_size, threads):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.train_set = Cifar10Subset(percentage=percentage, root='./cifar', train=True, transform=train_transform)
        self.test_set = Cifar10Subset(percentage=percentage, root='./cifar', train=False, transform=test_transform)
        
        self.train = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        train_set = datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])


class Cifar10Subset(Dataset):
    def __init__(self, percentage, root, train=True, transform=None):
        cifar_dataset = datasets.CIFAR10(root=root, train=train, download=True)

        # Divide the dataset into separate classes based on labels
        class_data = [[] for _ in range(10)]
        for i in range(len(cifar_dataset)):
            image, label = cifar_dataset[i]
            class_data[label].append(image)

        # Calculate the number of samples required for each class
        num_samples = int(percentage * len(cifar_dataset) / 10)
        
        # Randomly select samples from each class to make a balanced subset
        subset_data = []
        for class_samples in class_data:
            subset_data.extend(torch.utils.data.random_split(class_samples, [num_samples, len(class_samples) - num_samples])[0])

        self.data = subset_data
        self.targets = [i for i in range(10) for j in range(num_samples)]
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


"""
# the former code was:

class Cifar:
    def __init__(self, batch_size, threads):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        
        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
        
"""
