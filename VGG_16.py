import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def load_data(batch_size, num_workers):
    """
    Tải dữ liệu CIFAR-100 và tạo DataLoader cho tập huấn luyện, đánh giá và kiểm tra.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    data_directory = 'cifar100_data'
    train_dataset = datasets.CIFAR100(root=data_directory, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root=data_directory, train=False, download=True, transform=transform)

    train_size = int(0.8 * len(train_dataset))  
    val_size = len(train_dataset) - train_size  
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader