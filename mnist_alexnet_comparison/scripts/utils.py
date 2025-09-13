import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

def load_data(dataset_path, batch_size=32):
    """
    Loads CIFAR-10 and MNIST subsets from the specified path.
    
    Args:
        dataset_path (str): Path to the dataset folder (e.g., 'data/cifar_subset')
        batch_size (int): Batch size for DataLoader
    
    Returns:
        train_loader, test_loader
    """
    # Transform: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_path,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_path,
        train=False,
        download=True,
        transform=transform
    )

    # Optional: take only a small subset (e.g., first 50 per class)
    # For simplicity, we can take first 250 images
    train_dataset = Subset(train_dataset, range(250))
    test_dataset = Subset(test_dataset, range(50))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
