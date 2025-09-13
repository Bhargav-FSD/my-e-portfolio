from scripts.train import train_cnn, train_alexnet
from scripts.utils import load_data

# Load CIFAR-10 subset
train_loader, test_loader = load_data("data/cifar_subset")

# Train CNN
cnn_results = train_cnn(train_loader, test_loader)

# Train AlexNet
alexnet_results = train_alexnet(train_loader, test_loader)

print("CNN Results:", cnn_results)
print("AlexNet Results:", alexnet_results)
