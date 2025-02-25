import torch
import torch.nn as nn
from googlenet import GoogLeNet
from dataset import get_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(data_path):
    test_loader = get_data_loader(data_path, batch_size=64, train=False)
    model = GoogLeNet(num_classes=len(test_loader.dataset.classes)).to(device)
    model.load_state_dict(torch.load("../models/googlenet_imagenet.pth"))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    test_model("/path/to/imagenet-1k")
