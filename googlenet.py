import torch
import torch.nn as nn
import torchvision.models as models

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()
        self.model = models.googlenet(pretrained=True)
        self.model.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = GoogLeNet(num_classes=10)  # Example for 10 classes
    print(model)
