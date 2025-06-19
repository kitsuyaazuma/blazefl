import torch
from blazefl.core import ModelSelector
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class FedAvgModelSelector(ModelSelector):
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def select_model(self, model_name: str) -> nn.Module:
        match model_name:
            case "cnn":
                return CNN(num_classes=self.num_classes)
            case "resnet18":
                return resnet18(num_classes=self.num_classes)
            case _:
                raise ValueError


class CNN(nn.Module):
    """
    Based on
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    with slight modifications.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
