import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18, resnet50, resnet101


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


def get_model(model_name: str, num_classes: int) -> nn.Module:
    match model_name:
        case "CNN":
            return CNN(num_classes=num_classes)
        case "RESNET18":
            return resnet18(num_classes=num_classes)
        case "RESNET50":
            return resnet50(num_classes=num_classes)
        case "RESNET101":
            return resnet101(num_classes=num_classes)
        case _:
            raise ValueError(f"Invalid model name: {model_name}")
