import torch
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

fds = None  # Cache FederatedDataset

pytorch_transforms = Compose(
    [
        ToTensor(),
        RandomHorizontalFlip(p=0.5),
        RandomCrop(size=32, padding=4),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

pytorch_test_transforms = Compose(
    [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def apply_test_transforms(batch):
    batch["img"] = [pytorch_test_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = PathologicalPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            num_classes_per_partition=2,
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id).with_transform(apply_transforms)
    trainloader = DataLoader(
        partition,  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        shuffle=True,
    )
    return trainloader


def load_centralized_dataset(batch_size: int):
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_test_transforms)  # pyright: ignore[reportAttributeAccessIssue]
    return DataLoader(dataset, batch_size=batch_size)  # pyright: ignore[reportArgumentType]


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
