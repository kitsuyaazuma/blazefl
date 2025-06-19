from enum import StrEnum
from torch.utils.data import DataLoader, Dataset
from typing import Protocol, TypeVar

PartitionType = TypeVar('PartitionType', bound=StrEnum, contravariant=True)

class PartitionedDataset(Protocol[PartitionType]):
    def get_dataset(self, type_: PartitionType, cid: int | None) -> Dataset: ...
    def get_dataloader(self, type_: PartitionType, cid: int | None, batch_size: int | None) -> DataLoader: ...
