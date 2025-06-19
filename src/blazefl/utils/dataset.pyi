from _typeshed import Incomplete
from collections.abc import Callable
from torch.utils.data import Dataset

class FilteredDataset(Dataset):
    data: Incomplete
    targets: Incomplete
    transform: Incomplete
    target_transform: Incomplete
    def __init__(self, indices: list[int], original_data: list, original_targets: list | None = None, transform: Callable | None = None, target_transform: Callable | None = None) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple: ...
