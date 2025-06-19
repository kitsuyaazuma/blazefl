import torch
from dataclasses import dataclass

def seed_everything(seed: int, device: str) -> None: ...

@dataclass
class CUDARandomState:
    manual_seed: int
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cuda_rng_state: torch.Tensor

@dataclass
class RandomState:
    @classmethod
    def get_random_state(cls, device: str) -> RandomState: ...
    @staticmethod
    def set_random_state(random_state: RandomState) -> None: ...
