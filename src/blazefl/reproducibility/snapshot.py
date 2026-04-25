import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from blazefl.reproducibility.generator import setup_reproducibility


def seed_everything(seed: int) -> None:
    """
    Seeds the global random number generators for all relevant libraries.

    This function sets a single seed for Python's `random` module, NumPy, and PyTorch
    to ensure that results are consistent across runs. It directly manipulates the
    global state of these libraries.

    Args:
        seed: The integer value for the seed.
    """
    setup_reproducibility(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@dataclass
class RandomStateSnapshot:
    """
    A snapshot of the global random state for major libraries.

    This class provides a mechanism to capture the exact state of the global random
    number generators and later restore it. This is useful for scenarios that require
    resuming a process from a specific point in time with the identical sequence of
    random numbers.

    Attributes:
        environ: The value of the `PYTHONHASHSEED` environment variable.
        python: The internal state of Python's `random` module.
        numpy: The internal state of NumPy's legacy random number generator.
        torch_cpu: The RNG state of the PyTorch CPU generator.
        torch_cuda: The RNG state of the PyTorch CUDA generator, if available.
    """

    environ: str
    python: tuple[Any, ...]
    numpy: tuple[str, npt.NDArray[np.uint32], int, int, float]
    torch_cpu: torch.Tensor
    torch_cuda: torch.Tensor | None

    @classmethod
    def capture(cls) -> RandomStateSnapshot:
        """
        Captures the current global random state.

        Returns:
            A `RandomStateSnapshot` instance containing the captured states.
        """
        _environ = os.environ["PYTHONHASHSEED"]
        _python = random.getstate()
        _numpy = np.random.get_state(legacy=True)
        assert isinstance(_numpy, tuple)
        _torch_cpu = torch.get_rng_state()

        snapshot = cls(
            environ=_environ,
            python=_python,
            numpy=_numpy,
            torch_cpu=_torch_cpu,
            torch_cuda=None,
        )
        if torch.cuda.is_available():
            snapshot.torch_cuda = torch.cuda.get_rng_state()
        return snapshot

    @staticmethod
    def restore(snapshot: RandomStateSnapshot) -> None:
        """
        Restores the global random state from a snapshot object.

        Args:
            snapshot: The `RandomStateSnapshot` to restore from.
        """
        os.environ["PYTHONHASHSEED"] = snapshot.environ
        random.setstate(snapshot.python)
        np.random.set_state(snapshot.numpy)
        torch.set_rng_state(snapshot.torch_cpu)
        if snapshot.torch_cuda is not None:
            torch.cuda.set_rng_state(snapshot.torch_cuda)
