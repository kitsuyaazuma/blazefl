import threading
from dataclasses import dataclass

import pytest
import torch
import torch.multiprocessing as mp

from blazefl.core.utils import SHMHandle
from src.blazefl.core import ProcessPoolClientTrainer


@dataclass
class UplinkPackage:
    cid: int
    message: str
    tensor: torch.Tensor | SHMHandle


@dataclass
class DownlinkPackage:
    message: str


@dataclass
class ClientConfig:
    cid: int


class DummyProcessPoolClientTrainer(
    ProcessPoolClientTrainer[
        UplinkPackage, DownlinkPackage, ClientConfig, UplinkPackage
    ]
):
    def __init__(
        self,
        num_parallels: int,
        device: str,
    ):
        self.num_parallels = num_parallels
        self.device = device
        self.device_count = torch.cuda.device_count()
        self.cache: list[UplinkPackage] = []
        self.manager = mp.Manager()
        self.stop_event = self.manager.Event()

    def uplink_package(self) -> list[UplinkPackage]:
        return self.cache

    def get_client_config(self, cid: int) -> ClientConfig:
        return ClientConfig(cid=cid)

    def prepare_uplink_package_buffer(self) -> UplinkPackage:
        return UplinkPackage(cid=-1, message="", tensor=torch.zeros(1))

    def convert_buffer_to_uplink(self, buffer: UplinkPackage) -> UplinkPackage:
        return buffer

    @staticmethod
    def worker(
        config: ClientConfig,
        payload: DownlinkPackage,
        device: str,
        stop_event: threading.Event,
        *,
        shm_buffer: UplinkPackage | None = None,
    ) -> UplinkPackage:
        _ = stop_event
        _ = device
        dummy_uplink_package = UplinkPackage(
            cid=config.cid,
            tensor=torch.rand(1),
            message=payload.message + "<client_to_server>",
        )

        assert shm_buffer is not None
        shm_buffer.tensor = dummy_uplink_package.tensor
        dummy_uplink_package.tensor = SHMHandle()
        return dummy_uplink_package


class DummyManager:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class ShutdownAwareProcessPoolClientTrainer(
    ProcessPoolClientTrainer[
        UplinkPackage, DownlinkPackage, ClientConfig, UplinkPackage
    ]
):
    def __init__(self) -> None:
        self.num_parallels = 1
        self.device = "cpu"
        self.device_count = 0
        self.cache: list[UplinkPackage] = []
        self.stop_event = threading.Event()
        self.manager = DummyManager()

    def uplink_package(self) -> list[UplinkPackage]:
        return self.cache

    def get_client_config(self, cid: int) -> ClientConfig:
        return ClientConfig(cid=cid)

    def prepare_uplink_package_buffer(self) -> UplinkPackage:
        return UplinkPackage(cid=-1, message="", tensor=torch.zeros(1))

    def convert_buffer_to_uplink(self, buffer: UplinkPackage) -> UplinkPackage:
        return buffer

    @staticmethod
    def worker(
        config: ClientConfig,
        payload: DownlinkPackage,
        device: str,
        stop_event: threading.Event,
        *,
        shm_buffer: UplinkPackage | None = None,
    ) -> UplinkPackage:
        raise NotImplementedError


class InterruptingProcessPoolClientTrainer(ShutdownAwareProcessPoolClientTrainer):
    def progress_fn(self, it: list) -> list:
        raise KeyboardInterrupt


class FakePool:
    def __init__(self) -> None:
        self.close_calls = 0
        self.terminate_calls = 0
        self.join_calls = 0

    def apply_async(self, *args, **kwargs) -> object:
        return object()

    def close(self) -> None:
        self.close_calls += 1

    def terminate(self) -> None:
        self.terminate_calls += 1

    def join(self) -> None:
        self.join_calls += 1


@pytest.mark.parametrize("num_parallels", [1, 2, 4])
@pytest.mark.parametrize("cid_list", [[], [42], [0, 1, 2]])
def test_process_pool_client_trainer(num_parallels: int, cid_list: list[int]) -> None:
    trainer = DummyProcessPoolClientTrainer(
        num_parallels=num_parallels,
        device="cpu",
    )

    dummy_payload = DownlinkPackage(message="<server_to_client>")

    trainer.local_process(dummy_payload, cid_list)

    assert len(trainer.cache) == len(cid_list)
    for i, cid in enumerate(cid_list):
        result = trainer.cache[i]
        assert result.cid == cid
        assert result.message == "<server_to_client><client_to_server>"

    package = trainer.uplink_package()
    assert len(package) == len(cid_list)

    for i, cid in enumerate(cid_list):
        result = package[i]
        assert result.cid == cid
        assert result.message == "<server_to_client><client_to_server>"

    trainer.shutdown()


def test_process_pool_client_trainer_shutdown_is_idempotent() -> None:
    trainer = ShutdownAwareProcessPoolClientTrainer()

    trainer.shutdown()
    assert not hasattr(trainer, "manager")

    trainer.shutdown()


def test_process_pool_client_trainer_terminates_pool_on_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = InterruptingProcessPoolClientTrainer()
    fake_pool = FakePool()

    monkeypatch.setattr(mp, "Pool", lambda *args, **kwargs: fake_pool)

    with pytest.raises(KeyboardInterrupt):
        trainer.local_process(
            DownlinkPackage(message="<server_to_client>"),
            [0, 1],
        )

    assert fake_pool.terminate_calls == 1
    assert fake_pool.close_calls == 0
    assert fake_pool.join_calls == 1
    assert trainer.stop_event.is_set()
    trainer.shutdown()
