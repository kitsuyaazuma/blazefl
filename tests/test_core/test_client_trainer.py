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
    ProcessPoolClientTrainer[UplinkPackage, DownlinkPackage, ClientConfig]
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
