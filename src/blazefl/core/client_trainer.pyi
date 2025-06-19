import threading
from blazefl.utils import move_tensor_to_shared_memory as move_tensor_to_shared_memory
from multiprocessing.pool import ApplyResult as ApplyResult
from pathlib import Path
from typing import Literal, Protocol, TypeVar

UplinkPackage = TypeVar('UplinkPackage')
DownlinkPackage = TypeVar('DownlinkPackage', contravariant=True)

class BaseClientTrainer(Protocol[UplinkPackage, DownlinkPackage]):
    def uplink_package(self) -> list[UplinkPackage]: ...
    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None: ...
ClientConfig = TypeVar('ClientConfig')

class ProcessPoolClientTrainer(BaseClientTrainer[UplinkPackage, DownlinkPackage], Protocol[UplinkPackage, DownlinkPackage, ClientConfig]):
    num_parallels: int
    share_dir: Path
    device: str
    device_count: int
    cache: list[UplinkPackage]
    ipc_mode: Literal['storage', 'shared_memory']
    stop_event: threading.Event
    def get_client_config(self, cid: int) -> ClientConfig: ...
    def get_client_device(self, cid: int) -> str: ...
    @staticmethod
    def worker(config: ClientConfig | Path, payload: DownlinkPackage | Path, device: str, stop_event: threading.Event) -> UplinkPackage | Path: ...
    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None: ...

class ThreadPoolClientTrainer(BaseClientTrainer[UplinkPackage, DownlinkPackage], Protocol[UplinkPackage, DownlinkPackage]):
    num_parallels: int
    device: str
    device_count: int
    cache: list[UplinkPackage]
    stop_event: threading.Event
    def worker(self, cid: int, device: str, payload: DownlinkPackage, stop_event: threading.Event) -> UplinkPackage: ...
    def get_client_device(self, cid: int) -> str: ...
    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None: ...
