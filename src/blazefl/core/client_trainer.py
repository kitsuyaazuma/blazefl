import signal
import threading
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from multiprocessing.pool import ApplyResult
from typing import Protocol, TypeVar

from blazefl.core.utils import process_tensors_in_object, reconstruct_from_shared_memory

UplinkPackage = TypeVar("UplinkPackage")
DownlinkPackage = TypeVar("DownlinkPackage", contravariant=True)


class BaseClientTrainer(Protocol[UplinkPackage, DownlinkPackage]):
    """
    Abstract base class for serial client training in federated learning.

    This class defines the interface for training clients in a serial manner,
    where each client is processed one after the other.

    Raises:
        NotImplementedError: If the methods are not implemented in a subclass.
    """

    def uplink_package(self) -> list[UplinkPackage]:
        """
        Prepare the data package to be sent from the client to the server.

        Returns:
            list[UplinkPackage]: A list of data packages prepared for uplink
            transmission.
        """
        ...

    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None:
        """
        Process the downlink payload from the server for a list of client IDs.

        Args:
            payload (DownlinkPackage): The data package received from the server.
            cid_list (list[int]): A list of client IDs to process.

        Returns:
            None
        """
        ...


ClientConfig = TypeVar("ClientConfig")
BufferPackage = TypeVar("BufferPackage")


class ProcessPoolClientTrainer(
    BaseClientTrainer[UplinkPackage, DownlinkPackage],
    Protocol[UplinkPackage, DownlinkPackage, ClientConfig, BufferPackage],
):
    """
    Abstract base class for parallel client training using a process pool.

    This class enables parallel processing of clients by distributing tasks across
    multiple processes.

    ``BufferPackage`` is the internal transport type used between worker processes and
    the parent. It may differ from ``UplinkPackage`` when workers use shared memory
    placeholders (e.g. ``SHMHandle``). The conversion is handled by
    ``convert_buffer_to_uplink``, which is called inside ``local_process`` after
    reconstruction from shared memory.

    Attributes:
        num_parallels (int): Number of parallel processes to use.
        device (str): Primary device for computation (e.g., "cpu", "cuda").
        device_count (int): Number of available CUDA devices for distribution.
        cache (list[UplinkPackage]): Cache to store results from clients.
        stop_event (threading.Event): Event to signal workers to stop.

    Raises:
        NotImplementedError: If the abstract methods are not implemented in a subclass.
    """

    num_parallels: int
    device: str
    device_count: int
    cache: list[UplinkPackage]
    stop_event: threading.Event

    def progress_fn(
        self,
        it: list[ApplyResult],
    ) -> Iterable[ApplyResult]:
        """
        A no-op progress function that can be overridden to provide custom
        progress tracking.

        Args:
            it (list[ApplyResult]): A list of ApplyResult objects.

        Returns:
            Iterable[ApplyResult]: The original iterable.
        """
        return it

    def get_client_config(self, cid: int) -> ClientConfig:
        """
        Retrieve the configuration for a given client ID.

        Args:
            cid (int): Client ID.

        Returns:
            ClientConfig: The configuration for the specified client.
        """
        ...

    def get_client_device(self, cid: int) -> str:
        """
        Retrieve the device to use for processing a given client.

        Args:
            cid (int): Client ID.

        Returns:
            str: The device to use for processing the client.
        """
        if self.device == "cuda":
            return f"cuda:{cid % self.device_count}"
        return self.device

    @staticmethod
    def worker(
        config: ClientConfig,
        payload: DownlinkPackage,
        device: str,
        stop_event: threading.Event,
        *,
        shm_buffer: BufferPackage | None = None,
    ) -> BufferPackage:
        """
        Process a single client's training task.

        This method is executed by each worker process in the pool.
        It handles loading client configuration and payload, performing
        the client-specific operations, and returning the result.

        Args:
            config (ClientConfig):
                The client's configuration data.
            payload (DownlinkPackage):
                The downlink payload from the server
            device (str): Device to use for processing (e.g., "cpu", "cuda:0").
            stop_event (threading.Event): Event to signal stopping the worker.
            shm_buffer (BufferPackage | None):
                Optional shared memory buffer for the uplink package.

        Returns:
            BufferPackage:
                The transport package containing the client's results.
        """
        ...

    def prepare_uplink_package_buffer(self) -> BufferPackage:
        """
        Allocate a pre-initialized shared memory buffer for a single client's result.

        Returns:
            BufferPackage: A buffer object whose tensors are in shared memory.
        """
        raise NotImplementedError

    def convert_buffer_to_uplink(self, buffer: BufferPackage) -> UplinkPackage:
        """
        Convert a reconstructed ``BufferPackage`` to an ``UplinkPackage``.

        Called by ``local_process`` after shared memory reconstruction. When
        ``BufferPackage`` and ``UplinkPackage`` are the same type, implement this
        as ``return buffer``.

        Args:
            buffer (BufferPackage): The reconstructed buffer from shared memory.

        Returns:
            UplinkPackage: The uplink package to be stored in ``cache``.
        """
        raise NotImplementedError

    def shutdown(self) -> None:
        """
        Shut down process-shared coordination resources owned by the trainer.

        Subclasses that create a ``multiprocessing.Manager`` should store it on
        ``self.manager`` so this method can shut it down explicitly.
        """
        manager = getattr(self, "manager", None)
        if manager is None:
            return

        shutdown = getattr(manager, "shutdown", None)
        if shutdown is None:
            return

        shutdown()
        with suppress(AttributeError):
            delattr(self, "manager")

    def __del__(self) -> None:
        with suppress(Exception):
            self.shutdown()

    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None:
        """
        Manage the parallel processing of clients.

        This method distributes the processing of multiple clients across
        parallel processes, handling data saving, loading, and caching.

        Args:
            payload (DownlinkPackage): The data package received from the server.
            cid_list (list[int]): A list of client IDs to process.

        Returns:
            None
        """
        import torch.multiprocessing as mp

        shm_buffers = {}
        process_tensors_in_object(payload, mode="move")
        for cid in cid_list:
            buffer = self.prepare_uplink_package_buffer()
            process_tensors_in_object(buffer, mode="move")
            shm_buffers[cid] = buffer

        self.stop_event.clear()
        pool = mp.Pool(
            processes=self.num_parallels,
            initializer=signal.signal,
            initargs=(signal.SIGINT, signal.SIG_IGN),
        )
        should_terminate_pool = True
        try:
            jobs: list[ApplyResult] = []
            for cid in cid_list:
                config = self.get_client_config(cid)
                device = self.get_client_device(cid)
                jobs.append(
                    pool.apply_async(
                        self.worker,
                        (
                            config,
                            payload,
                            device,
                            self.stop_event,
                        ),
                        kwds={
                            "shm_buffer": shm_buffers.get(cid),
                        },
                    ),
                )

            for i, job in enumerate(self.progress_fn(jobs)):
                result = job.get()
                cid = cid_list[i]
                buffer = reconstruct_from_shared_memory(result, shm_buffers[cid])
                self.cache.append(self.convert_buffer_to_uplink(buffer))
            should_terminate_pool = False
        finally:
            self.stop_event.set()
            if should_terminate_pool:
                pool.terminate()
            else:
                pool.close()
            pool.join()


class ThreadPoolClientTrainer(
    BaseClientTrainer[UplinkPackage, DownlinkPackage],
    Protocol[UplinkPackage, DownlinkPackage],
):
    """
    Abstract base class for parallel client training using a thread pool.

    This class enables parallel processing of clients within a processes.

    Attributes:
        num_parallels (int): Number of parallel threads to use.
        device (str): Primary device for computation (e.g., "cpu", "cuda").
        device_count (int): Number of available CUDA devices for distribution.
        cache (list[UplinkPackage]): Cache to store results from clients.
        stop_event (threading.Event): Event to signal workers to stop.

    Raises:
        NotImplementedError: If the abstract methods are not implemented in a subclass.
    """

    num_parallels: int
    device: str
    device_count: int
    cache: list[UplinkPackage]
    stop_event: threading.Event

    def progress_fn(
        self, it: list[Future[UplinkPackage]]
    ) -> Iterable[Future[UplinkPackage]]:
        """
        A no-op progress function that can be overridden to provide custom
        progress tracking.

        Args:
            it (list[Future[UplinkPackage]]): A list of Future objects
                representing the results of client processing.

        Returns:
            Iterable[Future[UplinkPackage]]: The original iterable.
        """
        return it

    def worker(
        self,
        cid: int,
        device: str,
        payload: DownlinkPackage,
        stop_event: threading.Event,
    ) -> UplinkPackage:
        """
        Process a single client's training task in a thread.

        Args:
            cid (int): The client ID.
            device (str): The device to use for processing this client.
            payload (DownlinkPackage): The data package received from the server.
            stop_event (threading.Event): Event to signal stopping the worker.

        Returns:
            UplinkPackage: The uplink package containing the client's results.
        """
        ...

    def get_client_device(self, cid: int) -> str:
        """
        Retrieve the device to use for processing a given client.

        Args:
            cid (int): Client ID.

        Returns:
            str: The device to use for processing the client.
        """
        if self.device == "cuda":
            return f"cuda:{cid % self.device_count}"
        return self.device

    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None:
        """
        Manage the parallel processing of clients using threads.

        This method distributes the processing of multiple clients across
        a pool of threads.

        Args:
            payload (DownlinkPackage): The data package received from the server.
            cid_list (list[int]): A list of client IDs to process.
        """
        self.stop_event.clear()
        executor = ThreadPoolExecutor(max_workers=self.num_parallels)
        try:
            futures: list[Future[UplinkPackage]] = []
            for cid in cid_list:
                device = self.get_client_device(cid)
                future = executor.submit(
                    self.worker,
                    cid,
                    device,
                    payload,
                    self.stop_event,
                )
                futures.append(future)

            for future in self.progress_fn(futures):
                result = future.result()
                self.cache.append(result)
        finally:
            self.stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
