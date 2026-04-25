import threading
from collections.abc import Iterable, Sized
from concurrent.futures import Future
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.pool import ApplyResult
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from blazefl.core import (
    BaseClientTrainer,
    BaseServerHandler,
    ModelSelector,
    PartitionedDataset,
    ProcessPoolClientTrainer,
    SHMHandle,
    ThreadPoolClientTrainer,
    deserialize_model,
    serialize_model,
)
from blazefl.reproducibility import (
    RNGSuite,
    create_rng_suite,
    setup_reproducibility,
)


@dataclass
class FedAvgUplinkPackage:
    """
    Data structure representing the uplink package sent from clients to the server
    in the Federated Averaging algorithm.

    Attributes:
        cid (int): Client ID.
        model_parameters (torch.Tensor): Serialized model parameters from the client.
        data_size (int): Number of data samples used in the client's training.
        metadata (dict | None): Optional metadata, such as evaluation metrics.
    """

    cid: int
    model_parameters: torch.Tensor
    data_size: int
    metadata: dict[str, float] | None = None


@dataclass
class FedAvgProcessPoolUplinkPackage:
    """
    Internal transport package used within the process pool machinery.

    Identical to FedAvgUplinkPackage except model_parameters may be an SHMHandle
    placeholder while the actual tensor lives in shared memory. Converted to
    FedAvgUplinkPackage before being exposed to the server.

    Attributes:
        cid (int): Client ID.
        model_parameters (torch.Tensor | SHMHandle): Serialized model parameters or
            a shared memory placeholder.
        data_size (int): Number of data samples used in the client's training.
        metadata (dict | None): Optional metadata, such as evaluation metrics.
    """

    cid: int
    model_parameters: torch.Tensor | SHMHandle
    data_size: int
    metadata: dict[str, float] | None = None


@dataclass
class FedAvgDownlinkPackage:
    """
    Data structure representing the downlink package sent from the server to clients
    in the Federated Averaging algorithm.

    Attributes:
        model_parameters (torch.Tensor): Serialized global model parameters to be
        distributed to clients.
    """

    model_parameters: torch.Tensor


class FedAvgPartitionType(StrEnum):
    TRAIN = "train"
    TEST = "test"


FedAvgPartitionedDataset = PartitionedDataset[FedAvgPartitionType]


class FedAvgBaseServerHandler(
    BaseServerHandler[FedAvgUplinkPackage, FedAvgDownlinkPackage],
):
    """
    Server-side handler for the Federated Averaging (FedAvg) algorithm.

    Manages the global model, coordinates client sampling, aggregates client updates,
    and controls the training process across multiple rounds.

    Attributes:
        model (torch.nn.Module): The global model being trained.
        dataset (PartitionedDataset): Dataset partitioned across clients.
        global_round (int): Total number of federated learning rounds.
        num_clients (int): Total number of clients in the federation.
        sample_ratio (float): Fraction of clients to sample in each round.
        device (str): Device to run the model on ('cpu' or 'cuda').
        client_buffer_cache (list[FedAvgUplinkPackage]): Cache for storing client
        updates before aggregation.
        num_clients_per_round (int): Number of clients sampled per round.
        round (int): Current training round.
        seed (int): Seed for reproducibility.
    """

    def __init__(
        self,
        model_selector: ModelSelector,
        model_name: str,
        dataset: FedAvgPartitionedDataset,
        global_round: int,
        num_clients: int,
        sample_ratio: float,
        device: str,
        batch_size: int,
        seed: int,
    ) -> None:
        """
        Initialize the FedAvgBaseServerHandler.

        Args:
            model_selector (ModelSelector): Selector for initializing the model.
            model_name (str): Name of the model to be used.
            dataset (PartitionedDataset): Dataset partitioned across clients.
            global_round (int): Total number of federated learning rounds.
            num_clients (int): Total number of clients in the federation.
            sample_ratio (float): Fraction of clients to sample in each round.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.model = model_selector.select_model(model_name)
        self.dataset = dataset
        self.global_round = global_round
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        self.device = device
        self.batch_size = batch_size
        self.seed = seed

        self.client_buffer_cache: list[FedAvgUplinkPackage] = []
        self.num_clients_per_round = int(self.num_clients * self.sample_ratio)
        self.round = 0

        self.rng_suite = create_rng_suite(self.seed)

    def sample_clients(self) -> list[int]:
        """
        Randomly sample a subset of clients for the current training round.

        Returns:
            list[int]: Sorted list of sampled client IDs.
        """
        sampled_clients = self.rng_suite.python.sample(
            range(self.num_clients), self.num_clients_per_round
        )

        return sorted(sampled_clients)

    def if_stop(self) -> bool:
        """
        Check if the training process should stop.

        Returns:
            bool: True if the current round exceeds or equals the total number of
            global rounds; False otherwise.
        """
        return self.round >= self.global_round

    def load(self, payload: FedAvgUplinkPackage) -> bool:
        """
        Load a client's uplink package into the server's buffer and perform a global
        update if all expected packages for the round are received.

        Args:
            payload (FedAvgUplinkPackage): Uplink package from a client.

        Returns:
            bool: True if a global update was performed; False otherwise.
        """
        self.client_buffer_cache.append(payload)

        if len(self.client_buffer_cache) == self.num_clients_per_round:
            self.global_update(self.client_buffer_cache)
            self.round += 1
            self.client_buffer_cache = []
            return True
        else:
            return False

    def global_update(self, buffer: list[FedAvgUplinkPackage]) -> None:
        """
        Aggregate client updates and update the global model parameters.

        Args:
            buffer (list[FedAvgUplinkPackage]): List of uplink packages from clients.
        """
        parameters_list = [ele.model_parameters for ele in buffer]
        weights_list = [ele.data_size for ele in buffer]
        serialized_parameters = self.aggregate(parameters_list, weights_list)
        deserialize_model(self.model, serialized_parameters)

    @staticmethod
    def aggregate(
        parameters_list: list[torch.Tensor], weights_list: list[int]
    ) -> torch.Tensor:
        """
        Aggregate model parameters from multiple clients using weighted averaging.

        Args:
            parameters_list (list[torch.Tensor]): List of serialized model parameters
            from clients.
            weights_list (list[int]): List of data sizes corresponding to each client's
            parameters.

        Returns:
            torch.Tensor: Aggregated model parameters.
        """
        total_weight = sum(weights_list)
        aggregated_parameters = parameters_list[0].clone().zero_()
        for parameters, weight in zip(parameters_list, weights_list, strict=True):
            aggregated_parameters.add_(parameters, alpha=weight / total_weight)
        return aggregated_parameters

    @staticmethod
    def evaluate(
        model: torch.nn.Module, test_loader: DataLoader, device: str
    ) -> tuple[float, float]:
        """
        Evaluate the model with the given data loader.

        Args:
            model (torch.nn.Module): The model to evaluate.
            test_loader (DataLoader): DataLoader for the evaluation data.
            device (str): Device to run the evaluation on.

        Returns:
            tuple[float, float]: Average loss and accuracy.
        """
        model.to(device)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                correct = torch.sum(predicted.eq(labels)).item()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += int(correct)
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def get_summary(self) -> dict[str, float]:
        server_loss, server_acc = FedAvgBaseServerHandler.evaluate(
            self.model,
            self.dataset.get_dataloader(
                type_=FedAvgPartitionType.TEST,
                cid=None,
                batch_size=self.batch_size,
                generator=self.rng_suite.torch_cpu,
            ),
            self.device,
        )
        return {
            "server_acc": server_acc,
            "server_loss": server_loss,
        }

    def downlink_package(self) -> FedAvgDownlinkPackage:
        """
        Create a downlink package containing the current global model parameters to
        send to clients.

        Returns:
            FedAvgDownlinkPackage: Downlink package with serialized model parameters.
        """
        model_parameters = serialize_model(self.model)
        return FedAvgDownlinkPackage(model_parameters)


class FedAvgBaseClientTrainer(
    BaseClientTrainer[FedAvgUplinkPackage, FedAvgDownlinkPackage]
):
    """
    Base client trainer for the Federated Averaging (FedAvg) algorithm.

    This trainer processes clients sequentially, training and evaluating a local model
    for each client based on the server-provided model parameters.

    Attributes:
        model (torch.nn.Module): The client's local model.
        dataset (PartitionedDataset): Dataset partitioned across clients.
        device (str): Device to run the model on ('cpu' or 'cuda').
        num_clients (int): Total number of clients in the federation.
        epochs (int): Number of local training epochs per client.
        batch_size (int): Batch size for local training.
        lr (float): Learning rate for the optimizer.
        cache (list[FedAvgUplinkPackage]): Cache to store uplink packages for the
        server.
    """

    def __init__(
        self,
        model_selector: ModelSelector,
        model_name: str,
        dataset: FedAvgPartitionedDataset,
        device: str,
        num_clients: int,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
    ) -> None:
        """
        Initialize the FedAvgBaseClientTrainer.

        Args:
            model_selector (ModelSelector): Selector for initializing the local model.
            model_name (str): Name of the model to be used.
            dataset (PartitionedDataset): Dataset partitioned across clients.
            device (str): Device to run the model on ('cpu' or 'cuda').
            num_clients (int): Total number of clients in the federation.
            epochs (int): Number of local training epochs per client.
            batch_size (int): Batch size for local training.
            lr (float): Learning rate for the optimizer.
            seed (int): Seed for reproducibility.
        """
        self.model = model_selector.select_model(model_name)
        self.dataset = dataset
        self.device = device
        self.num_clients = num_clients
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.cache: list[FedAvgUplinkPackage] = []

        self.rng_suite = create_rng_suite(self.seed)

    def local_process(
        self, payload: FedAvgDownlinkPackage, cid_list: list[int]
    ) -> None:
        """
        Train and evaluate the model for each client in the given list.

        Args:
            payload (FedAvgDownlinkPackage): Downlink package with global model
            parameters.
            cid_list (list[int]): List of client IDs to process.

        Returns:
            None
        """
        model_parameters = payload.model_parameters
        for cid in tqdm(cid_list, desc="Client", leave=False):
            data_loader = self.dataset.get_dataloader(
                type_=FedAvgPartitionType.TRAIN,
                cid=cid,
                batch_size=self.batch_size,
                generator=self.rng_suite.torch_cpu,
            )
            pack = self.train(model_parameters, data_loader, cid)
            self.cache.append(pack)

    def train(
        self,
        model_parameters: torch.Tensor,
        train_loader: DataLoader,
        cid: int,
    ) -> FedAvgUplinkPackage:
        """
        Train the local model on the given training data loader.

        Args:
            model_parameters (torch.Tensor): Global model parameters to initialize the
            local model.
            train_loader (DataLoader): DataLoader for the training data.

        Returns:
            FedAvgUplinkPackage: Uplink package containing updated model parameters and
            data size.
        """
        deserialize_model(self.model, model_parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            for data, target in train_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        assert isinstance(train_loader.dataset, Sized)
        data_size = len(train_loader.dataset)
        model_parameters = serialize_model(self.model)

        return FedAvgUplinkPackage(cid, model_parameters, data_size)

    def uplink_package(self) -> list[FedAvgUplinkPackage]:
        """
        Retrieve the uplink packages for transmission to the server.

        Returns:
            list[FedAvgUplinkPackage]: A list of uplink packages.
        """
        package = self.cache
        self.cache = []
        return package


@dataclass
class FedAvgClientConfig:
    """
    Data structure representing shared data for parallel client training
    in the Federated Averaging (FedAvg) algorithm.

    This structure is used to store all necessary information for each client
    to perform local training in a parallelized setting.

    Attributes:
        model_selector (ModelSelector): Selector for initializing the local model.
        model_name (str): Name of the model to be used.
        dataset (PartitionedDataset): Dataset partitioned across clients.
        epochs (int): Number of local training epochs per client.
        batch_size (int): Batch size for local training.
        lr (float): Learning rate for the optimizer.
        cid (int): Client ID.
        seed (int): Seed for reproducibility.
        state_path (Path): Path to save the client's random state.
    """

    model_selector: ModelSelector
    model_name: str
    dataset: FedAvgPartitionedDataset
    epochs: int
    batch_size: int
    lr: float
    cid: int
    seed: int
    state_path: Path


def _fedavg_train(
    model: torch.nn.Module,
    model_parameters: torch.Tensor,
    train_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    stop_event: threading.Event,
) -> tuple[torch.Tensor, int]:
    model.to(device)
    deserialize_model(model, model_parameters)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        if stop_event.is_set():
            break
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    assert isinstance(train_loader.dataset, Sized)
    data_size = len(train_loader.dataset)
    serialized_parameters = serialize_model(model)
    return serialized_parameters, data_size


class FedAvgProcessPoolClientTrainer(
    ProcessPoolClientTrainer[
        FedAvgUplinkPackage,
        FedAvgDownlinkPackage,
        FedAvgClientConfig,
        FedAvgProcessPoolUplinkPackage,
    ]
):
    """
    Parallel client trainer for the Federated Averaging (FedAvg) algorithm.

    This trainer handles the parallelized training and evaluation of local models
    across multiple clients, distributing tasks to different processes or devices.

    Attributes:
        model_selector (ModelSelector): Selector for initializing the local model.
        model_name (str): Name of the model to be used.
        state_dir (Path): Directory to save random states for reproducibility.
        dataset (PartitionedDataset): Dataset partitioned across clients.
        device (str): Device to run the models on ('cpu' or 'cuda').
        num_clients (int): Total number of clients in the federation.
        epochs (int): Number of local training epochs per client.
        batch_size (int): Batch size for local training.
        lr (float): Learning rate for the optimizer.
        seed (int): Seed for reproducibility.
        num_parallels (int): Number of parallel processes for training.
        device_count (int): Number of CUDA devices available (0 if not using GPU).
    """

    def __init__(
        self,
        model_selector: ModelSelector,
        model_name: str,
        state_dir: Path,
        dataset: FedAvgPartitionedDataset,
        device: str,
        num_clients: int,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
        num_parallels: int,
    ) -> None:
        """
        Initialize the FedAvgParalleClientTrainer.

        Args:
            model_selector (ModelSelector): Selector for initializing the local model.
            model_name (str): Name of the model to be used.
            state_dir (Path): Directory to save random states for reproducibility.
            dataset (PartitionedDataset): Dataset partitioned across clients.
            device (str): Device to run the models on ('cpu' or 'cuda').
            num_clients (int): Total number of clients in the federation.
            epochs (int): Number of local training epochs per client.
            batch_size (int): Batch size for local training.
            lr (float): Learning rate for the optimizer.
            seed (int): Seed for reproducibility.
            num_parallels (int): Number of parallel processes for training.
        """
        self.num_parallels = num_parallels
        self.device = device
        self.device_count = torch.cuda.device_count()
        self.cache: list[FedAvgUplinkPackage] = []

        self.model_selector = model_selector
        self.model_name = model_name
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.num_clients = num_clients
        self.seed = seed
        self.manager = mp.Manager()
        self.stop_event = self.manager.Event()

        self.model_parameters_buffer = serialize_model(
            self.model_selector.select_model(model_name)
        )

    def progress_fn(
        self,
        it: list[ApplyResult],
    ) -> Iterable[ApplyResult]:
        return tqdm(it, desc="Client", leave=False)

    def prepare_uplink_package_buffer(self) -> FedAvgProcessPoolUplinkPackage:
        return FedAvgProcessPoolUplinkPackage(
            cid=-1,
            model_parameters=self.model_parameters_buffer.clone(),
            data_size=0,
        )

    def convert_buffer_to_uplink(
        self, buffer: FedAvgProcessPoolUplinkPackage
    ) -> FedAvgUplinkPackage:
        assert isinstance(buffer.model_parameters, torch.Tensor)
        return FedAvgUplinkPackage(
            cid=buffer.cid,
            model_parameters=buffer.model_parameters,
            data_size=buffer.data_size,
            metadata=buffer.metadata,
        )

    @staticmethod
    def worker(
        config: FedAvgClientConfig,
        payload: FedAvgDownlinkPackage,
        device: str,
        stop_event: threading.Event,
        *,
        shm_buffer: FedAvgProcessPoolUplinkPackage | None = None,
    ) -> FedAvgProcessPoolUplinkPackage:
        """
        Process a single client's local training and evaluation.

        This method is executed by a worker process and handles loading client
        configuration and payload, performing the client-specific training,
        and returning the result.

        Args:
            config (FedAvgClientConfig):
                The client's configuration data.
            payload (FedAvgDownlinkPackage):
                The downlink payload from the server.
            device (str): Device to use for processing (e.g., "cpu", "cuda:0").
            shm_buffer (FedAvgProcessPoolUplinkPackage | None):
                Optional shared memory buffer for the uplink package.

        Returns:
            FedAvgUplinkPackage:
                The uplink package containing the client's results.
        """

        setup_reproducibility(config.seed)
        if config.state_path.exists():
            state = torch.load(config.state_path, weights_only=False)
            if not isinstance(state, RNGSuite):
                raise TypeError(
                    f"Expected RNGSuite in {config.state_path}, got {type(state)}"
                )
        else:
            state = create_rng_suite(config.seed)

        model = config.model_selector.select_model(config.model_name)
        train_loader = config.dataset.get_dataloader(
            type_=FedAvgPartitionType.TRAIN,
            cid=config.cid,
            batch_size=config.batch_size,
            generator=state.torch_cpu,
        )
        package = FedAvgProcessPoolClientTrainer.train(
            model=model,
            model_parameters=payload.model_parameters,
            train_loader=train_loader,
            device=device,
            epochs=config.epochs,
            lr=config.lr,
            stop_event=stop_event,
            cid=config.cid,
        )
        torch.save(state, config.state_path)
        config.dataset.set_dataset(
            type_=FedAvgPartitionType.TRAIN,
            cid=config.cid,
            dataset=train_loader.dataset,
        )

        if shm_buffer is None or not isinstance(
            shm_buffer.model_parameters, torch.Tensor
        ):
            raise RuntimeError(
                "shm_buffer must be a pre-allocated shared memory package"
            )
        if not isinstance(package.model_parameters, torch.Tensor):
            raise RuntimeError(
                "package.model_parameters must be a Tensor before SHM copy"
            )
        shm_buffer.model_parameters.copy_(package.model_parameters)
        package.model_parameters = SHMHandle()
        return package

    @staticmethod
    def train(
        model: torch.nn.Module,
        model_parameters: torch.Tensor,
        train_loader: DataLoader,
        device: str,
        epochs: int,
        lr: float,
        stop_event: threading.Event,
        cid: int,
    ) -> FedAvgProcessPoolUplinkPackage:
        serialized_parameters, data_size = _fedavg_train(
            model, model_parameters, train_loader, device, epochs, lr, stop_event
        )
        return FedAvgProcessPoolUplinkPackage(cid, serialized_parameters, data_size)

    def get_client_config(self, cid: int) -> FedAvgClientConfig:
        """
        Generate the client configuration for a specific client.

        Args:
            cid (int): Client ID.

        Returns:
            FedAvgClientConfig: Client configuration data structure.
        """
        data = FedAvgClientConfig(
            model_selector=self.model_selector,
            model_name=self.model_name,
            dataset=self.dataset,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            cid=cid,
            seed=self.seed + cid,
            state_path=self.state_dir.joinpath(f"{cid}.pt"),
        )
        return data

    def uplink_package(self) -> list[FedAvgUplinkPackage]:
        """
        Retrieve the uplink packages for transmission to the server.

        Returns:
            list[FedAvgUplinkPackage]: A list of uplink packages.
        """
        package = self.cache
        self.cache = []
        return package


class FedAvgThreadPoolClientTrainer(
    ThreadPoolClientTrainer[
        FedAvgUplinkPackage,
        FedAvgDownlinkPackage,
    ]
):
    def __init__(
        self,
        model_selector: ModelSelector,
        model_name: str,
        dataset: FedAvgPartitionedDataset,
        device: str,
        num_clients: int,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
        num_parallels: int,
    ) -> None:
        self.num_parallels = num_parallels
        self.device = device
        self.device_count = torch.cuda.device_count()
        self.cache: list[FedAvgUplinkPackage] = []

        self.model_selector = model_selector
        self.model_name = model_name
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_clients = num_clients
        self.seed = seed
        self.stop_event = threading.Event()
        self.rng_suite_list = [
            create_rng_suite(self.seed + i) for i in range(num_clients)
        ]

    def progress_fn(
        self, it: list[Future[FedAvgUplinkPackage]]
    ) -> Iterable[Future[FedAvgUplinkPackage]]:
        return tqdm(it, desc="Client", leave=False)

    def worker(
        self,
        cid: int,
        device: str,
        payload: FedAvgDownlinkPackage,
        stop_event: threading.Event,
    ) -> FedAvgUplinkPackage:
        model = self.model_selector.select_model(self.model_name)
        train_loader = self.dataset.get_dataloader(
            type_=FedAvgPartitionType.TRAIN,
            cid=cid,
            batch_size=self.batch_size,
            generator=self.rng_suite_list[cid].torch_cpu,
        )
        package = self.train(
            model=model,
            model_parameters=payload.model_parameters,
            train_loader=train_loader,
            device=device,
            epochs=self.epochs,
            lr=self.lr,
            stop_event=stop_event,
            cid=cid,
        )
        return package

    def train(
        self,
        model: torch.nn.Module,
        model_parameters: torch.Tensor,
        train_loader: DataLoader,
        device: str,
        epochs: int,
        lr: float,
        stop_event: threading.Event,
        cid: int,
    ) -> FedAvgUplinkPackage:
        serialized_parameters, data_size = _fedavg_train(
            model, model_parameters, train_loader, device, epochs, lr, stop_event
        )
        return FedAvgUplinkPackage(cid, serialized_parameters, data_size)

    def uplink_package(self) -> list[FedAvgUplinkPackage]:
        package = self.cache
        self.cache = []
        return package
