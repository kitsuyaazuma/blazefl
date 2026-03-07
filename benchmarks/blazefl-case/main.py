import logging
import shutil
import sys
import time
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import torch
import torch.multiprocessing as mp
import typer
from blazefl.contrib import (
    FedAvgBaseClientTrainer,
    FedAvgBaseServerHandler,
    FedAvgProcessPoolClientTrainer,
    FedAvgThreadPoolClientTrainer,
)
from blazefl.reproducibility import setup_reproducibility
from dataset import PartitionedCIFAR10
from models import FedAvgModelName, FedAvgModelSelector


class FedAvgBenchmarkPipeline:
    def __init__(
        self,
        handler: FedAvgBaseServerHandler,
        trainer: FedAvgBaseClientTrainer
        | FedAvgProcessPoolClientTrainer
        | FedAvgThreadPoolClientTrainer,
    ) -> None:
        self.handler = handler
        self.trainer = trainer

    def main(self):
        start_time = time.perf_counter()
        while not self.handler.if_stop():
            round_ = self.handler.round
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package()

            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package()

            # server side
            for pack in uploads:
                self.handler.load(pack)

            summary = self.handler.get_summary()
            formatted_summary = ", ".join(f"{k}: {v:.3f}" for k, v in summary.items())
            logging.info(f"round: {round_}, {formatted_summary}")

        end_time = time.perf_counter()
        logging.info(f"BENCHMARK_RESULT_TIME: {end_time - start_time:.4f}")


class ExecutionMode(StrEnum):
    SINGLE_THREADED = "SINGLE_THREADED"
    MULTI_PROCESS = "MULTI_PROCESS"
    MULTI_THREADED = "MULTI_THREADED"


EXECUTION_MODE = ExecutionMode.MULTI_THREADED


def main(
    model_name: FedAvgModelName = FedAvgModelName.CNN,
    num_clients: int = 100,
    global_round: int = 5,
    sample_ratio: float = 1.0,
    partition: str = "shards",
    num_shards: int = 200,
    dir_alpha: float = 1.0,
    seed: int = 42,
    epochs: int = 5,
    lr: float = 0.1,
    batch_size: int = 50,
    num_parallels: int = 10,
    dataset_root_dir: Path = Path("/tmp/blazefl-case/dataset"),
    state_dir_base: Path = Path("/tmp/blazefl-case/state"),
    execution_mode: ExecutionMode = EXECUTION_MODE,
):
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_split_dir = dataset_root_dir / timestamp
    state_dir = state_dir_base / timestamp

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    setup_reproducibility(seed)

    dataset = PartitionedCIFAR10(
        root=dataset_root_dir,
        path=dataset_split_dir,
        num_clients=num_clients,
        seed=seed,
        partition=partition,
        num_shards=num_shards,
        dir_alpha=dir_alpha,
    )
    model_selector = FedAvgModelSelector(num_classes=10, seed=seed)

    handler = FedAvgBaseServerHandler(
        model_selector=model_selector,
        model_name=model_name,
        dataset=dataset,
        global_round=global_round,
        num_clients=num_clients,
        device=device,
        sample_ratio=sample_ratio,
        batch_size=batch_size,
        seed=seed,
    )
    trainer: (
        FedAvgBaseClientTrainer
        | FedAvgProcessPoolClientTrainer
        | FedAvgThreadPoolClientTrainer
        | None
    ) = None
    match execution_mode:
        case ExecutionMode.SINGLE_THREADED:
            trainer = FedAvgBaseClientTrainer(
                model_selector=model_selector,
                model_name=model_name,
                dataset=dataset,
                device=device,
                num_clients=num_clients,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                seed=seed,
            )
        case ExecutionMode.MULTI_PROCESS:
            trainer = FedAvgProcessPoolClientTrainer(
                model_selector=model_selector,
                model_name=model_name,
                dataset=dataset,
                state_dir=state_dir,
                seed=seed,
                device=device,
                num_clients=num_clients,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                num_parallels=num_parallels,
            )
        case ExecutionMode.MULTI_THREADED:
            assert not sys._is_gil_enabled()
            trainer = FedAvgThreadPoolClientTrainer(
                model_selector=model_selector,
                model_name=model_name,
                dataset=dataset,
                seed=seed,
                device=device,
                num_clients=num_clients,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                num_parallels=num_parallels,
            )
    pipeline = FedAvgBenchmarkPipeline(handler=handler, trainer=trainer)
    try:
        pipeline.main()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt")
    finally:
        shutil.rmtree(dataset_split_dir, ignore_errors=True)
        shutil.rmtree(state_dir, ignore_errors=True)


if __name__ == "__main__":
    # NOTE: To use CUDA with multiprocessing, you must use the 'spawn' start method
    mp.set_start_method("spawn")

    typer.run(main)
