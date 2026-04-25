# ruff: noqa: E402
import logging
import signal
import warnings

from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import torch
import torch.multiprocessing as mp
import typer
import wandb
from blazefl.contrib import (
    FedAvgBaseClientTrainer,
    FedAvgBaseServerHandler,
    FedAvgProcessPoolClientTrainer,
    FedAvgThreadPoolClientTrainer,
)
from blazefl.reproducibility import setup_reproducibility

from dataset import PartitionedCIFAR10
from models import FedAvgModelSelector
from models.selector import FedAvgModelName


class FedAvgPipeline:
    def __init__(
        self,
        handler: FedAvgBaseServerHandler,
        trainer: FedAvgBaseClientTrainer
        | FedAvgProcessPoolClientTrainer
        | FedAvgThreadPoolClientTrainer,
        run: wandb.Run,
    ) -> None:
        self.handler = handler
        self.trainer = trainer
        self.run = run

    def main(self):
        while not self.handler.is_stopped():
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
            self.run.log(summary, step=round_)
            formatted_summary = ", ".join(f"{k}: {v:.3f}" for k, v in summary.items())
            logging.info(f"round: {round_}, {formatted_summary}")

        logging.info("done!")


def setup_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def raise_keyboard_interrupt_on_signal(signum: int, frame: object | None) -> None:
    _ = signum
    _ = frame
    raise KeyboardInterrupt


def main(
    model_name: Annotated[
        FedAvgModelName, typer.Option(help="Name of the model to be used.")
    ] = FedAvgModelName.CNN,
    num_clients: Annotated[
        int, typer.Option(help="Total number of clients in the federation.")
    ] = 100,
    global_round: Annotated[
        int, typer.Option(help="Total number of federated learning rounds.")
    ] = 5,
    sample_ratio: Annotated[
        float, typer.Option(help="Fraction of clients to sample in each round.")
    ] = 1.0,
    partition: Annotated[
        str,
        typer.Option(
            help="Dataset partition strategy ('shards' or 'client_inner_dirichlet')."
        ),
    ] = "shards",
    num_shards: Annotated[
        int, typer.Option(help="Number of shards for shard-based partitioning.")
    ] = 200,
    dir_alpha: Annotated[
        float,
        typer.Option(help="Alpha for Dirichlet distribution based partitioning."),
    ] = 1.0,
    seed: Annotated[int, typer.Option(help="Seed for reproducibility.")] = 42,
    epochs: Annotated[
        int, typer.Option(help="Number of local training epochs per client.")
    ] = 5,
    lr: Annotated[
        float, typer.Option(help="Learning rate for the client optimizer.")
    ] = 0.1,
    batch_size: Annotated[
        int, typer.Option(help="Batch size for local training.")
    ] = 50,
    num_parallels: Annotated[
        int,
        typer.Option(help="Number of parallel processes for training."),
    ] = 10,
    dataset_root_dir: Annotated[
        Path, typer.Option(help="Root directory for the dataset.")
    ] = Path("/tmp/quickstart-fedavg/dataset"),
    state_dir_base: Annotated[
        Path, typer.Option(help="Directory path for saving data between processes.")
    ] = Path("/tmp/quickstart-fedavg/state"),
    execution_mode: Annotated[
        str,
        typer.Option(
            help=(
                "Execution mode: 'single-threaded', 'multi-process', or "
                "'multi-threaded'."
            )
        ),
    ] = "multi-threaded",
) -> None:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    config = locals()
    run = wandb.init(mode="offline", config=config)

    setup_logging()
    logging.info("\n" + "\n".join([f"  {k}: {v}" for k, v in config.items()]))
    signal.signal(signal.SIGINT, raise_keyboard_interrupt_on_signal)
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt_on_signal)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_split_dir = dataset_root_dir / timestamp
    state_dir = state_dir_base / timestamp

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
        case "single-threaded":
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
        case "multi-process":
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
        case "multi-threaded":
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
        case _:
            raise ValueError(f"Invalid execution mode: {execution_mode}")
    pipeline = FedAvgPipeline(handler=handler, trainer=trainer, run=run)
    try:
        pipeline.main()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt")
    finally:
        shutdown = getattr(trainer, "shutdown", None)
        if callable(shutdown):
            shutdown()


if __name__ == "__main__":
    # NOTE: To use CUDA with multiprocessing, you must use the 'spawn' start method
    mp.set_start_method("spawn")

    typer.run(main)
