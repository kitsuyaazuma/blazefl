# ruff: noqa: E402
import logging
import warnings

from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import torch
import typer
import wandb
from blazefl.reproducibility import setup_reproducibility

from algorithm import DSFLBaseServerHandler, DSFLThreadPoolClientTrainer
from dataset import DSFLPartitionedDataset
from models import DSFLModelName, DSFLModelSelector


class DSFLPipeline:
    def __init__(
        self,
        handler: DSFLBaseServerHandler,
        trainer: DSFLThreadPoolClientTrainer,
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


def main(
    model_name: Annotated[
        DSFLModelName, typer.Option(help="Name of the model to be used.")
    ] = DSFLModelName.CNN,
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
        typer.Option(help="Dataset partition strategy ('client_inner_dirichlet')."),
    ] = "client_inner_dirichlet",
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
    ] = Path("/tmp/step-by-step-dsfl/dataset"),
    state_dir_base: Annotated[
        Path, typer.Option(help="Directory path for saving data between processes.")
    ] = Path("/tmp/step-by-step-dsfl/state"),
    open_size: Annotated[int, typer.Option(help="Size of the open dataset.")] = 10000,
    kd_epochs: Annotated[
        int, typer.Option(help="Number of knowledge distillation epochs.")
    ] = 5,
    kd_batch_size: Annotated[
        int, typer.Option(help="Batch size for knowledge distillation.")
    ] = 50,
    kd_lr: Annotated[
        float, typer.Option(help="Learning rate for knowledge distillation.")
    ] = 0.1,
    era_temperature: Annotated[
        float, typer.Option(help="Temperature for Entropy Reduction Aggregation.")
    ] = 0.1,
    open_size_per_round: Annotated[
        int, typer.Option(help="Size of the open dataset to use per round.")
    ] = 1000,
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_split_dir = dataset_root_dir / timestamp
    state_dir = state_dir_base / timestamp

    setup_reproducibility(seed)

    dataset = DSFLPartitionedDataset(
        root=dataset_root_dir,
        path=dataset_split_dir,
        num_clients=num_clients,
        dir_alpha=dir_alpha,
        seed=seed,
        partition=partition,
        open_size=open_size,
    )
    model_selector = DSFLModelSelector(num_classes=10, seed=seed)

    handler = DSFLBaseServerHandler(
        model_selector=model_selector,
        model_name=model_name,
        dataset=dataset,
        global_round=global_round,
        num_clients=num_clients,
        kd_epochs=kd_epochs,
        kd_batch_size=kd_batch_size,
        kd_lr=kd_lr,
        era_temperature=era_temperature,
        open_size_per_round=open_size_per_round,
        device=device,
        sample_ratio=sample_ratio,
        seed=seed,
    )
    trainer = DSFLThreadPoolClientTrainer(
        model_selector=model_selector,
        model_name=model_name,
        dataset=dataset,
        state_dir=state_dir,
        seed=seed,
        device=device,
        num_clients=num_clients,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        kd_epochs=kd_epochs,
        kd_batch_size=kd_batch_size,
        kd_lr=kd_lr,
        num_parallels=num_parallels,
    )
    pipeline = DSFLPipeline(handler=handler, trainer=trainer, run=run)

    try:
        pipeline.main()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt")


if __name__ == "__main__":
    typer.run(main)
