import logging
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
from blazefl.utils import seed_everything
from hydra.core import hydra_config
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard.writer import SummaryWriter

from algorithm import DSFLBaseServerHandler, DSFLProcessPoolClientTrainer
from dataset import DSFLPartitionedDataset
from models import DSFLModelSelector


class DSFLPipeline:
    def __init__(
        self,
        handler: DSFLBaseServerHandler,
        trainer: DSFLProcessPoolClientTrainer,
        writer: SummaryWriter,
    ) -> None:
        self.handler = handler
        self.trainer = trainer
        self.writer = writer

    def main(self):
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
            for key, value in summary.items():
                self.writer.add_scalar(key, value, round_)
            formatted_summary = ", ".join(f"{k}: {v:.3f}" for k, v in summary.items())
            logging.info(f"round: {round_}, {formatted_summary}")

        logging.info("done!")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(
    cfg: DictConfig,
):
    print(OmegaConf.to_yaml(cfg))

    log_dir = hydra_config.HydraConfig.get().runtime.output_dir
    writer = SummaryWriter(log_dir=log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root_dir = Path(cfg.dataset_root_dir)
    dataset_split_dir = dataset_root_dir.joinpath(timestamp)
    share_dir = Path(cfg.share_dir).joinpath(timestamp)
    state_dir = Path(cfg.state_dir).joinpath(timestamp)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(cfg.seed, device=device)

    dataset = DSFLPartitionedDataset(
        root=dataset_root_dir,
        path=dataset_split_dir,
        num_clients=cfg.num_clients,
        dir_alpha=cfg.dir_alpha,
        seed=cfg.seed,
        partition=cfg.partition,
        open_size=cfg.algorithm.open_size,
    )
    model_selector = DSFLModelSelector(num_classes=10)

    match cfg.algorithm.name:
        case "dsfl":
            handler = DSFLBaseServerHandler(
                model_selector=model_selector,
                model_name=cfg.model_name,
                dataset=dataset,
                global_round=cfg.global_round,
                num_clients=cfg.num_clients,
                kd_epochs=cfg.algorithm.kd_epochs,
                kd_batch_size=cfg.algorithm.kd_batch_size,
                kd_lr=cfg.algorithm.kd_lr,
                era_temperature=cfg.algorithm.era_temperature,
                open_size_per_round=cfg.algorithm.open_size_per_round,
                device=device,
                sample_ratio=cfg.sample_ratio,
            )
            trainer = DSFLProcessPoolClientTrainer(
                model_selector=model_selector,
                model_name=cfg.model_name,
                dataset=dataset,
                share_dir=share_dir,
                state_dir=state_dir,
                seed=cfg.seed,
                device=device,
                num_clients=cfg.num_clients,
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                kd_epochs=cfg.algorithm.kd_epochs,
                kd_batch_size=cfg.algorithm.kd_batch_size,
                kd_lr=cfg.algorithm.kd_lr,
                num_parallels=cfg.num_parallels,
            )
            pipeline = DSFLPipeline(handler=handler, trainer=trainer, writer=writer)
        case _:
            raise ValueError(f"Invalid algorithm: {cfg.algorithm.name}")

    try:
        pipeline.main()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt")


if __name__ == "__main__":
    # NOTE: To use CUDA with multiprocessing, you must use the 'spawn' start method
    mp.set_start_method("spawn")

    main()
