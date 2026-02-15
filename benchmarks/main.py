import logging
import math
import os
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal, NamedTuple

import toml
import torch
import typer
import wandb
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from rich.table import Table

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.log"


def setup_logging() -> None:
    console_handler = RichHandler(markup=True, rich_tracebacks=True)
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="[%X]")
    )

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler, file_handler],
    )


def run_benchmark(command: str, num_runs: int) -> list[float]:
    execution_times: list[float] = []
    for i in track(range(num_runs), description="Benchmark Progress"):
        logging.info(f"Running benchmark (run {i + 1}/{num_runs}): {command}")
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, check=True
            )

            if result.stdout:
                logging.debug(f"Run {i + 1} stdout:\n{result.stdout.strip()}")
            if result.stderr:
                logging.debug(f"Run {i + 1} stderr:\n{result.stderr.strip()}")

            output = result.stdout + result.stderr
            match = re.search(r"BENCHMARK_RESULT_TIME: ([\d.]+)", output)
            if match:
                time_taken = float(match.group(1))
                execution_times.append(time_taken)
                logging.info(
                    f"Run {i + 1}/{num_runs} finished in {time_taken:.4f} seconds."
                )
            else:
                logging.warning(
                    f"Could not find benchmark result in output for run {i + 1}."
                )
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed for run {i + 1}/{num_runs}")
            logging.error(f"Return code: {e.returncode}")

            if e.stdout:
                logging.error(f"stdout:\n{e.stdout.strip()}")
            if e.stderr:
                logging.error(f"stderr:\n{e.stderr.strip()}")
            continue

    return execution_times


class Result(NamedTuple):
    method: str
    avg_time: float
    std_time: float


class BenchmarkEntry(NamedTuple):
    result: Result
    num_parallel: int


def display_results(title: str, results: list[Result]) -> None:
    console = Console()

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Method", style="dim")
    table.add_column("Execution Time (s)", justify="right")
    for result in results:
        row = f"{result.avg_time:.4f} ± {result.std_time:.4f}"
        table.add_row(result.method, row)
        logging.info(f"Result - {result.method}: {row}")

    console.print(table)


def log_result(num_parallel: int, result: Result) -> None:
    wandb.log(
        {
            "num_parallel": num_parallel,
            f"benchmark/{result.method}/avg_time": result.avg_time,
            f"benchmark/{result.method}/std_time": result.std_time,
        }
    )


def main(num_runs: int = 3, model_name: Literal["CNN", "RESNET18"] = "CNN") -> None:
    logging.info("Starting benchmark...")
    cpu_count = os.cpu_count() or 1
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    num_parallels: list[int] = [2**i for i in range(int(math.log2(cpu_count) + 1))]

    wandb_config = {
        "model_name": model_name,
        "num_runs": num_runs,
        "cpu_count": cpu_count,
        "gpu_count": gpu_count,
    }
    run = wandb.init(config=wandb_config)
    run.define_metric("num_parallel")
    run.define_metric("benchmark/*", step_metric="num_parallel")

    data: list[BenchmarkEntry] = []

    for num_parallel in num_parallels:
        results: list[Result] = []

        execution_modes = ["MULTI_THREADED", "MULTI_PROCESS"]
        for mode in execution_modes:
            blazefl_command = (
                "cd blazefl-case && "
                "uv run python "
                f"{'-Xgil=0' if mode == 'MULTI_THREADED' else ''} main.py "
                f"--model-name={model_name} "
                f"--execution-mode={mode} "
                f"--num-parallels={num_parallel} "
            )
            blazefl_times = run_benchmark(blazefl_command, num_runs)
            if blazefl_times:
                method_name = f"BlazeFL ({mode})"
                avg_time = statistics.mean(blazefl_times)
                std_time = (
                    statistics.stdev(blazefl_times) if len(blazefl_times) > 1 else 0.0
                )
                result = Result(
                    method=method_name,
                    avg_time=avg_time,
                    std_time=std_time,
                )
                results.append(result)
                log_result(num_parallel, result)

        client_cpus = cpu_count // num_parallel

        # Update Flower config
        config_tmpl_path = Path("flower-case/config.toml.tmpl")
        config_path = Path("flower-case/config.toml")
        if config_tmpl_path.exists():
            config = toml.load(config_tmpl_path)
            config["superlink"]["local"]["options"]["backend"]["client-resources"][
                "num-cpus"
            ] = client_cpus
            with open(config_path, "w") as f:
                toml.dump(config, f)

        flower_command = (
            "cd flower-case && "
            "RAY_TMPDIR=/tmp/flower-case "
            "FLWR_HOME=$(pwd) "
            "uv run flwr run . local "
            f"--run-config 'model-name=\"{model_name}\"' "
            "&& rm -rf ${RAY_TMPDIR} "
            "&& cd .."
        )
        flower_times = run_benchmark(flower_command, num_runs)
        if flower_times:
            method_name = "Flower"
            avg_time = statistics.mean(flower_times)
            std_time = statistics.stdev(flower_times) if len(flower_times) > 1 else 0.0
            result = Result(
                method=method_name,
                avg_time=avg_time,
                std_time=std_time,
            )
            results.append(result)
            log_result(num_parallel, result)

        data.extend(
            [
                BenchmarkEntry(result=result, num_parallel=num_parallel)
                for result in results
            ]
        )
        display_results(f"FedAvg Benchmark ({num_parallel=})", results)

    table_columns = ["num_parallel", "method", "avg_time", "std_time"]
    table_data = [
        [
            entry.num_parallel,
            entry.result.method,
            entry.result.avg_time,
            entry.result.std_time,
        ]
        for entry in data
    ]

    wandb.log(
        {
            "benchmark_table": wandb.Table(columns=table_columns, data=table_data),
        }
    )

    logging.info("Benchmark finished.")
    run.finish()


if __name__ == "__main__":
    setup_logging()
    typer.run(main)
