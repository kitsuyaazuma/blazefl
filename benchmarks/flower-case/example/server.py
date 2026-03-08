import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable

import torch
from blazefl.reproducibility import seed_everything
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from torch import nn

from .models import get_model
from .task import load_centralized_dataset, test

seed_everything(42)

# Create ServerApp
app = ServerApp()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = float(context.run_config["fraction-evaluate"])
    num_rounds: int = int(context.run_config["num-server-rounds"])
    batch_size: int = int(context.run_config["batch-size"])
    lr: float = float(context.run_config["learning-rate"])

    # Load global model
    global_model = get_model(str(context.run_config["model-name"]), num_classes=10)
    global_model_state_dict = global_model.state_dict()
    assert isinstance(global_model_state_dict, OrderedDict)
    arrays = ArrayRecord(global_model_state_dict)

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate, min_evaluate_nodes=0)

    # Start strategy, run FedAvg for `num_rounds`
    start_time = time.perf_counter()
    _ = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(global_model, batch_size),
    )
    end_time = time.perf_counter()
    print(f"BENCHMARK_RESULT_TIME: {end_time - start_time:.4f}")


def get_evaluate_fn(
    model: nn.Module, batch_size: int
) -> Callable[[int, ArrayRecord], MetricRecord]:
    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""

        # Load the model and initialize it with the received weights
        state_dict = arrays.to_torch_state_dict()
        model.load_state_dict(state_dict)
        model.to(device)

        # Calculate model hash
        model_hash = hashlib.sha256()
        for name, param in state_dict.items():
            model_hash.update(name.encode("utf-8"))
            model_hash.update(param.cpu().numpy().tobytes())
        print(f"round: {server_round}, model_hash: {model_hash.hexdigest()}")

        # Load entire test set
        test_dataloader = load_centralized_dataset(batch_size)

        # Evaluate the global model on the test set
        test_loss, test_acc = test(model, test_dataloader, device)

        # Return the evaluation metrics
        return MetricRecord({"accuracy": test_acc, "loss": test_loss})

    return evaluate
