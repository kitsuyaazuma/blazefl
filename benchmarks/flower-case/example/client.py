from collections import OrderedDict

import torch
from blazefl.reproducibility import seed_everything
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from example.task import train as train_fn

from .models import get_model
from .task import load_data

# Flower ClientApp
app = ClientApp()

device = "cpu"
cuda_device_count = 0
if torch.cuda.is_available():
    device = "cuda"
    cuda_device_count = torch.cuda.device_count()
elif torch.backends.mps.is_available():
    device = "mps"


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    global device, cuda_device_count

    # Fix seed
    seed_everything(42)

    # Load the model and initialize it with the received weights
    model = get_model(str(context.run_config["model-name"]), num_classes=10)
    arrays = msg.content["arrays"]
    assert isinstance(arrays, ArrayRecord)
    model.load_state_dict(arrays.to_torch_state_dict())
    client_device = device
    if device == "cuda" and cuda_device_count > 1:
        client_device = f"cuda:{context.node_id % cuda_device_count}"
    model.to(client_device)

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
    trainloader = load_data(partition_id, num_partitions, batch_size)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        client_device,
        cid=partition_id,
    )

    # Construct and return reply Message
    model_state_dict = model.state_dict()
    assert isinstance(model_state_dict, OrderedDict)
    model_record = ArrayRecord(model_state_dict)
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),  # pyright: ignore[reportArgumentType]
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
