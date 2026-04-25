# Step-by-Step Tutorial: DS-FL

Welcome to this step-by-step tutorial on implementing DS-FL[^1] using BlazeFL!
DS-FL is a Federated Learning (FL) method that utilizes knowledge distillation by sharing model outputs on an open dataset. 

Thanks to BlazeFL's highly modular design, you can easily implement both standard FL approaches (like parameter exchange) and advanced methods (like distillation-based FL).
Think of it as assembling puzzle pieces to create your own unique FL methods—beyond the constraints of traditional frameworks.

In this tutorial, we’ll guide you through creating a DS-FL pipeline using BlazeFL.
By following along, you’ll be able to develop your own original FL methods.

## Setup a Project

Start by creating a new directory for your DS-FL project:

```bash
mkdir step-by-step-dsfl
cd step-by-step-dsfl
```

Next, Initialize the project with [uv](https://github.com/astral-sh/uv) (or any other package manager of your choice).

```bash
uv python pin 3.14t
uv init -p 3.14t
```

Then, create a virtual environment and install BlazeFL. 

```bash
uv venv -p 3.14t
# source .venv/bin/activate
uv add "blazefl[reproducibility]"
```

## Implementing a PartitionedDataset

Before running Federated Learning, it’s common to pre-split the dataset for each client.
By saving these partitions ahead of time, your server or clients can simply load the data each round without re-partitioning.

In BlazeFL, we recommend extending the `PartitionedDataset` abstract class to create your own dataset class. 
For example, you can implement `DSFLPartitionedDataset` like this:

```python
from blazefl.core import PartitionedDataset

class DSFLPartitionType(StrEnum):
    TRAIN = "TRAIN"
    OPEN = "OPEN"
    TEST = "TEST"

class DSFLPartitionedDataset(PartitionedDataset[DSFLPartitionType]):
    # Omited for brevity

    def get_dataset(self, type_: DSFLPartitionType, cid: int | None) -> Dataset:
        match type_:
            case DSFLPartitionType.TRAIN:
                dataset = torch.load(
                    self.path.joinpath(type_, f"{cid}.pkl"),
                    weights_only=False,
                )
            case DSFLPartitionType.OPEN:
                dataset = torch.load(
                    self.path.joinpath(f"{type_}.pkl"),
                    weights_only=False,
                )
            case DSFLPartitionType.TEST:
                if cid is not None:
                    dataset = torch.load(
                        self.path.joinpath(type_, f"{cid}.pkl"),
                        weights_only=False,
                    )
                else:
                    dataset = torch.load(
                        self.path.joinpath(type_, "default.pkl"), weights_only=False
                    )
        assert isinstance(dataset, Dataset)
        return dataset

    def set_dataset(
        self, type_: DSFLPartitionType, cid: int | None, dataset: Dataset
    ) -> None:
        ...  # Omitted for brevity

    def get_dataloader(
        self,
        type_: DSFLPartitionType,
        cid: int | None,
        batch_size: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DataLoader:
        dataset = self.get_dataset(type_, cid)
        assert isinstance(dataset, Sized)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=type_ == DSFLPartitionType.TRAIN,
            generator=generator,
        )
        return data_loader
```

Here, `get_dataset` returns a `Dataset` for the specified type (e.g., "train", "open", or "test") and client ID.
`set_dataset` saves a dataset to the specified path.
Meanwhile, `get_dataloader` wraps that dataset in a `DataLoader`.
This design is flexible enough even for methods like DS-FL, which rely on an open dataset.
If you don’t need one of these methods, you can simply implement it with `pass`.

You can view the complete source code [here](https://github.com/kitsuyaazuma/blazefl/tree/main/examples/step-by-step-dsfl/dataset).

## Implementing a ModelSelector

Most traditional FL frameworks assume all clients use the same model, but in distillation-based methods like DS-FL, clients can use different models.

BlazeFL provides an abstract class called `ModelSelector` to handle this scenario.
It lets you select different models on the fly for the server and clients.
For instance:

```python
from enum import StrEnum
from blazefl.core import ModelSelector

class DSFLModelName(StrEnum):
    CNN = "CNN"
    RESNET18 = "RESNET18"

class DSFLModelSelector(ModelSelector[DSFLModelName]):
    def __init__(self, num_classes: int, seed: int) -> None:
        self.num_classes = num_classes
        self.seed = seed

    def select_model(self, model_name: DSFLModelName) -> nn.Module:
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            match model_name:
                case DSFLModelName.CNN:
                    return CNN(num_classes=self.num_classes)
                case DSFLModelName.RESNET18:
                    return resnet18(num_classes=self.num_classes)
```

Here, we define model names using `StrEnum` for better type safety. The `select_model` method then takes a `DSFLModelName` and returns the corresponding `nn.Module`.
You can store useful information (like the number of classes) as attributes in your `ModelSelector`.

The full source code can be found [here](https://github.com/kitsuyaazuma/blazefl/tree/main/examples/step-by-step-dsfl/models).

## Defining DownlinkPackage and UplinkPackage

In many FL frameworks, communication between the server and clients is often handled through generic data structures like dictionaries or lists.
However, BlazeFL encourages you to define dedicated classes for these communication packets, making your code more organized and readable.

In DS-FL, you could define them like this:

```python
@dataclass
class DSFLUplinkPackage:
    cid: int
    soft_labels: torch.Tensor
    indices: torch.Tensor
    metadata: dict


@dataclass
class DSFLDownlinkPackage:
    soft_labels: torch.Tensor | None
    indices: torch.Tensor | None
    next_indices: torch.Tensor
```

Using Python’s `@dataclass` makes these classes concise and easy to maintain.
Including explicit types for each attribute also improves IDE support for debugging.

## Implementing a ServerHandler

The server in an FL setup typically handles aggregating information from clients and updating the global model.
BlazeFL does not force any specific "aggregation" or "update" strategy.
Instead, it provides a flexible `BaseServerHandler` class that focuses on the necessary client-server communication.

Below is an example for DS-FL:

```python
class DSFLBaseServerHandler(BaseServerHandler[DSFLUplinkPackage, DSFLDownlinkPackage]):
    # Omitted for brevity

    def sample_clients(self) -> list[int]:
        sampled_clients = self.rng_suite.python.sample(
            range(self.num_clients), self.num_clients_per_round
        )

        return sorted(sampled_clients)

    def is_stopped(self) -> bool:
        return self.round >= self.global_round

    def load(self, payload: DSFLUplinkPackage) -> bool:
        self.client_buffer_cache.append(payload)

        if len(self.client_buffer_cache) == self.num_clients_per_round:
            self.global_update(self.client_buffer_cache)
            self.round += 1
            self.client_buffer_cache = []
            return True
        else:
            return False

    def global_update(self, buffer) -> None:
        buffer.sort(key=lambda x: x.cid)
        soft_labels_list = [ele.soft_labels for ele in buffer]
        indices_list = [ele.indices for ele in buffer]
        self.metadata_list = [ele.metadata for ele in buffer]

        soft_labels_stack: defaultdict[int, list[torch.Tensor]] = defaultdict(
            list[torch.Tensor]
        )
        for soft_labels, indices in zip(soft_labels_list, indices_list, strict=True):
            for soft_label, index in zip(soft_labels, indices, strict=True):
                soft_labels_stack[int(index.item())].append(soft_label)

        global_soft_labels: list[torch.Tensor] = []
        global_indices: list[int] = []
        for indices, soft_labels in sorted(
            soft_labels_stack.items(), key=lambda x: x[0]
        ):
            global_indices.append(indices)
            mean_soft_labels = torch.mean(torch.stack(soft_labels), dim=0)
            # Entropy Reduction Aggregation (ERA)
            era_soft_labels = F.softmax(mean_soft_labels / self.era_temperature, dim=0)
            global_soft_labels.append(era_soft_labels)

        open_dataset = self.dataset.get_dataset(type_=DSFLPartitionType.OPEN, cid=None)
        open_loader = DataLoader(
            Subset(open_dataset, global_indices),
            batch_size=self.kd_batch_size,
        )
        DSFLBaseServerHandler.distill(
            self.model,
            self.kd_optimizer,
            open_loader,
            global_soft_labels,
            self.kd_epochs,
            self.kd_batch_size,
            self.device,
            stop_event=None,
        )

        self.global_soft_labels = torch.stack(global_soft_labels)
        self.global_indices = torch.tensor(global_indices) 

    # Omited for brevity

    def downlink_package(self) -> DSFLDownlinkPackage:
        next_indices = self.get_next_indices()
        return DSFLDownlinkPackage(
            self.global_soft_labels, self.global_indices, next_indices
        )
```

The `BaseServerHandler` class requires five core methods to be implemented:

- `sample_clients`
- `if_stop`
- `load`
- `global_update`
- `downlink_package`

Additionally, you can implement other methods like `get_summary` to get a summary of the round.

If any of these methods are not needed for your approach, you can simply implement them with `pass`.

In DS-FL, the `global_update` method aggregates the soft labels from clients and distills them into a global model.
However, you have the flexibility to place any custom operations in these or other methods.
You can find more details in the [official documentation](https://kitsuyaazuma.github.io/blazefl/generated/blazefl.core.BaseServerHandler.html#blazefl.core.BaseServerHandler).


## Implementing a ThreadPoolClientTrainer

Traditional FL frameworks often train each client sequentially or manage complex multiprocessing setups.
With BlazeFL, the `ThreadPoolClientTrainer` class lets you train multiple clients in parallel using threads, which can be sufficient and simpler for many simulation scenarios, especially when managing lightweight models or when I/O bound.

An example DS-FL client trainer looks like this:

```python
@dataclass
class DSFLClientState:
    random: RNGSuite
    model: dict[str, torch.Tensor]
    optimizer: dict[str, torch.Tensor]
    kd_optimizer: dict[str, torch.Tensor] | None


class DSFLThreadPoolClientTrainer(
    ThreadPoolClientTrainer[DSFLUplinkPackage, DSFLDownlinkPackage]
):
    # Omitted for brevity

    def worker(
        self,
        cid: int,
        device: str,
        payload: DSFLDownlinkPackage,
        stop_event: threading.Event,
    ) -> DSFLUplinkPackage:
        model = self.model_selector.select_model(self.model_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        kd_optimizer: torch.optim.SGD | None = None

        if cid in self.client_states:
            state = self.client_states[cid]
            rng_suite = state.random
            model.load_state_dict(state.model)
            optimizer.load_state_dict(state.optimizer)
            if state.kd_optimizer is not None:
                kd_optimizer = torch.optim.SGD(model.parameters(), lr=self.kd_lr)
                kd_optimizer.load_state_dict(state.kd_optimizer)
        else:
            rng_suite = create_rng_suite(self.seed)

        # Distill
        open_dataset = self.dataset.get_dataset(type_=DSFLPartitionType.OPEN, cid=None)
        if payload.indices is not None and payload.soft_labels is not None:
            global_soft_labels = list(torch.unbind(payload.soft_labels, dim=0))
            global_indices = payload.indices.tolist()
            if kd_optimizer is None:
                kd_optimizer = torch.optim.SGD(model.parameters(), lr=self.kd_lr)

            open_loader = DataLoader(
                Subset(open_dataset, global_indices),
                batch_size=self.kd_batch_size,
            )
            DSFLBaseServerHandler.distill(
                model=model,
                optimizer=kd_optimizer,
                open_loader=open_loader,
                global_soft_labels=global_soft_labels,
                kd_epochs=self.kd_epochs,
                kd_batch_size=self.kd_batch_size,
                device=device,
                stop_event=stop_event,
            )

        # Train
        train_loader = self.dataset.get_dataloader(
            type_=DSFLPartitionType.TRAIN,
            cid=cid,
            batch_size=self.batch_size,
            generator=rng_suite.torch_cpu,
        )
        DSFLThreadPoolClientTrainer.train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epochs=self.epochs,
            stop_event=stop_event,
        )
        self.dataset.set_dataset(
            dataset=train_loader.dataset, type_=DSFLPartitionType.TRAIN, cid=cid
        )

        # Predict
        open_loader = DataLoader(
            Subset(open_dataset, payload.next_indices.tolist()),
            batch_size=self.batch_size,
        )
        soft_labels = DSFLThreadPoolClientTrainer.predict(
            model=model,
            open_loader=open_loader,
            device=device,
        )

        # Evaluate
        test_loader = self.dataset.get_dataloader(
            type_=DSFLPartitionType.TEST,
            cid=cid,
            batch_size=self.batch_size,
        )
        loss, acc = DSFLBaseServerHandler.evaulate(
            model=model,
            test_loader=test_loader,
            device=device,
        )

        package = DSFLUplinkPackage(
            cid=cid,
            soft_labels=soft_labels,
            indices=payload.next_indices,
            metadata={"loss": loss, "acc": acc},
        )

        self.client_states[cid] = DSFLClientState(
            random=rng_suite,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            kd_optimizer=kd_optimizer.state_dict() if kd_optimizer else None,
        )

        return package

    def uplink_package(self) -> list[DSFLUplinkPackage]:
        package = deepcopy(self.cache)
        self.cache = []
        return package
```

This class uses Python’s `threading` library (wrapped under BlazeFL) to train clients concurrently. You mainly need to implement:

- `worker` (called by threads to process each client)
- `uplink_package` (to send final results back to the server)

Unlike process-based training, threads share the same memory space, making it easy to manage state using a simple dictionary (`self.client_states`) without complex serialization.

The complete source code is [here](https://github.com/kitsuyaazuma/blazefl/tree/main/examples/step-by-step-dsfl/algorithm/dsfl.py).

## Implementing a Pipeline

A `Pipeline` is optional but can help organize your simulation workflow, making it easy to run experiments in a structured way.
Here’s an example DS-FL pipeline:

```python
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
```

This pipeline is almost identical to one you might create for FedAvg or another standard FL method, showcasing how reusable these components are. 

In this snippet, we use [W&B](https://github.com/wandb/wandb) for logging, but you’re free to use alternatives like TensorBoard.

You can see the full source code [here](https://github.com/kitsuyaazuma/blazefl/tree/main/examples/step-by-step-dsfl/main.py).

## Running the Simulation

In our example, we use [Typer](https://github.com/fastapi/typer) to handle hyperparameter configuration. Feel free to use any configuration system you like.

To run the DS-FL simulation:

```bash
uv run python main.py
```

## Conclusion

In this tutorial, you learned how to implement DS-FL using BlazeFL.
BlazeFL’s flexible design eliminates many constraints seen in traditional FL frameworks, allowing you to mix and match components like building blocks.

Use BlazeFL to implement your own original FL methods and drive pioneering research in Federated Learning.
Push boundaries and have fun exploring innovative approaches!

[^1]: S. Itahara, T. Nishio, Y. Koda, M. Morikura, and K. Yamamoto, "Distillation-Based Semi-Supervised Federated Learning for Communication-Efficient Collaborative Training With Non-IID Private Data," IEEE Trans. Mobile Comput., vol. 22, no. 1, pp. 191–205, 2023.
