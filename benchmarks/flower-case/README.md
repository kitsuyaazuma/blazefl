# Benchmark: Flower

This benchmark measures the performance of [Flower](https://github.com/adap/flower).

## Setup

```bash
git clone https://github.com/kitsuyaazuma/blazefl.git
cd blazefl/benchmarks/flower-case

uv sync
```

## Usage

```bash
RAY_TMPDIR=/tmp/flower-case FLWR_HOME=$(pwd) uv run flwr run .
```
