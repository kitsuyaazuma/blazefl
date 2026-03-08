#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from statistics import mean, stdev

NUM_RUNS = 10

CMD = ["uv", "run", "flwr", "run", ".", "local"]

ENV = os.environ.copy()
ENV["RAY_TMPDIR"] = "/tmp/flower-case"
ENV["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
ENV["FLWR_HOME"] = os.getcwd()

# e.g.,
# INFO :      	{ 0: {'accuracy': '1.0150e-01', ...},
# INFO :      	  1: {'accuracy': '1.3110e-01', ...},
# INFO :      	  2: {'accuracy': '1.6620e-01', ...},
# INFO :      	  3: {'accuracy': '1.9470e-01', ...}}
ACCURACY_PATTERN = re.compile(r"'accuracy': '([0-9.eE+-]+)'")


def run_once(run_idx: int) -> float:
    print(f"\n=== Run {run_idx}/{NUM_RUNS} ===")
    proc = subprocess.Popen(
        CMD,
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    matches = []

    assert proc.stdout is not None
    for line in proc.stdout:
        if "model_hash" in line:
            print(line, end="")
        found = ACCURACY_PATTERN.findall(line)
        if found:
            matches.extend(found)

    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"Run {run_idx} failed with exit code {proc.returncode}")

    if not matches:
        raise ValueError(f"Could not find accuracy in run {run_idx}")

    final_accuracy = float(matches[-1])
    print(f"[Run {run_idx}] Final Accuracy = {final_accuracy:.6f}")
    return final_accuracy


def main() -> None:
    accuracies = []

    for i in range(1, NUM_RUNS + 1):
        try:
            acc = run_once(i)
            accuracies.append(acc)
        except Exception as e:
            print(f"Error in run {i}: {e}", file=sys.stderr)
            sys.exit(1)

    avg = mean(accuracies)
    sd = stdev(accuracies) if len(accuracies) > 1 else 0.0

    print("\n=== Summary ===")
    for i, acc in enumerate(accuracies, start=1):
        print(f"Run {i}: {acc:.6f}")
    print(f"\nFinal Accuracy Mean   : {avg:.4f}")
    print(f"Final Accuracy StdDev : {sd:.4f}")


if __name__ == "__main__":
    main()
