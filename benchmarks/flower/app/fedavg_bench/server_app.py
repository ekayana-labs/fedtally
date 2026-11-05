"""ServerApp: stock FedAvg over the shared model, full participation, no evaluation."""

import json
import time
from pathlib import Path

from common import initial_model, model_error
from flwr.app import Array, ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

app = ServerApp()


def count_replies(contents, weight_key) -> MetricRecord:
    return MetricRecord({"replies": len(contents)})


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = context.run_config
    arrays = ArrayRecord({k: Array(v) for k, v in initial_model(config["mib"]).items()})
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=config["sites"],
        min_available_nodes=config["sites"],
        train_metrics_aggr_fn=count_replies,
    )
    strategy_started = time.monotonic()
    result = strategy.start(grid=grid, initial_arrays=arrays, num_rounds=config["rounds"])
    check_started = time.monotonic()
    strategy_finished = time.time()

    # What the harness needs to check the model. run.py takes this time off total_seconds.
    final = {name: array.numpy() for name, array in result.arrays.items()}
    tensors = {
        name: {"shape": list(values.shape), "dtype": str(values.dtype)}
        for name, values in final.items()
    }
    error = model_error(final, config["mib"], config["sites"], config["rounds"])
    summary = {
        "replies": [m["replies"] for _, m in sorted(result.train_metrics_clientapp.items())],
        "tensors": tensors,
        "model_error": error,
        "strategy_seconds": check_started - strategy_started,
        "strategy_finished": strategy_finished,
        "check_seconds": time.monotonic() - check_started,
    }
    Path(config["result-file"]).write_text(json.dumps(summary))
