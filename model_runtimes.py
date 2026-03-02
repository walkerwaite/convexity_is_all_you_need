from __future__ import annotations

import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

import click
import numpy as np

from convexrobust.data import datamodules
from convexrobust.utils import dirs, file_utils, pretty


MODEL_CONVEX = "convex_reg"
MODEL_ABCROWN = "abcrown"


def _avg_cert_time(results, model_name: str) -> float:
    if model_name not in results:
        return -1.0
    cert_times = [
        r.debug_vars.get("cert_time")
        for r in results[model_name]
        if "cert_time" in r.debug_vars
    ]
    if not cert_times:
        return -1.0
    return float(np.mean(np.array(cert_times)))


def _avg_abcrown_time(experiment_directory: str, data: str) -> float:
    try:
        l1_content = file_utils.read_file(
            dirs.out_path(experiment_directory, "abcrown_times", "time_l1.txt")
        )
        l2_content = file_utils.read_file(
            dirs.out_path(experiment_directory, "abcrown_times", "time_l2.txt")
        )
        linf_content = file_utils.read_file(
            dirs.out_path(experiment_directory, "abcrown_times", "time_linf.txt")
        )
        
        # Sum all times from each file (files may contain multiple lines)
        l1_time = sum(float(line.strip()) for line in l1_content.strip().split('\n') if line.strip())
        l2_time = sum(float(line.strip()) for line in l2_content.strip().split('\n') if line.strip())
        linf_time = sum(float(line.strip()) for line in linf_content.strip().split('\n') if line.strip())
    except FileNotFoundError:
        return -1.0

    eval_n = 1000

    # Based on how many properties are evaluated in scripts/abcrown
    prop_n = {
        "mnist_38": 15 * eval_n,
        "fashion_mnist_shirts": 14 * eval_n,
        "cifar10_catsdogs": 14 * eval_n,
        "malimg": 13 * eval_n,
    }.get(data)

    if not prop_n:
        return -1.0

    return float((l1_time + l2_time + linf_time) / prop_n)


@click.command(context_settings={"show_default": True})
@click.option("--data", type=click.Choice(datamodules.names), default="mnist_38")
def main(data: str) -> None:
    pretty.init()

    experiment_directory = f"{data}-standard"
    results = file_utils.read_pickle(dirs.out_path(experiment_directory, "results.pkl"))

    convex_time = _avg_cert_time(results, MODEL_CONVEX)
    abcrown_time = _avg_abcrown_time(experiment_directory, data)

    print(f"{MODEL_CONVEX}: {convex_time}")
    print(f"{MODEL_ABCROWN}: {abcrown_time}")


if __name__ == "__main__":
    main()
