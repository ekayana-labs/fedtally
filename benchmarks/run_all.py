"""Runs every harness one at a time and keeps each result.

Run it on an otherwise idle machine:

    FEDTALLY_BIN=target/release/fedtally \\
    FLOWER_PYTHON=/path/to/flower/venv/bin/python \\
    NVFLARE_PYTHON=/path/to/nvflare/venv/bin/python \\
    python benchmarks/run_all.py --out benchmarks/results --workdir /some/fast/disk
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def harnesses():
    python = sys.executable
    return {
        "fedtally-masked": [python, HERE / "fedtally/run.py", "--helpers", "1"],
        "fedtally-plain": [python, HERE / "fedtally/run.py", "--helpers", "0"],
        "flower": [os.environ["FLOWER_PYTHON"], HERE / "flower/run.py"],
        "nvflare": [os.environ["NVFLARE_PYTHON"], HERE / "nvflare/run.py"],
    }


# Each config runs with 1 and 3 rounds, so that (T3 - T1) / 2 cancels start-up and shutdown.
CONFIGS = [
    {"sites": 4, "mib": 64, "repeats": 3},
    {"sites": 4, "mib": 256, "repeats": 3},
    {"sites": 4, "mib": 1024, "repeats": 1},
    {"sites": 2, "mib": 256, "repeats": 1},
    {"sites": 8, "mib": 256, "repeats": 1},
    {"sites": 16, "mib": 256, "repeats": 1},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--only", default=None, help="run one harness")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    for config in CONFIGS:
        for repeat in range(config["repeats"]):
            for rounds in (1, 3):
                for name, command in harnesses().items():
                    if args.only and name != args.only:
                        continue
                    target = out / f"{name}-s{config['sites']}-m{config['mib']}-r{rounds}-{repeat}.json"
                    if target.exists():
                        continue
                    print(f"{target.name}", flush=True)
                    done = subprocess.run(
                        [
                            *map(str, command),
                            "--sites", str(config["sites"]),
                            "--mib", str(config["mib"]),
                            "--rounds", str(rounds),
                            "--out", str(target),
                            "--workdir", args.workdir,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    if done.returncode != 0:
                        print(f"  FAILED: {done.stderr[-600:]}", flush=True)
                        target.with_suffix(".failed").write_text(done.stderr)


if __name__ == "__main__":
    main()
