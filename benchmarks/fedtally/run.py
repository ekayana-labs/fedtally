"""One fedtally run as separate processes: aggregator, helper(s), sites."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODEL_TOLERANCE, Sampler, initial_model, model_error, write_result  # noqa: E402

import fedtally as ft  # noqa: E402

SITE = """
import sys
sys.path.insert(0, {common!r})
import json
from common import local_update
import fedtally as ft
i, root, addr, layout = int(sys.argv[1]), sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
out = ft.run_site(
    key=f"{{root}}/site-{{i}}.key", plan=f"{{root}}/plan.jsonl", grant=f"{{root}}/site-{{i}}.grant.json",
    connect=addr, dir=f"{{root}}/site-{{i}}", layout=[tuple(t) for t in layout],
    train=lambda r, p: (local_update(p, i), 1),
)
print(json.dumps(out))
"""


def disk_use(path) -> int:
    used = 0
    for folder, _, names in os.walk(path):
        for name in names:
            try:
                used += os.stat(os.path.join(folder, name)).st_size
            except FileNotFoundError:
                pass
    return used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, required=True)
    parser.add_argument("--mib", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--helpers", type=int, default=1)
    parser.add_argument("--epsilon", type=float, help="run with privacy terms at this target")
    parser.add_argument("--bounded", action="store_true", help="have every site prove its update short")
    parser.add_argument("--scoring", type=int, help="score groups of at least this many sites")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()

    binary = os.environ.get("FEDTALLY_BIN") or shutil.which("fedtally")
    root = Path(tempfile.mkdtemp(prefix="fedtally-bench-", dir=args.workdir))
    started_processes = []
    try:
        federation = ft.LocalFederation(
            root,
            initial_model(args.mib),
            sites=args.sites,
            helpers=args.helpers,
            rounds=args.rounds,
            min_cohort=2,
            chunk_coords=1 << 17,
            privacy=args.epsilon and ft.Privacy(epsilon=args.epsilon, delta=1e-5, clip=1.0),
            bounded=args.bounded or None,
            scoring=args.scoring,
        )
        federation.save()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            addr = f"127.0.0.1:{s.getsockname()[1]}"

        logs = open(root / "processes.log", "ab")

        def spawn(*command):
            process = subprocess.Popen(
                [binary, *map(str, command)], stdout=logs, stderr=logs
            )
            started_processes.append(process)
            return process

        started = time.monotonic()
        aggregator = spawn(
            "aggregator", "--key", root / "aggregator.key", "--plan", root / "plan.jsonl",
            "--model", root / "initial.safetensors", "--dir", root / "aggregator", "--listen", addr,
        )
        helpers = [
            spawn("helper", "--key", root / f"helper-{h}.key", "--plan", root / "plan.jsonl",
                  "--dir", root / f"helper-{h}", "--connect", addr)
            for h in range(args.helpers)
        ]
        script = SITE.format(common=str(Path(__file__).resolve().parents[1]))
        sites = [
            subprocess.Popen(
                [sys.executable, "-c", script, str(i), str(root), addr, json.dumps(federation.layout)],
                stdout=subprocess.PIPE,
                stderr=logs,
            )
            for i in range(args.sites)
        ]
        started_processes.extend(sites)
        groups = {
            "aggregator": [aggregator.pid],
            "helpers": [h.pid for h in helpers],
            "sites": [s.pid for s in sites],
        }
        peak_disk = 0
        with Sampler(groups) as sampler:
            results = []
            for site in sites:
                while site.poll() is None:
                    time.sleep(0.2)
                    peak_disk = max(peak_disk, disk_use(root / "aggregator"))
                results.append(json.loads(site.stdout.read()))
            aggregator.wait(timeout=120)
            for helper in helpers:
                helper.wait(timeout=120)
        total = time.monotonic() - started
        if not all(r["ok"] for r in results):
            raise SystemExit(f"a site failed: {results}")

        verify_started = time.monotonic()
        report = ft.verify(root / "aggregator" / "log.jsonl", results[0]["final_model"])
        verify_seconds = time.monotonic() - verify_started
        error = model_error(ft.load_model(results[0]["final_model"]), args.mib, args.sites, args.rounds)
        # Noise moves a private model away from the known answer by design.
        if error > MODEL_TOLERANCE and not args.epsilon:
            raise SystemExit(f"the final model is off by {error}")
        write_result(
            args.out,
            framework="fedtally",
            sites=args.sites,
            model_mib=args.mib,
            rounds=args.rounds,
            helpers=args.helpers,
            total_seconds=round(total, 2),
            memory=sampler.mib(),
            aggregator_peak_disk_mib=round(peak_disk / (1 << 20), 1),
            log_bytes=(root / "aggregator" / "log.jsonl").stat().st_size,
            verify_seconds=round(verify_seconds, 3),
            verified_complete=report["complete"],
            model_error=error,
            tls=True,
            masked=args.helpers > 0,
        )
    finally:
        for process in started_processes:
            if process.poll() is None:
                process.kill()
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
