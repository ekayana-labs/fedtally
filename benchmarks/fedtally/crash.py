"""SIGKILLs fedtally's server side mid-round and records what happens.

The same scenario as benchmarks/flower/crash.py and benchmarks/nvflare/crash.py:
4 sites, 32 MiB, 4 rounds, killed once the log shows a site contributing to round 2.
"""

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
from common import MODEL_TOLERANCE, initial_model, model_error  # noqa: E402

import fedtally as ft  # noqa: E402

SITE = """
import sys
sys.path.insert(0, {common!r})
import json
from common import local_update
import fedtally as ft
i, root, addr, layout = int(sys.argv[1]), sys.argv[2], sys.argv[3], json.loads(sys.argv[4])

def train(round_number, params):
    with open(f"{{root}}/trained-{{i}}.log", "a") as f:
        f.write(f"{{round_number}}\\n")
    return local_update(params, i), 1

print(json.dumps(ft.run_site(
    key=f"{{root}}/site-{{i}}.key", plan=f"{{root}}/plan.jsonl", grant=f"{{root}}/site-{{i}}.grant.json",
    connect=addr, dir=f"{{root}}/site-{{i}}", layout=[tuple(t) for t in layout], train=train,
)))
"""


def kinds(log: Path) -> list:
    if not log.exists():
        return []
    out = []
    for line in log.read_text().splitlines():
        try:
            out.append(json.loads(line)["body"]["kind"])
        except (ValueError, KeyError):
            pass
    return out


def trial(binary, sites, mib, rounds, outage, with_helper, workdir):
    root = Path(tempfile.mkdtemp(prefix="fedtally-crash-", dir=workdir))
    processes = []
    try:
        federation = ft.LocalFederation(
            root, initial_model(mib), sites=sites, helpers=1, rounds=rounds, chunk_coords=1 << 14
        )
        federation.save()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            addr = f"127.0.0.1:{s.getsockname()[1]}"
        logs = open(root / "processes.log", "ab")

        def spawn(*command):
            process = subprocess.Popen([binary, *map(str, command)], stdout=logs, stderr=logs)
            processes.append(process)
            return process

        def aggregator():
            return spawn(
                "aggregator", "--key", root / "aggregator.key", "--plan", root / "plan.jsonl",
                "--model", root / "initial.safetensors", "--dir", root / "aggregator", "--listen", addr,
            )

        def helper():
            return spawn(
                "helper", "--key", root / "helper-0.key", "--plan", root / "plan.jsonl",
                "--dir", root / "helper-0", "--connect", addr,
            )

        started = time.monotonic()
        running_aggregator, running_helper = aggregator(), helper()
        script = SITE.format(common=str(Path(__file__).resolve().parents[1]))
        site_processes = [
            subprocess.Popen(
                [sys.executable, "-c", script, str(i), str(root), addr, json.dumps(federation.layout)],
                stdout=subprocess.PIPE,
                stderr=logs,
            )
            for i in range(sites)
        ]
        processes.extend(site_processes)

        log = root / "aggregator" / "log.jsonl"
        while True:
            seen = kinds(log)
            opens = [i for i, k in enumerate(seen) if k == "open"]
            if len(opens) >= 2 and "contribute" in seen[opens[1]:]:
                break
            if time.monotonic() - started > 120:
                raise SystemExit("round 2 never started")
            time.sleep(0.005)
        running_aggregator.kill()
        running_aggregator.wait()
        killed = ["aggregator"]
        if with_helper:
            running_helper.kill()
            running_helper.wait()
            killed.append("helper")
        entries_at_kill = len(kinds(log))

        time.sleep(outage)
        restarted = time.monotonic()
        running_aggregator = aggregator()
        if with_helper:
            running_helper = helper()

        results = [json.loads(p.communicate(timeout=300)[0]) for p in site_processes]
        finished = time.monotonic()
        running_aggregator.wait(timeout=60)
        running_helper.wait(timeout=60)

        ok = all(r["ok"] for r in results)
        report = ft.verify(log, results[0]["final_model"]) if ok else ft.verify(log)
        trained = [
            sorted(int(x) for x in (root / f"trained-{i}.log").read_text().split()) for i in range(sites)
        ]
        error = model_error(ft.load_model(results[0]["final_model"]), mib, sites, rounds) if ok else None
        return {
            "killed": killed,
            "outage_seconds": outage,
            "entries_in_log_at_kill": entries_at_kill,
            "run_finished": ok,
            "site_errors": [r.get("error") for r in results if not r["ok"]],
            "verified_complete": report["complete"],
            "seconds_from_restart_to_finish": round(finished - restarted, 2),
            "rounds_each_site_trained": trained,
            "trained_more_than_once": any(t != list(range(1, rounds + 1)) for t in trained),
            "model_error": error,
            "model_correct": error is not None and error <= MODEL_TOLERANCE,
        }
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        shutil.rmtree(root, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()
    binary = os.environ.get("FEDTALLY_BIN") or shutil.which("fedtally")

    results = []
    for outage, with_helper in [(2, False), (2, False), (2, False), (30, False), (90, False), (300, False), (2, True), (90, True)]:
        result = trial(binary, 4, 32, 4, outage, with_helper, args.workdir)
        print(json.dumps(result), flush=True)
        results.append(result)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"sites": 4, "model_mib": 32, "rounds": 4, "trials": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
