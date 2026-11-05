"""Turns a results folder into the tables in docs/benchmarks.md."""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

NAMES = ["fedtally-masked", "fedtally-plain", "flower", "nvflare"]


def load(folder):
    runs = defaultdict(list)
    for path in Path(folder).glob("*.json"):
        name, sites, mib, rounds, _ = path.stem.rsplit("-", 4)
        runs[(name, int(sites[1:]), int(mib[1:]), int(rounds[1:]))].append(json.loads(path.read_text()))
    return runs


def server_mib(run):
    memory = run["memory"]
    if "aggregator" in memory:
        return memory["aggregator"]["peak_total_mib"] + memory["helpers"]["peak_total_mib"]
    return memory["server"]["peak_total_mib"]


def row(runs, name, sites, mib):
    one, three = runs.get((name, sites, mib, 1)), runs.get((name, sites, mib, 3))
    if not one or not three:
        return None
    t1 = statistics.median(r["total_seconds"] for r in one)
    t3 = statistics.median(r["total_seconds"] for r in three)
    disk = max((r.get("aggregator_peak_disk_mib", 0) for r in three), default=0)
    return {
        "per_round": (t3 - t1) / 2,
        "t3": t3,
        "spread": (max(r["total_seconds"] for r in three) - min(r["total_seconds"] for r in three)),
        "server": max(server_mib(r) for r in three),
        "site": max(r["memory"]["sites"]["peak_each_mib"] for r in three),
        "disk": disk,
        "repeats": len(three),
    }


def table(runs, keys):
    lines = [
        "| Sites | Model MiB | Framework | s per round | Server peak MiB | Peak per site MiB | Server disk MiB | Repeats |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for sites, mib in keys:
        for name in NAMES:
            r = row(runs, name, sites, mib)
            if r is None:
                lines.append(f"| {sites} | {mib} | {name} | no result | | | | |")
                continue
            disk = f"{r['disk']:.0f}" if r["disk"] else "not measured"
            lines.append(
                f"| {sites} | {mib} | {name} | {r['per_round']:.1f} | {r['server']:.0f} "
                f"| {r['site']:.0f} | {disk} | {r['repeats']} |"
            )
    return "\n".join(lines)


def main():
    runs = load(sys.argv[1])
    print("By model size\n")
    print(table(runs, [(4, 64), (4, 256), (4, 1024)]))
    print("\nBy site count\n")
    print(table(runs, [(2, 256), (4, 256), (8, 256), (16, 256)]))
    failed = list(Path(sys.argv[1]).glob("*.failed"))
    print(f"\nFailed runs: {[p.stem for p in failed]}")


if __name__ == "__main__":
    main()
