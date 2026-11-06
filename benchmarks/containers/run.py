"""A real run with each party in a container of its own, over a slow, lossy link.

Each party has its own network stack and sees delay and loss from netem, and
connects by name over TLS. Three runs: an undisturbed one, one whose aggregator
container is killed and started again, and one with a site taken off the network
for a while. Every container still shares one kernel, so this isn't separate
machines, and killing a container isn't a power cut.

    python benchmarks/containers/run.py --out benchmarks/results/containers.json
"""

import argparse
import json
import subprocess
import sys
import time

IMAGE = "fedtally-containers"
SITES, HELPERS, ROUNDS, COORDS, MIN_COHORT = 3, 2, 6, 65536, 2
DELAY, JITTER, LOSS = "40ms", "10ms", "1%"
PATIENCE = 900
NETEM = f"tc qdisc replace dev eth0 root netem delay {DELAY} {JITTER} loss {LOSS}"


def docker(*args, check=True):
    return subprocess.run(
        ["docker", *args], check=check, capture_output=True, text=True
    )


class Run:
    def __init__(self, name):
        self.name = name
        self.volume = f"{IMAGE}-{name}"
        self.network = f"{IMAGE}-{name}"
        self.parties = {}
        self.clean()
        try:
            docker("volume", "create", self.volume)
            docker("network", "create", self.network)
            self.tool(
                "python", "/opt/setup.py",
                *map(str, (SITES, HELPERS, ROUNDS, COORDS, MIN_COHORT)),
            )
        except BaseException:
            self.clean()
            raise

    def clean(self):
        for container in self.names():
            docker("rm", "-f", container, check=False)
        docker("network", "rm", self.network, check=False)
        docker("volume", "rm", "-f", self.volume, check=False)

    def names(self):
        return [self.container(p) for p in self.all_parties()]

    @staticmethod
    def all_parties():
        return (
            ["aggregator"]
            + [f"helper-{h}" for h in range(HELPERS)]
            + [f"site-{s}" for s in range(SITES)]
        )

    def container(self, party):
        return f"{self.name}-{party}"

    def tool(self, *command):
        return docker(
            "run", "--rm", "-v", f"{self.volume}:/run", IMAGE, *command
        ).stdout

    def party(self, party, *command):
        netem = f'{NETEM} && exec "$@"'
        docker(
            "run", "-d", "--name", self.container(party),
            "--network", self.network, "--network-alias", party,
            "--cap-add", "NET_ADMIN",
            "-v", f"{self.volume}:/run", IMAGE,
            "sh", "-c", netem, "party", "fedtally", *command,
        )

    def start(self):
        self.party(
            "aggregator", "aggregator", "--key", "/run/aggregator.key",
            "--plan", "/run/plan.jsonl", "--model", "/run/init.safetensors",
            "--dir", "/run/aggregator", "--listen", "0.0.0.0:7443",
        )
        for h in range(HELPERS):
            self.party(
                f"helper-{h}", "helper", "--key", f"/run/helper-{h}.key",
                "--plan", "/run/plan.jsonl", "--dir", f"/run/helper-{h}",
                "--connect", "aggregator:7443",
            )
        for s in range(SITES):
            self.party(
                f"site-{s}", "site", "--key", f"/run/site-{s}.key",
                "--plan", "/run/plan.jsonl", "--grant", f"/run/site-{s}.grant.json",
                "--connect", "aggregator:7443", "--dir", f"/run/site-{s}",
                "--", "python", "/opt/train.py",
            )

    def logged(self, kind):
        found = docker(
            "exec", self.container("aggregator"), "grep", "-c", f'"kind":"{kind}"',
            "/run/aggregator/log.jsonl", check=False,
        )
        return int(found.stdout.strip() or 0)

    def wait_for(self, kind, count=1):
        began = time.monotonic()
        while self.logged(kind) < count:
            running = docker(
                "inspect", "-f", "{{.State.Running}}", self.container("aggregator"), check=False
            ).stdout.strip()
            over = time.monotonic() - began > PATIENCE
            if running != "true" or self.logged("abort") or over:
                raise RuntimeError(f"{self.name}: no {kind} in the log")
            time.sleep(0.2)

    def finish(self, began):
        codes = {}
        for party in self.all_parties():
            waited = subprocess.run(
                ["docker", "wait", self.container(party)],
                capture_output=True, text=True, timeout=PATIENCE,
            )
            codes[party] = int(waited.stdout.strip())
        seconds = time.monotonic() - began
        logs = {
            p: docker("logs", "--tail", "2", self.container(p)).stdout.strip()
            for p in self.all_parties()
        }
        models = {p: logs[p].split()[-1] for p in logs if p.startswith("site-") and logs[p]}
        checked = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{self.volume}:/run", IMAGE,
             "fedtally", "verify", "/run/aggregator/log.jsonl"],
            capture_output=True, text=True,
        )
        # verify exits 1 on a log it can't accept, with nothing to read.
        report = json.loads(checked.stdout) if checked.returncode != 1 else {}
        result = {
            "seconds": round(seconds, 1),
            "exit_codes": codes,
            "verify_exit": checked.returncode,
            "verify_error": checked.stderr.strip()[-300:],
            "rounds": len(report.get("rounds", [])),
            "cohorts": [len(r["included"]) for r in report.get("rounds", [])],
            "same_model": len(set(models.values())) == 1 and len(models) == SITES,
        }
        result["ok"] = (
            not any(codes.values()) and checked.returncode == 0 and result["same_model"]
        )
        return result


def undisturbed(run):
    began = time.monotonic()
    run.start()
    return run.finish(began)


def aggregator_killed(run):
    began = time.monotonic()
    run.start()
    run.wait_for("outcome")
    docker("kill", run.container("aggregator"))
    time.sleep(2)
    docker("start", run.container("aggregator"))
    return run.finish(began)


def site_cut_off(run):
    began = time.monotonic()
    run.start()
    run.wait_for("seal")
    docker("network", "disconnect", run.network, run.container("site-2"))
    time.sleep(5)
    docker("network", "connect", "--alias", "site-2", run.network, run.container("site-2"))
    # A new interface comes up without the old one's delay and loss.
    docker("exec", run.container("site-2"), "sh", "-c", NETEM)
    return run.finish(began)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    if not args.no_build:
        subprocess.run(
            ["docker", "build", "-f", "benchmarks/containers/Dockerfile", "-t", IMAGE, "."],
            check=True,
        )
    results = {}
    for scenario in (undisturbed, aggregator_killed, site_cut_off):
        run = Run(scenario.__name__.replace("_", "-"))
        try:
            results[scenario.__name__] = scenario(run)
        finally:
            for party in run.all_parties():
                tail = docker("logs", "--tail", "3", run.container(party), check=False)
                print(party, tail.stdout.strip()[-200:], tail.stderr.strip()[-300:])
            run.clean()
        print(scenario.__name__, results[scenario.__name__], flush=True)
    failed = [name for name, result in results.items() if not result["ok"]]
    report = {
        "sites": SITES,
        "helpers": HELPERS,
        "rounds": ROUNDS,
        "coords": COORDS,
        "min_cohort": MIN_COHORT,
        "link": {"delay": DELAY, "jitter": JITTER, "loss": LOSS},
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    if failed:
        sys.exit(f"failed: {failed}")


if __name__ == "__main__":
    main()
