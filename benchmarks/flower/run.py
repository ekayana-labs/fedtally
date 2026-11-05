"""One Flower FedAvg run on the deployment runtime, every part a separate process."""

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

BENCHMARKS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARKS))
from common import MODEL_TOLERANCE, N_TENSORS, Sampler, tensor_shape, tree_rss, write_result  # noqa: E402

APP = Path(__file__).resolve().parent / "app"
BIN = Path(sys.executable).parent
LINK = "bench"
READY_TIMEOUT = 120

NOTES = [
    "total_seconds starts just before `flwr run` and ends when `flwr run --stream` exits, less "
    "check_seconds. It includes CLI start-up, the FAB build and upload, and the log stream's "
    "0.5 s poll. SuperLink and SuperNode start-up is excluded and given as startup_seconds.",
    "Waits before the timer until every SuperNode is online and every SuperExec is connected.",
    "strategy_seconds is FedAvg alone, timed in the ServerApp. after_strategy_seconds runs from "
    "FedAvg returning to the CLI exiting. superlink_elapsed_seconds is running to finished as "
    "the SuperLink records it.",
    "Stock polling stays in the numbers: SuperNode 3 s when idle, SuperExec 1 s, ServerApp 1 s. "
    "One ClientApp process starts per message, so per site per round.",
    "SuperLink ran with --disable-runtime-dependency-installation, so no `uv sync` per run. "
    "State and objects are in memory, the SuperLink default.",
    "Telemetry and the update check were off. Both apps read common.py through PYTHONPATH.",
    "common.py's tensors are random and each site's update differs in every coordinate, so "
    "Flower's content-addressed chunks find nothing to deduplicate. network_bytes should be "
    "close to nominal_bytes.",
    "Memory is summed RSS per process tree, so pages shared between processes count more than "
    "once. The `flwr run` CLI process is in neither group.",
]


def free_ports(count):
    sockets = [socket.socket() for _ in range(count)]
    for s in sockets:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in sockets]
    for s in sockets:
        s.close()
    return ports


def run_flwr_json(env, *args):
    done = subprocess.run(
        [BIN / "flwr", *args, "--format", "json"],
        env=env, cwd=env["FLWR_HOME"], capture_output=True, text=True, timeout=60,
    )
    try:
        reply = json.loads(done.stdout)
    except ValueError:
        return None
    return reply if reply.get("success") else None


def tree(pid):
    try:
        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except psutil.Error:
        return []


def script_name(process):
    try:
        return Path(process.cmdline()[1]).name
    except (psutil.Error, IndexError):
        return None


def superexec_connected(pid):
    for process in tree(pid):
        if script_name(process) == "flower-superexec":
            try:
                return any(c.status == "ESTABLISHED" for c in process.net_connections("tcp"))
            except psutil.Error:
                return False
    return False


def note_members(seen, roots):
    for name, pid in roots.items():
        for process in tree(pid):
            script = script_name(process)
            if script:
                seen[name].setdefault(script, set()).add(process.pid)


def check_alive(processes):
    for process in processes:
        if process.poll() is not None:
            raise SystemExit(f"{process.args[0]} exited with {process.returncode}")


def wait_ready(env, superlink, supernodes, sites):
    deadline = time.monotonic() + READY_TIMEOUT
    while time.monotonic() < deadline:
        check_alive([superlink, *supernodes])
        if all(superexec_connected(p.pid) for p in [superlink, *supernodes]):
            reply = run_flwr_json(env, "supernode", "ls", LINK)
            if reply and sum(n["status"] == "online" for n in reply["nodes"]) == sites:
                return
        time.sleep(0.2)
    raise SystemExit(f"SuperLink and {sites} SuperNodes not ready after {READY_TIMEOUT} s")


def check_model(summary, sites, mib, rounds):
    """The final model must be the mean of the updates, with every site in every round."""
    if summary["replies"] != [sites] * rounds:
        raise SystemExit(f"expected {sites} replies per round, got {summary['replies']}")
    tensors = summary["tensors"]
    if set(tensors) != {f"t{i}" for i in range(N_TENSORS)}:
        raise SystemExit(f"wrong tensors: {sorted(tensors)}")
    for name, t in tensors.items():
        if t["shape"] != list(tensor_shape(mib)) or t["dtype"] != "float32":
            raise SystemExit(f"{name}: wrong shape or type: {t}")
    if summary["model_error"] > MODEL_TOLERANCE:
        raise SystemExit(f"the final model is off by {summary['model_error']}")
    return summary["model_error"]


def running_in(root):
    # By working directory, which children inherit. psutil's environ() is empty on recent macOS.
    found = []
    for process in psutil.process_iter():
        try:
            if Path(process.cwd()).is_relative_to(root):
                found.append(process)
        except psutil.Error:
            pass
    return found


def signal_groups(processes, sig):
    groups = set()
    for process in processes:
        try:
            groups.add(os.getpgid(process.pid))
        except OSError:
            pass
    groups.discard(os.getpgrp())
    for group in groups:
        try:
            os.killpg(group, sig)
        except OSError:
            pass


def stop_all(started, root):
    # The SuperLink's SuperExec starts its own session, so its group is not the SuperLink's.
    processes = {p.pid: p for pid in started for p in tree(pid)}
    processes.update({p.pid: p for p in running_in(root)})
    signal_groups(processes.values(), signal.SIGTERM)
    _, alive = psutil.wait_procs(processes.values(), timeout=5)
    signal_groups(alive, signal.SIGKILL)
    psutil.wait_procs(alive, timeout=5)
    left = running_in(root)
    if left:
        raise SystemExit(f"processes still running after SIGKILL: {[p.pid for p in left]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, required=True)
    parser.add_argument("--mib", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--timeout", type=float, default=7200, help="seconds allowed for the run")
    args = parser.parse_args()

    if not (BIN / "flower-superlink").exists():
        raise SystemExit(f"no flower-superlink next to {sys.executable}, use the venv's python")
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda number, _: sys.exit(128 + number))

    root = Path(tempfile.mkdtemp(prefix="flower-bench-", dir=args.workdir)).resolve()
    started = []

    def spawn(name, *command):
        # Every part gets its own FLWR_HOME, as it would on its own machine.
        home = root / name
        home.mkdir(exist_ok=True)
        with open(root / f"{name}.log", "ab") as log:
            process = subprocess.Popen(
                [str(c) for c in command], env=env_for(home), cwd=home,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        started.append(process.pid)
        return process

    def env_for(home):
        return {
            **os.environ,
            "PATH": f"{BIN}{os.pathsep}{os.environ['PATH']}",
            "PYTHONPATH": str(BENCHMARKS),
            "FLWR_HOME": str(home),
            "FLWR_TELEMETRY_ENABLED": "0",
            "FLWR_DISABLE_UPDATE_CHECK": "1",
        }

    try:
        link_port, *node_ports = free_ports(args.sites + 1)
        address = f"127.0.0.1:{link_port}"
        (root / "cli").mkdir()
        (root / "cli" / "config.toml").write_text(
            f'[superlink]\ndefault = "{LINK}"\n\n'
            f'[superlink.{LINK}]\naddress = "{address}"\ninsecure = true\n'
        )
        cli_env = env_for(root / "cli")

        startup_started = time.monotonic()
        superlink = spawn(
            "superlink", BIN / "flower-superlink", "--insecure", "--host", "127.0.0.1",
            "--port", link_port, "--disable-runtime-dependency-installation",
        )
        supernodes = [
            spawn(
                f"site-{i}", BIN / "flower-supernode", "--insecure", "--superlink", address,
                "--host", "127.0.0.1", "--port", port, "--node-config", f"site={i}",
            )
            for i, port in enumerate(node_ports)
        ]
        wait_ready(cli_env, superlink, supernodes, args.sites)
        startup_seconds = time.monotonic() - startup_started
        if superlink.pid not in {p.pid for p in running_in(root)}:
            raise SystemExit("cannot find our own processes by working directory")

        groups = {"server": [superlink.pid], "sites": [p.pid for p in supernodes]}
        idle = {
            name: round(sum(tree_rss(p) for p in pids) / (1 << 20), 1)
            for name, pids in groups.items()
        }
        roots = {"server": superlink.pid, "site_0": supernodes[0].pid}
        seen = {name: {} for name in roots}
        result_file = root / "result.json"
        run_log = root / "cli.log"
        run_config = (
            f"sites={args.sites} mib={args.mib} rounds={args.rounds} "
            f'result-file="{result_file}"'
        )
        submit_seconds = None
        next_look = 0.0
        with Sampler(groups) as sampler:
            timer = time.monotonic()
            run = spawn(
                "cli", BIN / "flwr", "run", APP, LINK, "--run-config", run_config, "--stream",
            )
            while run.poll() is None:
                now = time.monotonic()
                if now - timer > args.timeout:
                    raise SystemExit(f"run not finished after {args.timeout} s")
                check_alive([superlink, *supernodes])
                if submit_seconds is None and b"Successfully started run" in run_log.read_bytes():
                    submit_seconds = now - timer
                if now >= next_look:
                    next_look = now + 0.25
                    note_members(seen, roots)
                time.sleep(0.05)
            observed = time.monotonic() - timer
            exited_at = time.time()
        if run.returncode != 0:
            raise SystemExit(f"flwr run exited with {run.returncode}")

        match = re.search(rb"Successfully started run (\d+)", run_log.read_bytes())
        run_id = match.group(1).decode()
        reply = run_flwr_json(cli_env, "ls", LINK, "--run-id", run_id)
        if reply is None:
            raise SystemExit(f"could not read the status of run {run_id}")
        status = reply["runs"][0]
        if status["status"] != "finished:completed":
            raise SystemExit(f"run ended as {status['status']}: {status['status-details']}")

        # The memory groups are only right if the apps really ran inside these trees.
        if "flwr-serverapp" not in seen["server"] or "flwr-clientapp" not in seen["site_0"]:
            raise SystemExit(f"apps not found in the process trees: {seen}")

        summary = json.loads(result_file.read_text())
        model_error = check_model(summary, args.sites, args.mib, args.rounds)
        write_result(
            args.out,
            framework="flower",
            version=importlib.metadata.version("flwr"),
            sites=args.sites,
            model_mib=args.mib,
            rounds=args.rounds,
            total_seconds=round(observed - summary["check_seconds"], 2),
            memory=sampler.mib(),
            tls=False,
            masked=False,
            startup_seconds=round(startup_seconds, 2),
            submit_seconds=round(submit_seconds, 2),
            check_seconds=round(summary["check_seconds"], 3),
            strategy_seconds=round(summary["strategy_seconds"], 2),
            after_strategy_seconds=round(exited_at - summary["strategy_finished"], 2),
            superlink_elapsed_seconds=round(status["elapsed"], 2),
            network_bytes=status["network-traffic"],
            nominal_bytes=2 * args.sites * args.rounds * args.mib * (1 << 20),
            idle_rss_mib=idle,
            process_counts={
                name: {script: len(pids) for script, pids in scripts.items()}
                for name, scripts in seen.items()
            },
            model_error=model_error,
            notes=NOTES,
        )
    except BaseException:
        for log in sorted(root.glob("*.log")):
            print(f"--- {log.name} (tail)", file=sys.stderr)
            lines = log.read_text(errors="replace").splitlines()
            print("\n".join(lines[-30:]), file=sys.stderr)
        raise
    finally:
        try:
            stop_all(started, root)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
