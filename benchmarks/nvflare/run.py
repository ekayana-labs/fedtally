"""One NVFlare FedAvg run on a POC deployment: server and sites as separate processes."""

import argparse
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import MODEL_TOLERANCE, Sampler, model_error, tree_rss, write_result  # noqa: E402

import nvflare  # noqa: E402
from nvflare.app_common.decomposers.numpy_decomposers import NumpyArrayDecomposer  # noqa: E402
from nvflare.fuel.f3.streaming.transfer_progress import DEFAULT_STREAMING_IDLE_TIMEOUT  # noqa: E402
from nvflare.fuel.flare_api.flare_api import new_secure_session  # noqa: E402
from nvflare.fuel.utils.constants import FrameworkType  # noqa: E402
from nvflare.recipe.fedavg import FedAvgRecipe  # noqa: E402
from nvflare.tool.poc.poc_commands import POC_DEFAULT_FED_LEARN_PORT  # noqa: E402

from persistor import MODEL_FILE, DictModelPersistor  # noqa: E402

ADMIN = "admin@nvidia.com"
JOB_NAME = "fedavg-bench"
POLL_SECONDS = 0.2
NOTES = [
    "POC deployment from nvflare poc prepare and start; job from FedAvgRecipe, submitted through the FLARE API",
    "total_seconds runs from before submit to the terminal job status, polled every 0.2s, so it includes job "
    "scheduling, deployment and teardown; fedavg_seconds is the Start to Finished span in the server job log",
    "POC start-up, result download and the model check are not timed",
    "the server job process builds the initial model through the persistor, inside total_seconds and "
    "fedavg_seconds; model_load_seconds is that step",
    "site job processes reach the server parent over mTLS; each job process to its own parent is local plain TCP",
    "the custom persistor stands in for NPModelPersistor, which holds a single array",
]


def poc(env, *command, groups=None, timeout=300):
    """Runs nvflare poc in its own process group, which the services it starts inherit."""
    process = subprocess.Popen(
        [sys.executable, "-m", "nvflare.cli", "poc", *command],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    if groups is not None:
        groups.append(process.pid)
    try:
        return process, process.communicate(timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        raise


def job_processes_of(pid):
    try:
        children = psutil.Process(pid).children(recursive=True)
    except psutil.Error:
        return []
    found = []
    for child in children:
        try:
            if any(name in part for part in child.cmdline() for name in ("runner_process", "worker_process")):
                found.append(child)
        except psutil.Error:
            pass
    return found


def ours(root):
    """Every process with this run's directory on its command line."""
    found = []
    for process in psutil.process_iter():
        try:
            if process.pid != os.getpid() and any(str(root) in part for part in process.cmdline()):
                found.append(process)
        except psutil.Error:
            pass
    return found


def stop_everything(root, env, known, groups):
    if (root / "poc" / "project.yml").exists():
        try:
            poc(env, "stop", timeout=90)
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"nvflare poc stop did not finish: {e}", file=sys.stderr)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except OSError:
            pass
    processes = known.union(ours(root))
    for process in processes:
        try:
            process.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(processes, timeout=10)
    left = ours(root)
    if left:
        raise SystemExit(f"processes still running after cleanup: {[p.pid for p in left]}")


def log_spans(result):
    """The workflow's own timing, from the server job log."""
    marks = {}
    for line in (result / "workspace" / "log.json").read_text().splitlines():
        record = json.loads(line)
        for mark in ("Start FedAvg.", "loading initial model", "Round 0 started.", "Finished FedAvg."):
            if mark in record.get("message", ""):
                marks[mark] = datetime.strptime(record["asctime"], "%Y-%m-%d %H:%M:%S,%f")

    def span(start, end):
        return round((marks[end] - marks[start]).total_seconds(), 3)

    return {
        "fedavg_seconds": span("Start FedAvg.", "Finished FedAvg."),
        "model_load_seconds": span("loading initial model", "Round 0 started."),
    }


def log_tails(prod, lines=40):
    for log in sorted(prod.rglob("log.txt")) + sorted(prod.rglob("poc_console.log")):
        print(f"--- {log}", file=sys.stderr)
        print("\n".join(log.read_text(errors="replace").splitlines()[-lines:]), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, required=True)
    parser.add_argument("--mib", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--timeout", type=float, default=1800, help="give up on the job after this many seconds")
    parser.add_argument("--keep", action="store_true", help="keep the POC workspace and logs")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    try:
        socket.create_connection(("localhost", POC_DEFAULT_FED_LEARN_PORT), timeout=2).close()
    except OSError:
        pass
    else:
        raise SystemExit(f"something is already listening on port {POC_DEFAULT_FED_LEARN_PORT}, which POC needs")

    root = Path(tempfile.mkdtemp(prefix="nvflare-bench-", dir=args.workdir)).resolve()
    # poc prepare registers its startup kits under $HOME/.nvflare, so HOME stays in the run directory too
    (root / "home").mkdir()
    env = {**os.environ, "HOME": str(root / "home"), "NVFLARE_POC_WORKSPACE": str(root / "poc")}
    prod = root / "poc" / "example_project" / "prod_00"
    sites = [f"site-{i + 1}" for i in range(args.sites)]
    groups, known = [], set()
    session = None
    failed = True
    try:
        process, output = poc(env, "prepare", "-n", str(args.sites), "--force")
        if process.returncode != 0:
            raise SystemExit(f"nvflare poc prepare failed:\n{output}")

        # the CLI reports ready within seconds but only exits when this timeout runs out
        wait = max(30, 5 * args.sites)
        process, output = poc(env, "start", "--timeout", str(wait), groups=groups, timeout=wait + 60)
        if process.returncode != 0:
            raise SystemExit(f"nvflare poc start failed:\n{output}")

        server_pid = int((prod / "server" / "pid.fl").read_text())
        site_pids = [int((prod / site / "pid.fl").read_text()) for site in sites]
        known.update(psutil.Process(pid) for pid in [server_pid, *site_pids])

        server_kit = json.loads((prod / "server" / "startup" / "fed_server.json").read_text())["servers"][0]
        security = {server_kit["connection_security"]} | {
            json.loads((prod / site / "startup" / "fed_client.json").read_text())["client"]["connection_security"]
            for site in sites
        }

        recipe = FedAvgRecipe(
            name=JOB_NAME,
            min_clients=args.sites,
            num_rounds=args.rounds,
            train_script=str(HERE / "train.py"),
            framework=FrameworkType.NUMPY,
            model_persistor=DictModelPersistor(args.mib),
        )
        recipe.add_client_file(str(HERE.parent / "common.py"))
        recipe.add_server_file(str(HERE.parent / "common.py"))
        recipe.export(str(root / "job"))
        decomposer = NumpyArrayDecomposer()

        session = new_secure_session(ADMIN, str(prod / ADMIN), timeout=60)
        seen = {pid: set() for pid in [server_pid, *site_pids]}

        def watch(_session, _job_id, _meta):
            for pid, found in seen.items():
                found.update(job_processes_of(pid))
                known.update(found)
            return True

        idle = {"server": tree_rss(server_pid), "sites": sum(tree_rss(pid) for pid in site_pids)}
        with Sampler({"server": [server_pid], "sites": list(site_pids)}) as sampler:
            started = time.monotonic()
            job_id = session.submit_job(str(root / "job" / JOB_NAME))
            _, meta = session.monitor_job_and_return_job_meta(
                job_id, timeout=args.timeout, poll_interval=POLL_SECONDS, cb=watch
            )
            total = time.monotonic() - started

        if meta is None:
            raise SystemExit(f"job {job_id} did not finish within {args.timeout} seconds")
        if meta["status"] != "FINISHED:COMPLETED":
            raise SystemExit(f"job {job_id} ended as {meta['status']}: {meta}")
        missing = [pid for pid, found in seen.items() if not found]
        if missing:
            raise SystemExit(f"no job process seen under {missing}, memory would be undercounted")

        result = Path(session.download_job_result(job_id))
        with np.load(next(result.rglob(MODEL_FILE))) as saved:
            error = model_error(dict(saved), args.mib, args.sites, args.rounds)
        if error > MODEL_TOLERANCE:
            raise SystemExit(f"the final model is off by {error}")

        write_result(
            args.out,
            framework="nvflare",
            version=nvflare.__version__,
            sites=args.sites,
            model_mib=args.mib,
            rounds=args.rounds,
            total_seconds=round(total, 2),
            memory=sampler.mib(),
            idle_rss_mib={name: round(rss / (1 << 20), 1) for name, rss in idle.items()},
            model_error=error,
            tls=security <= {"tls", "mtls"},
            masked=False,
            connection_security=sorted(security),
            transport=server_kit["service"]["scheme"],
            **log_spans(result),
            job_status=meta["status"],
            settings={
                "execution_mode": "external_process" if recipe.launch_external_process else "in_process",
                "server_expected_format": recipe.server_expected_format.value,
                "params_transfer_type": recipe.params_transfer_type.value,
                "enable_tensor_disk_offload": recipe.enable_tensor_disk_offload,
                "np_download_chunk_size": decomposer.max_chunk_size,
                "streaming_idle_timeout": DEFAULT_STREAMING_IDLE_TIMEOUT,
            },
            notes=NOTES,
        )
        failed = False
    except BaseException:
        log_tails(prod)
        raise
    finally:
        if session is not None:
            with contextlib.suppress(Exception):
                session.close()
        stop_everything(root, env, known, groups)
        if args.keep or failed:
            print(f"kept {root}", file=sys.stderr)
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
