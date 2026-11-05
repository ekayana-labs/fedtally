"""What NVFlare does when the server is killed during a job. One POC deployment per trial."""

import argparse
import contextlib
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
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import MODEL_TOLERANCE, model_error, write_result  # noqa: E402

import nvflare  # noqa: E402
from nvflare.apis.dxo import DataKind  # noqa: E402
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import (  # noqa: E402
    InTimeAccumulateWeightedAggregator,
)
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import (  # noqa: E402
    FullModelShareableGenerator,
)
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather  # noqa: E402
from nvflare.fuel.flare_api.flare_api import new_secure_session  # noqa: E402
from nvflare.fuel.utils.constants import FrameworkType  # noqa: E402
from nvflare.job_config.api import FedJob  # noqa: E402
from nvflare.job_config.script_runner import ScriptRunner  # noqa: E402
from nvflare.recipe.fedavg import FedAvgRecipe  # noqa: E402
from nvflare.tool.poc.poc_commands import POC_DEFAULT_FED_LEARN_PORT  # noqa: E402

from checkpoint_persistor import CheckpointPersistor  # noqa: E402
from persistor import MODEL_FILE, DictModelPersistor  # noqa: E402
from run import ADMIN, ours, poc, stop_everything  # noqa: E402

JOB_NAME = "crash-bench"
KILL_ROUND = 2
RESTART_GRACE = 20
TRIALS = {
    "server_parent": {"workflow": "fedavg", "kill": ["parent"]},
    "server_tree": {"workflow": "fedavg", "kill": ["parent", "job"]},
    "server_tree_and_supervisor": {"workflow": "fedavg", "kill": ["supervisor", "parent", "job"]},
    "snapshot_setting": {"workflow": "scatter_and_gather", "kill": ["parent", "job"]},
}
KINDS = {
    "sub_start.sh": "supervisor",
    "server_train": "parent",
    "client_train": "parent",
    "runner_process": "job",
    "worker_process": "job",
}
STAMP = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})")
OF_INTEREST = re.compile(
    r"abort|joined|re-?regist|token|heartbeat|dead|recover|restart|replac|restor|snapshot|missing|not running|"
    r"not found|error|warning|fail",
    re.IGNORECASE,
)


def export_job(directory, name, args, workflow, persistor, rounds, start_round=0):
    common_py = str(HERE.parent / "common.py")
    if workflow == "fedavg":
        recipe = FedAvgRecipe(
            name=name,
            min_clients=args.sites,
            num_rounds=rounds,
            train_script=str(HERE / "train.py"),
            framework=FrameworkType.NUMPY,
            model_persistor=persistor,
        )
        recipe.add_client_file(common_py)
        recipe.add_server_file(common_py)
        recipe.export(str(directory))
    else:
        job = FedJob(name=name, min_clients=args.sites)
        job.to_server(
            ScatterAndGather(
                min_clients=args.sites,
                num_rounds=rounds,
                wait_time_after_min_received=0,
                aggregator_id="aggregator",
                persistor_id="persistor",
                shareable_generator_id="shareable_generator",
                persist_every_n_rounds=1,
                snapshot_every_n_rounds=1,
            )
        )
        job.to_server(InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS), id="aggregator")
        job.to_server(FullModelShareableGenerator(), id="shareable_generator")
        job.to_server(persistor, id="persistor")
        job.to_server(common_py)
        job.to_clients(ScriptRunner(script=str(HERE / "train.py"), framework=FrameworkType.NUMPY))
        job.to_clients(common_py)
        job.export_job(str(directory))
    # neither job API passes start_round on, so it goes into the exported config
    if start_round:
        config_file = next(Path(directory).rglob("config_fed_server.json"))
        config = json.loads(config_file.read_text())
        config["workflows"][0]["args"]["start_round"] = start_round
        config_file.write_text(json.dumps(config, indent=2))
    return str(Path(directory) / name)


def processes(root):
    """pid to a label such as 'site-1 job', for everything of this run."""
    found = {}
    for process in ours(root):
        with contextlib.suppress(psutil.Error):
            command = " ".join(process.cmdline())
            site = re.search(r"prod_00/([^/\s]+)", command)
            kind = next((kind for name, kind in KINDS.items() if name in command), "other")
            job = re.search(r" -n ([0-9a-f-]{36})", command)
            label = f"{site.group(1) if site else '?'} {kind}"
            found[process.pid] = f"{label} (job {job.group(1)[:8]})" if job else label
    return found


def status_text(error):
    return f"{type(error).__name__}: {error}"


def watch(root, kit, job_id, sites, started, wait, out, seen, on_tick):
    """Records every change in processes, job status and connected clients until the wait is over."""
    session, last_status, last_clients, last_jobs = None, None, None, None
    settled = None
    reconnected = None
    while time.monotonic() - started < wait:
        now = time.monotonic() - started
        current = processes(root)
        on_tick(now, current)
        for pid, label in current.items():
            if pid not in seen:
                out["events"].append({"t": round(now, 1), "event": f"{label} pid {pid} running"})
        for pid, label in seen.items():
            if pid not in current:
                out["events"].append({"t": round(now, 1), "event": f"{label} pid {pid} gone"})
        seen = current

        clients, jobs = None, None
        try:
            if session is None:
                session = new_secure_session(ADMIN, kit, timeout=5)
            status = session.get_job_meta(job_id)["status"]
            info = session.get_system_info()
            clients = sorted(client.name for client in info.client_info)
            jobs = sorted(job.job_id for job in info.job_info)
        except Exception as e:
            status = status_text(e)
            if session is not None:
                with contextlib.suppress(Exception):
                    session.close()
            session = None
        if status != last_status:
            out["status_timeline"].append({"t": round(time.monotonic() - started, 1), "status": status})
            last_status = status
        if clients is not None and clients != last_clients:
            out["events"].append({"t": round(now, 1), "event": f"connected clients per server: {clients}"})
            last_clients = clients
            if reconnected is None and set(sites) <= set(clients):
                reconnected = round(now, 1)
        if jobs is not None and jobs != last_jobs:
            out["events"].append({"t": round(now, 1), "event": f"running jobs per server: {[j[:8] for j in jobs]}"})
            last_jobs = jobs

        job_alive = any(job_id[:8] in label for label in current.values())
        if status.startswith("FINISHED") and not job_alive:
            settled = settled if settled is not None else now
            if now - settled >= 10:
                break
        time.sleep(1)
    out["clients_all_connected_at"] = reconnected
    return session


def timed_lines(path):
    if not path.exists():
        return
    current = None
    for line in path.read_text(errors="replace").splitlines():
        stamp = STAMP.match(line)
        if stamp:
            current = datetime.strptime(stamp.group(1), "%Y-%m-%d %H:%M:%S,%f")
        yield current, line


def rounds_started(log, since=None):
    rounds = []
    for stamp, line in timed_lines(log):
        match = re.search(r"Round (\d+) started\.", line)
        if match and stamp and (since is None or stamp > since):
            rounds.append(int(match.group(1)))
    return rounds


def count_results(log):
    return sum("Starting to push execute result" in line for _, line in timed_lines(log))


def log_after(path, since, root, limit=30):
    """Lines of interest after the kill, each distinct message once with how often it came."""
    messages = {}
    for stamp, line in timed_lines(path):
        if stamp is None or stamp <= since or not OF_INTEREST.search(line):
            continue
        message = STAMP.sub("", line, count=1).strip().replace(str(root), "<run>")[:300]
        key = re.sub(r"\b\d+\b", "N", message)
        entry = messages.setdefault(key, {"first": stamp.strftime("%H:%M:%S.%f")[:-3], "text": message, "count": 0})
        entry["count"] += 1
    return list(messages.values())[:limit]


def follow_up(session, job_folder, args, expected_rounds, timeout):
    started = time.monotonic()
    out = {}
    try:
        job_id = session.submit_job(job_folder)
        out["job_id"] = job_id
        _, meta = session.monitor_job_and_return_job_meta(job_id, timeout=timeout, poll_interval=1)
        out["seconds"] = round(time.monotonic() - started, 1)
        out["status"] = meta["status"] if meta else f"not finished within {timeout} s"
        if meta and meta["status"] == "FINISHED:COMPLETED":
            result = Path(session.download_job_result(job_id))
            with np.load(next(result.rglob(MODEL_FILE))) as saved:
                out["model_error"] = model_error(dict(saved), args.mib, args.sites, expected_rounds)
            out["model_right"] = out["model_error"] <= MODEL_TOLERANCE
    except Exception as e:
        out["error"] = status_text(e)
    return out


def trial(name, plan, args):
    try:
        socket.create_connection(("localhost", POC_DEFAULT_FED_LEARN_PORT), timeout=2).close()
    except OSError:
        pass
    else:
        raise SystemExit(f"something is already listening on port {POC_DEFAULT_FED_LEARN_PORT}, which POC needs")

    root = Path(tempfile.mkdtemp(prefix="nvflare-bench-", dir=args.workdir)).resolve()
    (root / "home").mkdir()
    env = {**os.environ, "HOME": str(root / "home"), "NVFLARE_POC_WORKSPACE": str(root / "poc")}
    prod = root / "poc" / "example_project" / "prod_00"
    sites = [f"site-{i + 1}" for i in range(args.sites)]
    groups = []
    session = None
    out = {
        "trial": name,
        "workflow": plan["workflow"],
        "killed_roles": plan["kill"],
        "events": [],
        "status_timeline": [],
    }
    failed = True
    try:
        process, output = poc(env, "prepare", "-n", str(args.sites), "--force")
        if process.returncode != 0:
            raise SystemExit(f"nvflare poc prepare failed:\n{output}")
        process, output = poc(env, "start", "--timeout", "30", groups=groups, timeout=90)
        if process.returncode != 0:
            raise SystemExit(f"nvflare poc start failed:\n{output}")

        kit = prod / "server"
        admin_kit = str(prod / ADMIN)
        session = new_secure_session(ADMIN, admin_kit, timeout=60)
        job_folder = export_job(
            root / "job", JOB_NAME, args, plan["workflow"], DictModelPersistor(args.mib), args.rounds
        )
        job_id = session.submit_job(job_folder)
        out["job_id"] = job_id
        server_log = kit / job_id / "log.txt"
        site_logs = {site: prod / site / job_id / "log.txt" for site in sites}

        deadline = time.monotonic() + 120
        mark = f"Round {KILL_ROUND} started."
        while not (server_log.exists() and mark in server_log.read_text(errors="replace")):
            if time.monotonic() > deadline:
                raise SystemExit(f"never saw {mark!r}: job status {session.get_job_meta(job_id)['status']}")
            time.sleep(0.05)
        # long enough for the sites to pick up the task, so the kill lands inside the round
        time.sleep(0.5)

        out["status_before_kill"] = session.get_job_meta(job_id)["status"]
        out["rounds_started_before_kill"] = rounds_started(server_log)
        out["results_sent_by_site_before_kill"] = {site: count_results(log) for site, log in site_logs.items()}
        before = processes(root)
        out["processes_before_kill"] = sorted(before.values())

        roles = {
            role: [pid for pid, label in before.items() if label.startswith(f"server {role}")]
            for role in ("supervisor", "parent", "job")
        }
        out["daemon_pid_fl_content"] = (kit / "daemon_pid.fl").read_text()
        with contextlib.suppress(Exception):
            session.close()
        session = None

        out["killed"] = {role: roles[role] for role in plan["kill"]}
        killed_at = datetime.now()
        started = time.monotonic()
        for role in plan["kill"]:
            for pid in roles[role]:
                os.kill(pid, signal.SIGKILL)

        checkpoint = root / "checkpoint.npz"
        shutil.copyfile(kit / job_id / MODEL_FILE, checkpoint)

        restart_log = root / "start_sh.log"

        def run_start_sh():
            # python3 has to resolve to the same install
            start_env = {**env, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}"}
            with open(restart_log, "w") as log:
                starter = subprocess.Popen(
                    ["bash", str(kit / "startup" / "start.sh")],
                    cwd=kit,
                    env=start_env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            groups.append(starter.pid)
            out["start_sh_exit_code"] = starter.wait(timeout=30)
            out["restart_by"] = "startup/start.sh"
            out["start_sh_at"] = round(time.monotonic() - started, 1)

        # a live supervisor restarts the server by itself; start.sh next to it would start a second
        # supervisor where bash lacks $BASHPID, because its guard reads the empty daemon_pid.fl
        if "supervisor" in plan["kill"]:
            run_start_sh()

        def on_tick(now, current):
            back = "server parent" in current.values()
            if back and "server_parent_back_at" not in out:
                out["server_parent_back_at"] = round(now, 1)
                out.setdefault("restart_by", "sub_start.sh supervisor, already running")
            elif not back and now > RESTART_GRACE and "restart_by" not in out:
                run_start_sh()

        seen = {pid: label for pid, label in before.items()}
        session = watch(root, admin_kit, job_id, sites, started, args.wait, out, seen, on_tick)
        if session is None:
            session = new_secure_session(ADMIN, admin_kit, timeout=30)

        status = session.get_job_meta(job_id)["status"]
        out["status_after_wait"] = status
        out["seconds_watched"] = round(time.monotonic() - started, 1)
        out["job_finished"] = status.startswith("FINISHED")
        out["rounds_started_after_kill"] = rounds_started(server_log, since=killed_at)
        out["results_sent_by_site_after_wait"] = {site: count_results(log) for site, log in site_logs.items()}
        if restart_log.exists():
            out["start_sh_output"] = restart_log.read_text(errors="replace").replace(str(root), "<run>")[:600]
        out["processes_after_wait"] = sorted(processes(root).values())
        out["snapshot_storage_files"] = sorted(
            str(p.relative_to(root / "poc" / "snapshot-storage"))
            for p in (root / "poc" / "snapshot-storage").rglob("*")
            if p.is_file()
        )
        out["run_dir_survived_restart"] = (kit / job_id).is_dir()
        out["checkpoint_at_kill_error_vs_rounds_done"] = model_error(
            dict(np.load(checkpoint)), args.mib, args.sites, KILL_ROUND
        )
        out["logs_after_kill"] = {
            "server_console": log_after(kit / "poc_console.log", killed_at, root),
            "server_job": log_after(server_log, killed_at, root),
            "site-1_console": log_after(prod / "site-1" / "poc_console.log", killed_at, root),
            "site-1_job": log_after(site_logs["site-1"], killed_at, root),
        }

        out["recovered"] = False
        if status == "FINISHED:COMPLETED":
            result = Path(session.download_job_result(job_id))
            with np.load(next(result.rglob(MODEL_FILE))) as saved:
                out["final_model_error"] = model_error(dict(saved), args.mib, args.sites, args.rounds)
            out["recovered"] = out["final_model_error"] <= MODEL_TOLERANCE

        out["resubmit_from_start"] = follow_up(
            session,
            export_job(
                root / "again", JOB_NAME, args, plan["workflow"], DictModelPersistor(args.mib), args.rounds
            ),
            args,
            args.rounds,
            args.followup_timeout,
        )
        out["resume_from_checkpoint"] = follow_up(
            session,
            export_job(
                root / "resume",
                JOB_NAME,
                args,
                plan["workflow"],
                CheckpointPersistor(args.mib, str(checkpoint)),
                args.rounds - KILL_ROUND,
                start_round=KILL_ROUND,
            ),
            args,
            args.rounds,
            args.followup_timeout,
        )

        out["status_after_follow_ups"] = session.get_job_meta(job_id)["status"]
        if not out["status_after_follow_ups"].startswith("FINISHED"):
            try:
                out["abort_response"] = str(session.abort_job(job_id))
            except Exception as e:
                out["abort_response"] = status_text(e)
            time.sleep(5)
            out["status_after_abort"] = session.get_job_meta(job_id)["status"]
        entry = next((job for job in session.list_jobs() if job.get("job_id") == job_id), None)
        out["list_jobs_entry"] = entry
        try:
            out["delete_response"] = str(session.delete_job(job_id))
        except Exception as e:
            out["delete_response"] = status_text(e)
        failed = False
        return out
    except Exception as e:
        out["trial_error"] = status_text(e)
        return out
    finally:
        if session is not None:
            with contextlib.suppress(Exception):
                session.close()
        stop_everything(root, env, set(), groups)
        out["left_running_after_cleanup"] = len(ours(root))
        if args.keep or failed:
            print(f"kept {root}", file=sys.stderr)
        else:
            shutil.rmtree(root, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=4)
    parser.add_argument("--mib", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--wait", type=float, default=180, help="seconds to watch after the kill")
    parser.add_argument("--followup-timeout", type=float, default=120)
    parser.add_argument("--trial", action="append", choices=sorted(TRIALS), help="repeat; default is all of them")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--keep", action="store_true", help="keep the POC workspaces and logs")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    results = []
    for name in args.trial or TRIALS:
        results.append(trial(name, TRIALS[name], args))
        write_result(
            args.out,
            framework="nvflare",
            version=nvflare.__version__,
            sites=args.sites,
            model_mib=args.mib,
            rounds=args.rounds,
            kill_in_round=KILL_ROUND,
            wait_seconds=args.wait,
            settings="defaults, except the snapshot_setting trial, which runs ScatterAndGather",
            bash_version=subprocess.run(
                ["bash", "--version"], capture_output=True, text=True
            ).stdout.splitlines()[0],
            trials=results,
        )


if __name__ == "__main__":
    main()
