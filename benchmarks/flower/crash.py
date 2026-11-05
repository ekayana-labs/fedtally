"""What a SuperLink crash does to a Flower run: kill it in round 2, restart it, watch."""

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import psutil

from run import (
    APP,
    BENCHMARKS,
    BIN,
    LINK,
    check_model,
    free_ports,
    running_in,
    script_name,
    stop_all,
    tree,
    wait_ready,
)

from common import write_result  # isort: skip

ALL_TRIALS = ["memory:superlink", "memory:tree", "sqlite:superlink", "sqlite:tree"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP_TIMEOUT = 240
NODE_GRACE = 30
STUCK_LINES = 6
LOG_LINES = 12

COMMANDS = {
    "superlink": "flower-superlink --insecure --host 127.0.0.1 --port PORT "
    "--disable-runtime-dependency-installation [--database ROOT/state.db]",
    "superexec_with_isolation_process": "flower-superexec --insecure --runtime-api-address "
    "127.0.0.1:PORT",
    "supernode": "flower-supernode --insecure --superlink 127.0.0.1:PORT --host 127.0.0.1 "
    "--port NODE_PORT --node-config site=I",
    "submit": f"flwr run APP {LINK} --run-config 'sites=4 mib=32 rounds=4 result-file=\"...\"' "
    "--stream",
    "run_status": f"flwr ls {LINK} --run-id RUN_ID --format json",
    "all_runs": f"flwr ls {LINK} --format json",
    "nodes": f"flwr supernode ls {LINK} --format json",
    "logs": f"flwr log RUN_ID {LINK} --show",
}

NOTES = [
    "SIGKILL in round 2: at least one SuperNode had received its round 2 message, see at_kill.",
    "The killed SuperLink is reaped at once, as init or a shell would, so its pid is gone.",
    "Restart is outage_seconds after the kill, with the same flags, port and database file.",
    "With process isolation, kill `tree` also kills and restarts the separate SuperExec.",
    "Watching stops early only once the run is finished and every SuperNode is online.",
    "Resubmit is a new `flwr run` of the same app after watching, to the same federation. If "
    f"SuperNodes have exited it waits {NODE_GRACE} s, then restarts them with the same flags.",
    "No Flower code changed. Stock FedAvg, with no checkpointing added to the app.",
]


def clean(text):
    return ANSI.sub("", text).strip()


def flwr(env, *args):
    try:
        done = subprocess.run(
            [str(BIN / "flwr"), *map(str, args)],
            env=env, cwd=env["FLWR_HOME"], capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return {"exit_code": None, "stdout": "", "stderr": "timed out after 60 s"}
    return {
        "exit_code": done.returncode,
        "stdout": clean(done.stdout),
        "stderr": clean(done.stderr),
    }


def parsed(reply):
    try:
        return json.loads(reply["stdout"])
    except ValueError:
        return {"success": False, "error-message": f"{reply['stdout']} {reply['stderr']}".strip()}


def run_status(env, run_id):
    reply = parsed(flwr(env, "ls", LINK, "--run-id", run_id, "--format", "json"))
    if not reply.get("runs"):
        return {"status": "ERROR", "text": reply.get("error-message", str(reply))[:300]}
    run = reply["runs"][0]
    return {"status": run["status"], "text": run["status-details"]}


def node_view(env):
    reply = parsed(flwr(env, "supernode", "ls", LINK, "--format", "json"))
    if not reply.get("success"):
        return {"error": reply.get("error-message", str(reply))[:300]}
    return {"ids": sorted(n["node-id"] for n in reply["nodes"]),
            "statuses": dict(Counter(n["status"] for n in reply["nodes"]))}


def processes(root):
    found = Counter()
    for process in running_in(root):
        script = script_name(process)
        if script:
            found[script] += 1
    return dict(sorted(found.items()))


def count_in(path, text):
    return path.read_text(errors="replace").count(text) if path.exists() else 0


def rounds_in(text):
    return sorted({int(r) for r in re.findall(r"\[ROUND (\d+)/", text)})


def text_since(path, offset):
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        f.seek(offset)
        return clean(f.read().decode(errors="replace"))


def lines_since(path, offset):
    """New log lines since the offset, without access lines, first and last few if long."""
    lines = text_since(path, offset).splitlines()
    lines = [clean(line)[:220] for line in lines if line.strip() and '"POST /' not in line]
    if len(lines) > 2 * LOG_LINES:
        skipped = f"... {len(lines) - 2 * LOG_LINES} lines ..."
        return [*lines[:LOG_LINES], skipped, *lines[-LOG_LINES:]]
    return lines


def model_result(path, args):
    if not path.exists():
        return "no final model written"
    try:
        error = check_model(json.loads(path.read_text()), args.sites, args.mib, args.rounds)
    except SystemExit as err:
        return f"wrong: {err}"
    return f"right, largest error {error:.2e}"


def trial(args, database, kill, workdir):
    root = Path(tempfile.mkdtemp(prefix="flower-crash-", dir=workdir)).resolve()
    started = []
    record = {
        "database": database,
        "kill": kill,
        "isolation": args.isolation,
        "outage_seconds": args.outage,
        "load_average": os.getloadavg()[0],
    }

    def env_for(home):
        return {
            **os.environ,
            "PATH": f"{BIN}{os.pathsep}{os.environ['PATH']}",
            "PYTHONPATH": str(BENCHMARKS),
            "PYTHONUNBUFFERED": "1",
            "FLWR_HOME": str(home),
            "FLWR_TELEMETRY_ENABLED": "0",
            "FLWR_DISABLE_UPDATE_CHECK": "1",
        }

    def spawn(name, *command):
        home = root / name
        home.mkdir(exist_ok=True)
        with open(root / f"{name}.log", "ab") as log:
            process = subprocess.Popen(
                [str(c) for c in command], env=env_for(home), cwd=home,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        started.append(process.pid)
        return process

    def run_config(name):
        return (
            f"sites={args.sites} mib={args.mib} rounds={args.rounds} "
            f'result-file="{root / name}"'
        )

    try:
        link_port, *node_ports = free_ports(args.sites + 1)
        address = f"127.0.0.1:{link_port}"
        (root / "cli").mkdir()
        (root / "cli" / "config.toml").write_text(
            f'[superlink]\ndefault = "{LINK}"\n\n'
            f'[superlink.{LINK}]\naddress = "{address}"\ninsecure = true\n'
        )
        env = env_for(root / "cli")

        link_command = [
            BIN / "flower-superlink", "--insecure", "--host", "127.0.0.1", "--port", link_port,
            "--disable-runtime-dependency-installation",
        ]
        if database == "sqlite":
            link_command += ["--database", root / "state.db"]
        if args.isolation == "process":
            link_command += ["--isolation", "process"]
        superlink = spawn("superlink", *link_command)

        def start_superexec():
            return spawn(
                "superexec", BIN / "flower-superexec", "--insecure",
                "--runtime-api-address", address,
            )

        # With process isolation the ServerApp's SuperExec is not the SuperLink's child.
        separate = args.isolation == "process"
        superexec = start_superexec() if separate else superlink

        def start_supernode(i):
            return spawn(
                f"site-{i}", BIN / "flower-supernode", "--insecure", "--superlink", address,
                "--host", "127.0.0.1", "--port", node_ports[i], "--node-config", f"site={i}",
            )

        supernodes = [start_supernode(i) for i in range(args.sites)]
        wait_ready(env, superexec, supernodes, args.sites)
        nodes_before = node_view(env)
        record["nodes_before_kill"] = nodes_before["statuses"]

        submit_log = root / "cli.log"
        run = spawn(
            "cli", BIN / "flwr", "run", APP, LINK, "--run-config", run_config("result.json"),
            "--stream",
        )
        site_logs = [root / f"site-{i}.log" for i in range(args.sites)]
        deadline = time.monotonic() + STEP_TIMEOUT
        while b"[ROUND 2/" not in submit_log.read_bytes():
            if time.monotonic() > deadline or run.poll() is not None:
                raise SystemExit(f"round 2 never started: {lines_since(submit_log, 0)}")
            time.sleep(0.1)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if any(count_in(log, "Received successfully") >= 2 for log in site_logs):
                break
            time.sleep(0.1)

        match = re.search(r"Successfully started run (\d+)", text_since(submit_log, 0))
        run_id = match.group(1)
        record["run_id"] = run_id
        victims = [psutil.Process(superlink.pid)]
        if kill == "tree":
            victims = tree(superlink.pid) + (tree(superexec.pid) if separate else [])
        record["at_kill"] = {
            "processes": processes(root),
            "killed": [script_name(v) for v in victims],
            "rounds_logged": rounds_in(submit_log.read_text(errors="replace")),
            "rounds_aggregated": count_in(submit_log, "aggregate_train: Received"),
            "received_by_site": [count_in(log, "Received successfully") for log in site_logs],
            "replied_by_site": [count_in(log, "Sent successfully") for log in site_logs],
        }
        offsets = {log.name: log.stat().st_size for log in root.glob("*.log")}
        killed_at = time.monotonic()
        for victim in victims:
            victim.kill()
        superlink.wait()
        if separate and kill == "tree":
            superexec.wait()

        time.sleep(args.outage)
        record["before_restart"] = {
            "seconds_after_kill": round(time.monotonic() - killed_at, 1),
            "processes": processes(root),
            "flwr_run_exit_code": run.poll(),
        }
        superlink = spawn("superlink", *link_command)
        if separate and kill == "tree":
            superexec = start_superexec()
        restarted_at = time.monotonic()

        timeline = []
        last = None
        cli_exit_seen = None
        while time.monotonic() - restarted_at < args.observe:
            status = run_status(env, run_id)
            nodes = node_view(env)
            if cli_exit_seen is None and run.poll() is not None:
                cli_exit_seen = round(time.monotonic() - killed_at, 1)
            state = {
                "run": status,
                "nodes": nodes.get("statuses", nodes.get("error")),
                "node_ids_kept": nodes.get("ids") == nodes_before["ids"],
                "processes": processes(root),
                "superlink_alive": superlink.poll() is None,
                "supernodes_alive": sum(p.poll() is None for p in supernodes),
            }
            if state != last:
                seconds = round(time.monotonic() - killed_at, 1)
                timeline.append({"seconds_after_kill": seconds, **state})
                last = state
            online = nodes.get("statuses", {}).get("online")
            if status["status"].startswith("finished") and online == args.sites:
                break
            time.sleep(args.poll)
        record["watched_seconds_after_restart"] = round(time.monotonic() - restarted_at, 1)
        record["timeline"] = timeline

        logs = flwr(env, "log", run_id, LINK, "--show")
        all_runs = parsed(flwr(env, "ls", LINK, "--format", "json"))
        record["after_watching"] = {
            "flwr_ls_run": parsed(flwr(env, "ls", LINK, "--run-id", run_id, "--format", "json")),
            "flwr_ls_all_run_ids": [r["run-id"] for r in all_runs.get("runs", [])],
            "flwr_supernode_ls": node_view(env).get("statuses"),
            "flwr_log_show": {
                "exit_code": logs["exit_code"],
                "rounds": rounds_in(logs["stdout"]),
                "rounds_aggregated": logs["stdout"].count("aggregate_train: Received"),
                "error": logs["stderr"][:300],
            },
            "model": model_result(root / "result.json", args),
            "supernode_exit_codes": [p.poll() for p in supernodes],
        }
        record["flwr_run"] = {
            "exit_code": run.poll(),
            "exit_seen_seconds_after_kill": cli_exit_seen,
            "still_running_after_watching": run.poll() is None,
            "output_after_kill": lines_since(submit_log, offsets["cli.log"]),
        }
        if run.poll() is None:
            run.kill()
            run.wait()
        after_kill = [text_since(log, offsets[log.name]) for log in site_logs]
        record["after_kill_counts"] = {
            "superlink_tasks_started": text_since(
                root / "superlink.log", offsets["superlink.log"]
            ).count("Started task"),
            "clientapps_started_by_site": [t.count("Start `flwr-clientapp`") for t in after_kill],
            "messages_received_by_site": [t.count("Received successfully") for t in after_kill],
            "replies_sent_by_site": [t.count("Sent successfully") for t in after_kill],
        }
        record["logs_after_kill"] = {
            "site-0": lines_since(root / "site-0.log", offsets["site-0.log"]),
            "superlink": lines_since(root / "superlink.log", offsets["superlink.log"]),
            "superexec": lines_since(root / "superexec.log", offsets.get("superexec.log", 0)),
        }

        # Is the federation still usable, and where does a new run start?
        offset = submit_log.stat().st_size
        site_offsets = [log.stat().st_size if log.exists() else 0 for log in site_logs]
        again_at = time.monotonic()
        again = spawn(
            "cli", BIN / "flwr", "run", APP, LINK, "--run-config", run_config("result-2.json"),
            "--stream",
        )
        waiting = None
        while again.poll() is None and time.monotonic() - again_at < args.resubmit_timeout:
            dead = [i for i, p in enumerate(supernodes) if p.poll() is not None]
            if dead and waiting is None and time.monotonic() - again_at > NODE_GRACE:
                new_run = re.search(r"started run (\d+)", text_since(submit_log, offset))
                waiting = {
                    "seconds": round(time.monotonic() - again_at, 1),
                    "run": run_status(env, new_run.group(1)),
                    "nodes": node_view(env).get("statuses"),
                    "last_output": lines_since(submit_log, offset)[-1],
                    "supernodes_restarted": len(dead),
                }
                for i in dead:
                    supernodes[i] = start_supernode(i)
            time.sleep(0.2)
        output = text_since(submit_log, offset)
        match = re.search(r"Successfully started run (\d+)", output)
        record["resubmit"] = {
            "before_restarting_supernodes": waiting,
            "finished_in_time": again.poll() is not None,
            "seconds": round(time.monotonic() - again_at, 1),
            "exit_code": again.poll(),
            "status": run_status(env, match.group(1)) if match else None,
            "rounds_logged": rounds_in(output),
            "model": model_result(root / "result-2.json", args),
            "nodes": node_view(env).get("statuses"),
            "supernodes_alive": sum(p.poll() is None for p in supernodes),
            "output_tail": [] if again.poll() == 0 else lines_since(submit_log, offset)[-6:],
        }
        if again.poll() is None:
            record["resubmit"]["when_stuck"] = {
                "processes": processes(root),
                "clientapp_started_by_site": [
                    text_since(log, o).count("Start `flwr-clientapp`")
                    for log, o in zip(site_logs, site_offsets)
                ],
                "new_lines_by_site": [
                    lines_since(log, o)[-STUCK_LINES:] for log, o in zip(site_logs, site_offsets)
                ],
                "last_lines_by_site": [lines_since(log, 0)[-STUCK_LINES:] for log in site_logs],
                "superlink_errors": [
                    line[:220]
                    for line in text_since(root / "superlink.log", 0).splitlines()
                    if line.startswith("ERROR") and "list-runs" not in line
                ][-12:],
            }
        if database == "sqlite":
            record["database_bytes"] = (root / "state.db").stat().st_size
    except (Exception, SystemExit) as err:
        record["harness_error"] = f"{type(err).__name__}: {err}"
        for log in sorted(root.glob("*.log")):
            record.setdefault("logs_on_error", {})[log.name] = lines_since(log, 0)[-10:]
    finally:
        try:
            stop_all(started, root)
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=4)
    parser.add_argument("--mib", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--observe", type=float, default=180, help="seconds to watch after restart")
    parser.add_argument("--outage", type=float, default=2, help="seconds between kill and restart")
    parser.add_argument("--poll", type=float, default=5)
    parser.add_argument("--resubmit-timeout", type=float, default=STEP_TIMEOUT)
    parser.add_argument("--isolation", choices=["subprocess", "process"], default="subprocess")
    parser.add_argument("--trial", action="append", choices=ALL_TRIALS, help="default: all four")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()

    if not (BIN / "flower-superlink").exists():
        raise SystemExit(f"no flower-superlink next to {sys.executable}, use the venv's python")
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda number, _: sys.exit(128 + number))

    trials = []
    for name in args.trial or ALL_TRIALS:
        database, kill = name.split(":")
        trials.append(trial(args, database, kill, args.workdir))
        write_result(
            args.out,
            framework="flower",
            version=importlib.metadata.version("flwr"),
            sites=args.sites,
            model_mib=args.mib,
            rounds=args.rounds,
            observe_seconds=args.observe,
            outage_seconds=args.outage,
            isolation=args.isolation,
            commands=COMMANDS,
            notes=NOTES,
            trials=trials,
        )


if __name__ == "__main__":
    main()
