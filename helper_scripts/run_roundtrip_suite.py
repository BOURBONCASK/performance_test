#!/usr/bin/env python3
"""Automate roundtrip latency sweeps for performance_test across message sizes."""

# python3 run_roundtrip_suite.py --env-script /app/script/env.sh --relay-env-script /app/script/env.sh --max-runtime 300 --relay-duration-margin 15 --relay-wait-seconds 6 --wait-for-matched-timeout 120 --retry-on-zero 2 --relay-debug

import argparse
import csv
import datetime as dt
import json
import math
import pathlib
import re
import shlex
import subprocess
import sys
import time
from typing import Iterable, List, Optional, Tuple


DEFAULT_MESSAGES = [
    "Array128",
    "Array256",
    "Array512",
    "Array1k",
    "Array4k",
    "Array16k",
    "Array32k",
    "Array60k",
    "Array64k",
    "Array256k",
    "Array1m",
    "Array2m",
    "Array4m",
]

BITS_PER_BYTE = 8
MAX_BANDWIDTH_BITS = 50_000_000  # 50 Mbps ceiling per requirements.


class ConfigError(Exception):
    """Raised when the provided configuration is invalid."""


def parse_arguments() -> argparse.Namespace:
    """Define and parse CLI arguments for the sweep runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run performance_test roundtrip experiments over a message matrix "
            "and capture logs in a structured directory."
        )
    )
    parser.add_argument(
        "--messages",
        nargs="+",
        default=DEFAULT_MESSAGES,
        help=(
            "Message types to sweep. Defaults to Array128..Array4m subset. "
            "Provide a whitespace-separated list."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="/userdata",
        type=pathlib.Path,
        help="Root directory for experiment outputs (default: /userdata).",
    )
    parser.add_argument(
        "--topic-prefix",
        default="roundtrip",
        help="Topic prefix shared with the relay node (default: roundtrip).",
    )
    parser.add_argument(
        "--roundtrip-mode",
        default="Main",
        choices=["Main", "Relay", "None"],
        help="Roundtrip mode for perf_test (default: Main).",
    )
    parser.add_argument(
        "--reliability",
        default="RELIABLE",
        choices=["RELIABLE", "BEST_EFFORT"],
        help="Set perf_test reliability policy (default: RELIABLE).",
    )
    parser.add_argument(
        "--max-runtime",
        default=30,
        type=int,
        help="Duration in seconds for each run (default: 30).",
    )
    parser.add_argument(
        "--ignore-seconds",
        default=2,
        type=int,
        help="Seconds to ignore at the start of each run (default: 2).",
    )
    parser.add_argument(
        "--wait-for-matched-timeout",
        default=60,
        type=int,
        help="Seconds to wait for publishers/subscribers to match (default: 60).",
    )
    parser.add_argument(
        "--env-script",
        default=None,
        help="Optional shell script to source before each run "
        "(e.g., /app/script/env.sh).",
    )
    parser.add_argument(
        "--log-format",
        choices=["json", "csv"],
        default="json",
        help="Perf_test logfile format (default: json).",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="Additional perf_test arguments, appended verbatim to each command.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    parser.add_argument(
        "--relay-host",
        default="192.168.127.2",
        help="Relay machine hostname or IP. Leave empty to disable automatic relay management.",
    )
    parser.add_argument(
        "--relay-user",
        default="root",
        help="SSH user for the relay machine (default: root).",
    )
    parser.add_argument(
        "--relay-env-script",
        default=None,
        help="Optional script on the relay host to source before launching perf_test.",
    )
    parser.add_argument(
        "--relay-debug",
        action="store_true",
        help="Enable 'set -x' tracing on the relay command.",
    )
    parser.add_argument(
        "--relay-extra-args",
        default="",
        help="Additional perf_test arguments for the relay side.",
    )
    parser.add_argument(
        "--relay-wait-seconds",
        default=5.0,
        type=float,
        help="Seconds to wait after starting the relay before launching the main run.",
    )
    parser.add_argument(
        "--relay-duration-margin",
        default=5,
        type=int,
        help="Extra seconds added to the relay max-runtime so it outlives the main run.",
    )
    parser.add_argument(
        "--disable-relay",
        action="store_true",
        help="Skip automatic relay management even if --relay-host is set.",
    )
    parser.add_argument(
        "--retry-on-zero",
        default=0,
        type=int,
        help="Number of times to retry a run if no samples are received (default: 0).",
    )
    return parser.parse_args()


def infer_size_bytes(msg_type: str) -> int:
    """Infer payload size in bytes based on message type naming."""
    match = re.match(r"^(?:Array|Struct|BoundedSequence|PointCloud)(\d+)([kKmM]?)$", msg_type)
    if not match:
        raise ConfigError(f"Cannot infer size for message type '{msg_type}'.")

    value = int(match.group(1))
    unit = match.group(2).lower()
    if unit == "k":
        return value * 1024
    if unit == "m":
        return value * 1024 * 1024
    return value


def pick_base_rate(size_bytes: int) -> int:
    """Return the baseline publish rate based on message size thresholds."""
    threshold_bytes = 16 * 1024
    return 100 if size_bytes <= threshold_bytes else 20


def clamp_rate_for_bandwidth(size_bytes: int, base_rate: int) -> int:
    """Ensure publish rate keeps bandwidth below the 50 Mbps ceiling."""
    if size_bytes <= 0:
        return base_rate

    max_rate = MAX_BANDWIDTH_BITS // (size_bytes * BITS_PER_BYTE)
    if max_rate <= 0:
        max_rate = 1
    return min(base_rate, max_rate)


def build_perf_test_command(
    msg_type: str,
    rate: int,
    args: argparse.Namespace,
    log_path: pathlib.Path,
    topic: str,
) -> List[str]:
    """Compose the perf_test CLI command."""
    cmd: List[str] = [
        "ros2",
        "run",
        "performance_test",
        "perf_test",
        "--msg",
        msg_type,
        "--rate",
        str(rate),
        "--max-runtime",
        str(args.max_runtime),
        "--ignore",
        str(args.ignore_seconds),
        "--reliability",
        args.reliability,
        "--wait-for-matched-timeout",
        str(args.wait_for_matched_timeout),
        "--roundtrip-mode",
        args.roundtrip_mode,
        "--topic",
        topic,
        "--logfile",
        str(log_path),
    ]

    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def build_relay_command(
    msg_type: str,
    args: argparse.Namespace,
    topic: str,
) -> List[str]:
    runtime = max(args.max_runtime + args.relay_duration_margin, args.max_runtime)
    cmd: List[str] = [
        "ros2",
        "run",
        "performance_test",
        "perf_test",
        "--msg",
        msg_type,
        "--roundtrip-mode",
        "Relay",
        "--topic",
        topic,
        "--reliability",
        args.reliability,
        "--max-runtime",
        str(runtime),
    ]
    if args.relay_extra_args:
        cmd.extend(shlex.split(args.relay_extra_args))
    return cmd


def run_command(cmd: List[str], env_script: Optional[str], dry_run: bool) -> int:
    """Execute perf_test command, optionally sourcing a setup script first."""
    if env_script:
        quoted_script = shlex.quote(env_script)
        command_str = " ".join(shlex.quote(part) for part in cmd)
        print(f"[debug] main env script: {env_script}")
        shell_cmd = f"source {quoted_script} && {command_str}"
        print(f"[command] bash -lc '{shell_cmd}'")
        if dry_run:
            return 0
        result = subprocess.run(["bash", "-lc", shell_cmd], check=False)
        return result.returncode

    print("[debug] main env script: <none>")
    print(f"[command] {' '.join(shlex.quote(part) for part in cmd)}")
    if dry_run:
        return 0
    result = subprocess.run(cmd, check=False)
    print(f"[debug] main command exit code: {result.returncode}")
    return result.returncode


def start_relay(
    msg_type: str,
    topic: str,
    args: argparse.Namespace,
) -> Optional[subprocess.Popen]:
    if args.disable_relay or not args.relay_host:
        return None

    relay_cmd = build_relay_command(msg_type, args, topic)
    relay_cmd_str = " ".join(shlex.quote(part) for part in relay_cmd)
    segments = []
    if getattr(args, "relay_debug", False):
        segments.append("set -x")
    if args.relay_env_script:
        segments.append(f"source {shlex.quote(args.relay_env_script)}")
    relay_cmd_str = " && ".join(segments + [relay_cmd_str]) if segments else relay_cmd_str

    remote_command = ["bash", "-lc", relay_cmd_str]
    target = f"{args.relay_user}@{args.relay_host}"
    ssh_cmd = ["ssh", "-o", "BatchMode=yes", target] + remote_command

    print(f"[debug] relay env script: {args.relay_env_script or '<none>'}")
    print(f"[relay] {' '.join(shlex.quote(part) for part in ssh_cmd)}")
    if args.dry_run:
        return None
    try:
        proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except OSError as exc:
        print(f"[error] Failed to start relay via SSH: {exc}", file=sys.stderr)
        return None

    time.sleep(max(args.relay_wait_seconds, 0.0))

    if proc.poll() is not None and proc.returncode is not None and proc.returncode != 0:
        print(
            f"[error] Relay process exited early with code {proc.returncode}. "
            "Check the relay logs.",
            file=sys.stderr,
        )
        return None

    if proc.stdout:
        print(f"[debug] relay stdout prefix: {msg_type}")
    return proc


def stop_relay(proc: Optional[subprocess.Popen], timeout: float = 5.0) -> None:
    if not proc:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if proc.stdout:
        output = proc.stdout.read()
        if output:
            print(f"[relay stdout]\n{output.strip()}")
    if proc.returncode not in (0, None):
        print(f"[relay] exited with code {proc.returncode}", file=sys.stderr)


def write_manifest(
    manifest_path: pathlib.Path, rows: Iterable[Tuple[int, str, int, int, pathlib.Path]]
) -> None:
    """Persist manifest CSV describing the executed experiments."""
    with manifest_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "message_type", "size_bytes", "rate_hz", "logfile"])
        for row in rows:
            writer.writerow(row)


def ensure_directory(path: pathlib.Path) -> None:
    """Create path (and parents) if missing."""
    path.mkdir(parents=True, exist_ok=True)


def count_latency_samples(log_path: pathlib.Path) -> int:
    """Return total latency samples recorded in the given log file."""
    if not log_path.exists():
        return 0

    suffix = log_path.suffix.lower()
    try:
        if suffix == ".json":
            with log_path.open("r") as handle:
                data = json.load(handle)
            return sum(
                entry.get("latency_n", 0) for entry in data.get("analysis_results", [])
            )
        if suffix == ".csv":
            # CSV logs contain multiple sections; scan for latency_n column.
            with log_path.open("r") as handle:
                for line in handle:
                    if line.startswith("---EXPERIMENT-START---"):
                        break
                reader = csv.DictReader(handle)
                total = 0
                for row in reader:
                    latency_n = row.get("latency_n")
                    if latency_n is not None:
                        try:
                            total += int(latency_n)
                        except ValueError:
                            continue
                return total
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] Failed to read log {log_path}: {exc}", file=sys.stderr)
    return 0


def main() -> int:
    args = parse_arguments()

    try:
        messages = [msg.strip() for msg in args.messages if msg.strip()]
        if not messages:
            raise ConfigError("No message types provided.")
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root / "perf_test_roundtrip" / timestamp
    ensure_directory(run_root)

    manifest_rows: List[Tuple[int, str, int, int, pathlib.Path]] = []

    print(f"Output directory: {run_root}")
    if args.disable_relay or not args.relay_host:
        print(
            "Reminder: start the relay node with matching --topic before running this script "
            f"(default topic prefix: {args.topic_prefix})."
        )
    else:
        print(
            f"Relay instances will be launched via SSH: {args.relay_user}@{args.relay_host} "
            f"(topic prefix: {args.topic_prefix})."
        )

    for index, msg in enumerate(messages, start=1):
        try:
            size_bytes = infer_size_bytes(msg)
        except ConfigError as exc:
            print(f"Skipping message '{msg}': {exc}", file=sys.stderr)
            continue

        base_rate = pick_base_rate(size_bytes)
        rate = clamp_rate_for_bandwidth(size_bytes, base_rate)

        if rate < base_rate:
            print(
                f"[info] Adjusted rate for {msg} from {base_rate}Hz to {rate}Hz to stay under 50 Mbps "
                f"(size: {size_bytes} bytes)."
            )
        else:
            print(f"[info] Using rate {rate}Hz for {msg} (size: {size_bytes} bytes).")

        log_name = f"{index:02d}_{msg}_{rate}hz.{args.log_format}"
        log_path = run_root / log_name

        topic = f"{args.topic_prefix}_{msg.lower()}"
        cmd = build_perf_test_command(msg, rate, args, log_path, topic)

        max_attempts = max(1, args.retry_on_zero + 1)
        samples_recorded = 0
        for attempt in range(max_attempts):
            relay_proc = start_relay(msg, topic, args)
            if relay_proc is None and not args.dry_run and not args.disable_relay and args.relay_host:
                print("[error] Relay failed to start; aborting sweep.", file=sys.stderr)
                return 3

            try:
                return_code = run_command(cmd, args.env_script, args.dry_run)
            finally:
                if relay_proc:
                    stop_relay(
                        relay_proc,
                        timeout=args.max_runtime + args.relay_duration_margin + 5,
                    )

            if return_code != 0:
                print(
                    f"[error] perf_test exited with code {return_code} for message {msg}.",
                    file=sys.stderr,
                )
                return return_code

            samples_recorded = count_latency_samples(log_path)
            if samples_recorded > 0 or args.dry_run or attempt == max_attempts - 1:
                if samples_recorded == 0 and not args.dry_run:
                    print(
                        f"[warn] No samples recorded for {msg} after {attempt + 1} attempt(s).",
                        file=sys.stderr,
                    )
                break

            print(
                f"[warn] No samples recorded for {msg} on attempt {attempt + 1}; retrying...",
                file=sys.stderr,
            )
            time.sleep(max(args.relay_wait_seconds, 0.5))

        if samples_recorded > 0 and not args.dry_run:
            print(f"[info] Samples recorded for {msg}: {samples_recorded}")

        manifest_rows.append((index, msg, size_bytes, rate, log_path))

    manifest_path = run_root / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    print(f"Manifest written to {manifest_path}")

    print("Sweep completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
