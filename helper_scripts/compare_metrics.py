#!/usr/bin/env python3
"""Compare perf_test metrics (beyond latency) across multiple runs."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


MetricRow = Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare perf_test roundtrip metrics (samples, lost, data volume, latency, CPU) "
            "across multiple run directories."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Run directory with label, e.g., --run fastrtps=/path/to/run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path for the comparison chart (default: comparison_metrics.png in root).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional CSV output summarising the metrics.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Base directory used for default outputs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Figure DPI (default: 120).",
    )
    return parser.parse_args()


def parse_run_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Run specification must be LABEL=PATH, got: {spec}")
    label, path = spec.split("=", 1)
    return label.strip(), Path(path.strip()).resolve()


def load_run_metrics(run_dir: Path) -> Dict[str, MetricRow]:
    metrics: Dict[str, MetricRow] = {}
    for json_path in sorted(run_dir.glob("*.json")):
        with json_path.open("r") as handle:
            payload = json.load(handle)

        msg = payload.get("msg_name")
        if not msg:
            continue

        entries = payload.get("analysis_results", [])
        samples_received = sum(e.get("num_samples_received", 0) for e in entries)
        samples_sent = sum(e.get("num_samples_sent", 0) for e in entries)
        samples_lost = sum(e.get("num_samples_lost", 0) for e in entries)
        data_received = sum(e.get("total_data_received", 0) for e in entries)
        cpu_total = sum(e.get("cpu_info_cpu_usage", 0.0) for e in entries)
        cpu_count = len(entries)

        latency_n = sum(e.get("latency_n", 0) for e in entries)
        if latency_n:
            latency_mean = (
                sum(e.get("latency_mean", 0.0) * e.get("latency_n", 0) for e in entries)
                / latency_n
            )
        else:
            latency_mean = math.nan

        lat_min = min(
            (e.get("latency_min") for e in entries if isinstance(e.get("latency_min"), (int, float))),
            default=math.nan,
        )
        lat_max = max(
            (e.get("latency_max") for e in entries if isinstance(e.get("latency_max"), (int, float))),
            default=math.nan,
        )

        metrics[msg] = {
            "samples_received": samples_received,
            "samples_sent": samples_sent,
            "samples_lost": samples_lost,
            "data_bytes": data_received,
            "cpu_avg": cpu_total / cpu_count if cpu_count else math.nan,
            "latency_mean_ms": latency_mean * 1000 if latency_mean == latency_mean else math.nan,
            "latency_min_ms": lat_min * 1000 if lat_min == lat_min else math.nan,
            "latency_max_ms": lat_max * 1000 if lat_max == lat_max else math.nan,
            "latency_samples": latency_n,
            "rate_hz": payload.get("rate"),
            "rate_label": f"{payload.get('rate')} Hz" if payload.get("rate") is not None else "",
        }
    return metrics


def gather_all_messages(metrics_by_run: Dict[str, Dict[str, MetricRow]]) -> List[str]:
    def parse_order(msg: str) -> Tuple[int, int]:
        # Explicit ordering table
        order_map = {
            "Array128": 1,
            "Array256": 2,
            "Array512": 3,
            "Array1k": 4,
            "Array4k": 5,
            "Array16k": 6,
            "Array32k": 7,
            "Array60k": 8,
            "Array64k": 9,
            "Array128k": 10,
            "Array256k": 11,
            "Array512k": 12,
            "Array1m": 13,
            "Array2m": 14,
            "Array4m": 15,
        }
        if msg in order_map:
            return (order_map[msg], 0)
        digits = "".join(filter(str.isdigit, msg))
        return (int(digits) if digits else 9999, 1)

    msgs = set()
    for run_metrics in metrics_by_run.values():
        msgs.update(run_metrics.keys())
    return sorted(msgs, key=parse_order)


def gather_message_rates(metrics_by_run: Dict[str, Dict[str, MetricRow]]) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for run_metrics in metrics_by_run.values():
        for msg, row in run_metrics.items():
            rate = row.get("rate_hz")
            if rate is not None:
                rates[msg] = rate
    return rates


def plot_metrics(
    messages: List[str],
    metrics_by_run: Dict[str, Dict[str, MetricRow]],
    message_labels: List[str],
    output_path: Path,
    dpi: int,
) -> None:
    if not messages:
        print("[info] No messages to plot.")
        return

    run_labels = sorted(metrics_by_run.keys())
    x = range(len(messages))

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig.suptitle("perf_test Roundtrip Metrics Comparison")

    panels = [
        ("Samples Received", "samples_received", axes[0][0], "linear"),
        ("Samples Lost", "samples_lost", axes[0][1], "linear"),
        ("Total Data (MB)", "data_bytes", axes[1][0], "linear"),
        ("Avg CPU (%)", "cpu_avg", axes[1][1], "linear"),
        ("Mean Latency (ms)", "latency_mean_ms", axes[2][0], "log"),
        ("Max Latency (ms)", "latency_max_ms", axes[2][1], "log"),
    ]

    width = 0.8 / max(len(run_labels), 1)

    for title, key, ax, scale in panels:
        for idx, label in enumerate(run_labels):
            values = []
            for msg in messages:
                row = metrics_by_run.get(label, {}).get(msg)
                value = row.get(key) if row else math.nan
                if key == "data_bytes" and value == value:
                    value = value / 1e6  # bytes -> MB
                values.append(value)

            offsets = [val + (idx - (len(run_labels) - 1) / 2) * width for val in x]
            bars = ax.bar(offsets, values, width=width, label=label)

            for bar, value in zip(bars, values):
                if value != value or value <= 0:
                    continue
                if scale == "log":
                    label_y = value * 1.2
                else:
                    label_y = value * 1.02 if value != 0 else value + 0.01
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if scale == "log":
            ax.set_yscale("log")
            finite_vals = [
                row.get(key)
                for label in run_labels
                for row in metrics_by_run[label].values()
                if row.get(key) and row.get(key) == row.get(key)
            ]
            if finite_vals:
                min_val = min(v for v in finite_vals if v > 0)
                ax.set_ylim(bottom=min_val * 0.8)
        ax.legend()

    for ax in axes[-1]:
        ax.set_xticks(list(x))
        ax.set_xticklabels(message_labels, rotation=45, ha="right")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"[info] Comparison plot saved to {output_path}")


def write_summary_csv(
    messages: List[str],
    run_labels: List[str],
    metrics_by_run: Dict[str, Dict[str, MetricRow]],
    csv_path: Path,
) -> None:
    fieldnames = [
        "message",
        "stack",
        "samples_received",
        "samples_sent",
        "samples_lost",
        "latency_samples",
        "data_mb",
        "avg_cpu",
        "latency_mean_ms",
        "latency_min_ms",
        "latency_max_ms",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for msg in messages:
            for label in run_labels:
                row = metrics_by_run.get(label, {}).get(msg)
                if not row:
                    continue
                writer.writerow(
                    {
                        "message": msg,
                        "stack": label,
                        "samples_received": row.get("samples_received"),
                        "samples_sent": row.get("samples_sent"),
                        "samples_lost": row.get("samples_lost"),
                        "latency_samples": row.get("latency_samples"),
                        "data_mb": row.get("data_bytes", 0) / 1e6,
                        "avg_cpu": row.get("cpu_avg"),
                        "latency_mean_ms": row.get("latency_mean_ms"),
                        "latency_min_ms": row.get("latency_min_ms"),
                        "latency_max_ms": row.get("latency_max_ms"),
                    }
                )
    print(f"[info] Summary CSV saved to {csv_path}")


def main() -> int:
    args = parse_args()
    run_specs = dict(parse_run_spec(spec) for spec in args.run)

    metrics_by_run = {}
    for label, directory in run_specs.items():
        if not directory.exists():
            print(f"[warn] Run directory not found: {directory}")
            continue
        metrics_by_run[label] = load_run_metrics(directory)

    if not metrics_by_run:
        print("[error] No valid runs to compare.")
        return 1

    messages = gather_all_messages(metrics_by_run)
    run_labels = sorted(metrics_by_run.keys())
    msg_rates = gather_message_rates(metrics_by_run)
    message_labels = [
        f"{msg}\n({int(msg_rates.get(msg, 0))} Hz)" if msg_rates.get(msg) else msg
        for msg in messages
    ]

    output_path = args.output
    if output_path is None:
        output_path = args.root / "comparison_metrics.png"

    plot_metrics(messages, metrics_by_run, message_labels, output_path, args.dpi)

    if args.summary_csv:
        write_summary_csv(messages, run_labels, metrics_by_run, args.summary_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
