#!/usr/bin/env python3
"""Visualize latency quantiles for perf_test roundtrip runs."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict, namedtuple
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    _PLOT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - matplotlib might be unavailable
    plt = None
    _PLOT_IMPORT_ERROR = exc


RunMetrics = namedtuple(
    "RunMetrics",
    [
        "message_type",
        "size_bytes",
        "rate_hz",
        "samples",
        "latency_min_s",
        "latency_max_s",
        "latency_mean_s",
        "latency_std_s",
        "p50_s",
        "p99_s",
    ],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate perf_test roundtrip logs and visualise approximate "
            "latency quantiles (p50/p99) per message type."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/userdata/perf_test_roundtrip"),
        help="Root directory containing timestamped perf_test runs.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        type=Path,
        help="Specific run directories to include (absolute or relative). "
        "If omitted, all timestamped subdirectories under --root are used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the generated PNG plot. "
        "Defaults to `<root>/latency_quantiles.png`.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional path to write the aggregated table as CSV. "
        "Defaults to `<root>/latency_quantiles.csv`.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively (requires matplotlib).",
    )
    parser.add_argument(
        "--scale",
        choices=["auto", "linear", "log"],
        default="auto",
        help="YAxis scale for plots: auto (default), linear, or log.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a log file cannot be read (default: skip with warning).",
    )
    return parser.parse_args()


def discover_runs(root: Path) -> List[Path]:
    """Return sorted timestamped run directories below root."""
    if not root.exists() or not root.is_dir():
        return []
    runs = [p for p in root.iterdir() if p.is_dir()]
    runs.sort()
    return runs


def read_manifest(run_dir: Path) -> Iterable[Dict[str, str]]:
    manifest_path = run_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def aggregate_latency(entries: Iterable[dict]) -> Optional[dict]:
    total_n = 0
    total_sum = 0.0
    total_sum_sq = 0.0
    latency_min = math.inf
    latency_max = -math.inf

    for entry in entries:
        n = entry.get("latency_n", 0)
        if not n:
            continue

        mean = entry.get("latency_mean")
        m2 = entry.get("latency_M2")
        lmin = entry.get("latency_min")
        lmax = entry.get("latency_max")

        if None in (mean, m2, lmin, lmax):
            continue

        sum_i = mean * n
        sum_sq_i = m2 + (sum_i * sum_i) / n

        total_n += n
        total_sum += sum_i
        total_sum_sq += sum_sq_i
        latency_min = min(latency_min, lmin)
        latency_max = max(latency_max, lmax)

    if total_n == 0 or latency_min is math.inf or latency_max is -math.inf:
        return None

    mean = total_sum / total_n

    variance = 0.0
    if total_n > 1:
        numerator = total_sum_sq - (total_sum * total_sum) / total_n
        variance = max(0.0, numerator / (total_n - 1))

    return {
        "n": total_n,
        "mean": mean,
        "variance": variance,
        "min": latency_min,
        "max": latency_max,
    }


def approximate_quantile(mean: float, variance: float, bounds: Tuple[float, float], q: float) -> float:
    """Return a normal-approximation quantile clamped to [min, max]."""
    low, high = bounds
    if low > high:
        low, high = high, low

    if variance <= 0.0:
        return min(high, max(low, mean))

    std = math.sqrt(variance)
    nd = NormalDist(mu=mean, sigma=std)
    value = nd.inv_cdf(q)
    if math.isnan(value):
        value = mean
    return min(high, max(low, value))


def load_run(run_dir: Path, strict: bool = False) -> Dict[str, RunMetrics]:
    results: Dict[str, RunMetrics] = {}
    for row in read_manifest(run_dir):
        message = row["message_type"]
        size_bytes = int(row.get("size_bytes") or 0)
        rate_hz = float(row.get("rate_hz") or 0.0)
        log_filename = Path(row["logfile"]).name
        log_path = run_dir / log_filename

        if not log_path.exists():
            msg = f"Log file missing: {log_path}"
            if strict:
                raise FileNotFoundError(msg)
            print(f"[warn] {msg}", file=sys.stderr)
            continue

        try:
            with log_path.open("r") as handle:
                payload = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive
            if strict:
                raise
            print(f"[warn] Failed to parse {log_path}: {exc}", file=sys.stderr)
            continue

        analysis_entries = payload.get("analysis_results", [])
        stats = aggregate_latency(analysis_entries)
        if not stats:
            print(f"[warn] No latency data in {log_path}", file=sys.stderr)
            results[message] = RunMetrics(
                message_type=message,
                size_bytes=size_bytes,
                rate_hz=rate_hz,
                samples=0,
                latency_min_s=math.nan,
                latency_max_s=math.nan,
                latency_mean_s=math.nan,
                latency_std_s=math.nan,
                p50_s=math.nan,
                p99_s=math.nan,
            )
            continue

        mean = stats["mean"]
        variance = stats["variance"]
        samples = stats["n"]
        min_latency = stats["min"]
        max_latency = stats["max"]
        std = math.sqrt(variance) if variance > 0.0 else 0.0

        p50 = approximate_quantile(mean, variance, (min_latency, max_latency), 0.5)
        p99 = approximate_quantile(mean, variance, (min_latency, max_latency), 0.99)

        results[message] = RunMetrics(
            message_type=message,
            size_bytes=size_bytes,
            rate_hz=rate_hz,
            samples=samples,
            latency_min_s=min_latency,
            latency_max_s=max_latency,
            latency_mean_s=mean,
            latency_std_s=std,
            p50_s=p50,
            p99_s=p99,
        )
    return results


def to_ms(value: float) -> float:
    return value * 1000.0


def ensure_matplotlib_available(show: bool, output_path: Optional[Path]) -> None:
    if plt is None and (show or output_path):
        raise RuntimeError(
            "matplotlib is required for plotting. Install it (e.g., `pip install matplotlib`) "
            f"or rerun without --show/--output. Import failure: {_PLOT_IMPORT_ERROR}"
        )


def build_plot_data(run_summaries: Dict[str, Dict[str, RunMetrics]]) -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    size_and_name: List[Tuple[int, str]] = []
    seen: Dict[str, int] = {}

    for run_data in run_summaries.values():
        for metrics in run_data.values():
            if metrics.message_type not in seen:
                seen[metrics.message_type] = metrics.size_bytes

    size_and_name = sorted((size, name) for name, size in seen.items())
    messages = [name for size, name in size_and_name]

    p50_values: Dict[str, Dict[str, float]] = defaultdict(dict)
    p99_values: Dict[str, Dict[str, float]] = defaultdict(dict)

    for run_name, run_data in run_summaries.items():
        for msg in messages:
            if msg not in run_data:
                continue
            metrics = run_data[msg]
            p50_values[run_name][msg] = to_ms(metrics.p50_s)
            p99_values[run_name][msg] = to_ms(metrics.p99_s)

    return messages, p50_values, p99_values


def plot_quantiles(
    messages: List[str],
    p50_data: Dict[str, Dict[str, float]],
    p99_data: Dict[str, Dict[str, float]],
    output_path: Optional[Path],
    show: bool,
    scale_mode: str,
) -> None:
    ensure_matplotlib_available(show, output_path)
    if plt is None:
        return

    run_names = sorted(p50_data.keys())
    if not run_names:
        print("[info] No data available to plot.")
        return

    x_positions = list(range(len(messages)))

    all_positive: List[float] = []
    for dataset in (p50_data, p99_data):
        for run_vals in dataset.values():
            for value in run_vals.values():
                if not math.isnan(value) and value > 0:
                    all_positive.append(value)

    use_log = False
    if scale_mode == "log":
        use_log = True
    elif scale_mode == "linear":
        use_log = False
    else:  # auto
        if all_positive:
            min_positive = min(all_positive)
            max_positive = max(all_positive)
            if max_positive / max(min_positive, 1e-9) >= 20:
                use_log = True
                print(
                    "[info] Auto scale switched to log due to wide latency range "
                    f"({min_positive:.3f} ms .. {max_positive:.3f} ms)."
                )

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Approximate Roundtrip Latency Quantiles (ms)")

    bar_configs = [
        ("p50 (median)", axes[0], p50_data),
        ("p99", axes[1], p99_data),
    ]

    for title, ax, dataset in bar_configs:
        axis_values = []
        for run_name in run_names:
            axis_values.extend(dataset[run_name].get(msg, math.nan) for msg in messages)
        axis_positive = [v for v in axis_values if not math.isnan(v) and v > 0]
        axis_max = max(axis_positive) if axis_positive else 0.0
        linear_offset = axis_max * 0.015 if axis_max else 0.1

        width = 0.8 / max(len(run_names), 1)
        for idx, run_name in enumerate(run_names):
            values = [dataset[run_name].get(msg, math.nan) for msg in messages]
            offsets = [x + (idx - (len(run_names) - 1) / 2) * width for x in x_positions]
            bars = ax.bar(offsets, values, width=width, label=run_name)
            for bar, value in zip(bars, values):
                if math.isnan(value) or value <= 0:
                    continue
                if use_log and value > 0:
                    label_y = value * 1.15
                else:
                    label_y = value + linear_offset
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_ylabel(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend()
        if use_log and axis_positive:
            ax.set_yscale("log")
            lower = max(min(axis_positive) * 0.8, 1e-6)
            upper = max(axis_positive) * 1.3
            ax.set_ylim(bottom=lower, top=upper)
        elif axis_positive:
            ax.set_ylim(top=axis_max * 1.2)

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(messages, rotation=45, ha="right")
    axes[-1].set_xlabel("Message Type")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"[info] Plot saved to {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def write_summary_csv(
    run_summaries: Dict[str, Dict[str, RunMetrics]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "message_type",
        "size_bytes",
        "rate_hz",
        "samples",
        "latency_min_ms",
        "latency_max_ms",
        "latency_mean_ms",
        "latency_std_ms",
        "p50_ms",
        "p99_ms",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run_name, data in sorted(run_summaries.items()):
            for metrics in data.values():
                writer.writerow(
                    {
                        "run": run_name,
                        "message_type": metrics.message_type,
                        "size_bytes": metrics.size_bytes,
                        "rate_hz": metrics.rate_hz,
                        "samples": metrics.samples,
                        "latency_min_ms": to_ms(metrics.latency_min_s),
                        "latency_max_ms": to_ms(metrics.latency_max_s),
                        "latency_mean_ms": to_ms(metrics.latency_mean_s),
                        "latency_std_ms": to_ms(metrics.latency_std_s),
                        "p50_ms": to_ms(metrics.p50_s),
                        "p99_ms": to_ms(metrics.p99_s),
                    }
                )
    print(f"[info] Summary CSV written to {path}")


def main() -> int:
    args = parse_args()

    if args.runs:
        run_dirs = [Path(run).resolve() for run in args.runs]
    else:
        run_dirs = discover_runs(args.root.resolve())

    if not run_dirs:
        print("[error] No run directories found.", file=sys.stderr)
        return 1

    run_summaries: Dict[str, Dict[str, RunMetrics]] = {}
    for run_dir in run_dirs:
        try:
            data = load_run(run_dir, strict=args.strict)
        except Exception as exc:
            if args.strict:
                raise
            print(f"[warn] Skipping run {run_dir}: {exc}", file=sys.stderr)
            continue

        if not data:
            print(f"[warn] No usable data in {run_dir}", file=sys.stderr)
            continue

        run_summaries[run_dir.name] = data

    if not run_summaries:
        print("[error] No data available after parsing runs.", file=sys.stderr)
        return 2

    if args.output is not None:
        default_output = args.output
    elif plt is not None:
        default_output = args.root / "latency_quantiles.png"
    else:
        default_output = None

    default_csv = args.summary_csv or (args.root / "latency_quantiles.csv")

    messages, p50_data, p99_data = build_plot_data(run_summaries)
    plot_quantiles(messages, p50_data, p99_data, default_output, args.show, args.scale)
    write_summary_csv(run_summaries, default_csv)

    # Print a quick textual summary
    for run_name, run_data in sorted(run_summaries.items()):
        print(f"\n[run] {run_name}")
        for metrics in sorted(run_data.values(), key=lambda m: (m.size_bytes, m.message_type)):
            print(
                f"  {metrics.message_type:<10} "
                f"mean={to_ms(metrics.latency_mean_s):6.3f} ms "
                f"p50≈{to_ms(metrics.p50_s):6.3f} ms "
                f"p99≈{to_ms(metrics.p99_s):6.3f} ms "
                f"(samples={metrics.samples})"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
