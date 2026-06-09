#!/usr/bin/env python3
import argparse
import ast
import csv
import os
import re
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Summarize ann-benchmarks result HDF5 files in a directory."
    )
    p.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing result .hdf5 files, e.g. results/.../golden",
    )
    p.add_argument(
        "--dataset-hdf5",
        default="",
        help="Dataset HDF5 with ground-truth neighbors. If set, recall@k is computed.",
    )
    p.add_argument(
        "--count",
        type=int,
        default=10,
        help="Top-k used for recall and result slicing.",
    )
    p.add_argument(
        "--output",
        default="",
        help="Optional output CSV path. If omitted, only prints a table.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan results-dir for .hdf5 files.",
    )
    return p.parse_args()


def parse_params_from_name(name):
    metric = None
    m = efc = efs = None

    if name:
        mobj = re.search(r"\{[^}]*\}", name)
        if mobj:
            try:
                d = ast.literal_eval(mobj.group(0))
                m = d.get("M")
                efc = d.get("efConstruction")
            except Exception:
                pass

        eobj = re.search(r"ef:\s*(\d+)", name)
        if eobj:
            efs = int(eobj.group(1))

    return metric, m, efc, efs


def parse_params_from_filename(path):
    stem = path.stem
    thread = None

    mt = re.search(r"_(\d+)t$", stem)
    if mt:
        thread = int(mt.group(1))
        stem = stem[: mt.start()]

    metric = stem.split("_", 1)[0] if "_" in stem else ""
    m = efc = efs = None

    mh = re.search(r"_M_(\d+)_efConstruction_(\d+)_(\d+)$", stem)
    if mh:
        m = int(mh.group(1))
        efc = int(mh.group(2))
        efs = int(mh.group(3))

    return metric, m, efc, efs, thread


def recall_at_k(true_neighbors, run_neighbors, k):
    n = min(len(true_neighbors), len(run_neighbors))
    if n == 0:
        return np.nan

    vals = []
    for gt, got in zip(true_neighbors[:n, :k], run_neighbors[:n, :k]):
        got = [int(x) for x in got if int(x) >= 0]
        vals.append(len(set(map(int, gt)) & set(got)) / float(k))
    return float(np.mean(vals))


def result_files(results_dir, recursive):
    root = Path(results_dir)
    pattern = "**/*.hdf5" if recursive else "*.hdf5"
    return sorted(root.glob(pattern))


def summarize_file(path, count, true_neighbors):
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        times = np.asarray(f["times"], dtype=float) if "times" in f else np.array([])
        run_neighbors = np.asarray(f["neighbors"]) if "neighbors" in f else None

    metric_file, m_file, efc_file, efs_file, thread = parse_params_from_filename(path)
    _, m_name, efc_name, efs_name = parse_params_from_name(str(attrs.get("name", "")))

    m = m_name if m_name is not None else m_file
    efc = efc_name if efc_name is not None else efc_file
    efs = efs_name if efs_name is not None else efs_file

    total_s = float(np.sum(times)) if len(times) else np.nan
    mean_s = float(np.mean(times)) if len(times) else np.nan
    qps = float(len(times) / total_s) if len(times) and total_s > 0 else np.nan

    recall = np.nan
    if true_neighbors is not None and run_neighbors is not None:
        recall = recall_at_k(true_neighbors, run_neighbors, count)

    build_s = float(attrs.get("build_time", np.nan))
    index_size = float(attrs.get("index_size", np.nan))
    dist_comps = float(attrs.get("dist_comps", np.nan))
    run_count = float(attrs.get("run_count", np.nan))
    query_count = len(times)
    dist_comps_per_query_run = np.nan
    if not np.isnan(dist_comps) and not np.isnan(run_count) and run_count > 0 and query_count > 0:
        dist_comps_per_query_run = dist_comps / (run_count * query_count)

    return {
        "file": str(path),
        "filename": path.name,
        "metric": metric_file or attrs.get("distance", ""),
        "threads": thread if thread is not None else "",
        "M": m if m is not None else "",
        "efConstruction": efc if efc is not None else "",
        "efSearch": efs if efs is not None else "",
        "recall_at_k": recall,
        "query_count": query_count,
        "run_count": attrs.get("run_count", ""),
        "total_s": total_s,
        "mean_ms": mean_s * 1000.0 if not np.isnan(mean_s) else np.nan,
        "p50_ms": float(np.percentile(times, 50) * 1000.0) if len(times) else np.nan,
        "p95_ms": float(np.percentile(times, 95) * 1000.0) if len(times) else np.nan,
        "p99_ms": float(np.percentile(times, 99) * 1000.0) if len(times) else np.nan,
        "qps": qps,
        "build_s": build_s,
        "build_min": build_s / 60.0 if not np.isnan(build_s) else np.nan,
        "index_size_raw": index_size,
        "index_size_gib_if_kib": index_size / 1024.0 / 1024.0 if not np.isnan(index_size) else np.nan,
        "dist_comps": dist_comps,
        "dist_comps_per_query_run": dist_comps_per_query_run,
        "batch_mode": bool(attrs.get("batch_mode", False)),
        "name": attrs.get("name", ""),
    }


def fmt(value, digits=3):
    if value == "":
        return ""
    try:
        v = float(value)
    except Exception:
        return str(value)
    if np.isnan(v):
        return ""
    return f"{v:.{digits}f}"


def print_table(rows):
    cols = [
        "threads",
        "M",
        "efSearch",
        "recall_at_k",
        "qps",
        "mean_ms",
        "p95_ms",
        "p99_ms",
        "build_s",
        "filename",
    ]
    widths = {c: len(c) for c in cols}
    rendered = []
    for r in rows:
        item = {
            "threads": str(r["threads"]),
            "M": str(r["M"]),
            "efSearch": str(r["efSearch"]),
            "recall_at_k": fmt(r["recall_at_k"], 5),
            "qps": fmt(r["qps"], 1),
            "mean_ms": fmt(r["mean_ms"], 3),
            "p95_ms": fmt(r["p95_ms"], 3),
            "p99_ms": fmt(r["p99_ms"], 3),
            "build_s": fmt(r["build_s"], 1),
            "filename": r["filename"],
        }
        rendered.append(item)
        for c in cols:
            widths[c] = max(widths[c], len(item[c]))

    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for item in rendered:
        print("  ".join(item[c].ljust(widths[c]) for c in cols))


def main():
    args = parse_args()

    true_neighbors = None
    if args.dataset_hdf5:
        with h5py.File(args.dataset_hdf5, "r") as f:
            true_neighbors = np.asarray(f["neighbors"][:, : args.count])

    rows = [
        summarize_file(path, args.count, true_neighbors)
        for path in result_files(args.results_dir, args.recursive)
    ]
    if not rows:
        raise SystemExit(f"no .hdf5 result files found under {args.results_dir}")

    rows.sort(key=lambda r: (
        int(r["M"]) if r["M"] != "" else 0,
        int(r["efSearch"]) if r["efSearch"] != "" else 0,
        int(r["threads"]) if r["threads"] != "" else 0,
    ))

    print_table(rows)

    if args.output:
        out = Path(args.output)
        if out.parent:
            out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\ncsv: {out}")


if __name__ == "__main__":
    main()
