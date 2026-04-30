#!/usr/bin/env python3
"""
Generate all six presentation charts from the unified sweep CSVs.

Reads:
    <sweep_dir>/primary_sweep.csv  (distance x seed, all hops 1-7)
    <sweep_dir>/dbsize_sweep.csv   (db_size x seed)

Produces six PNG figures:
    chart1_success_heatmap.png     — ODO vs ACP success-rate heatmaps
    chart2_tts_small_multiples.png — TTS vs hops, 4 distance panels
    chart3_fidelity_lines.png      — ACP fidelity vs hops, 4 distance lines
    chart4_dbsize_scaling.png      — success rate and TTS vs n (log2 db size)
    chart5_failure_decomp.png      — stacked failure-reason bars
    chart6_pair_arrival.png        — time-to-first vs time-to-last pair

Usage:
    python plot.py --sweep-dir sweep2d_output --output-dir figures
"""

import os
import csv
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

C_ODO = "#5a6577"
C_ACP = "#2563eb"
C_THRESH = "#dc2626"

# Colorblind-friendly distance palette (4 distances)
DIST_COLORS = {
    10: "#0B6E4F",
    20: "#08A4BD",
    30: "#E6A817",
    40: "#C9302C",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str) -> List[dict]:
    """Load the unified sweep CSV with type coercion."""
    rows = []
    if not os.path.exists(path):
        print(f"  WARN: {path} not found")
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Type coercion
            for k in ("seed", "database_size_log", "num_nodes",
                      "query_id", "hop_distance"):
                if k in row and row[k] != "":
                    row[k] = int(row[k])
            for k in ("inter_node_distance_km", "tts_ms", "fidelity"):
                if k in row and row[k] != "":
                    row[k] = float(row[k])
            if "success" in row:
                row["success"] = row["success"] in ("True", "true", "1")
            # Parse pair arrivals
            row["pair_arrival_list"] = [
                float(t) for t in row.get("pair_arrival_ms", "").split(";")
                if t
            ]
            rows.append(row)
    return rows


def aggregate(rows: List[dict], key_fn) -> Dict:
    """Group rows and return success/tts/fid stats per group."""
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)

    out = {}
    for key, recs in groups.items():
        n = len(recs)
        successes = [r for r in recs if r["success"]]
        n_s = len(successes)
        tts = [r["tts_ms"] for r in successes if r["tts_ms"] > 0]
        fid = [r["fidelity"] for r in successes if r["fidelity"] > 0]
        out[key] = {
            "n": n,
            "n_success": n_s,
            "rate": n_s / n if n else 0,
            "tts_median": float(np.median(tts)) if tts else 0.0,
            "tts_mean": float(np.mean(tts)) if tts else 0.0,
            "tts_q1": float(np.percentile(tts, 25)) if tts else 0.0,
            "tts_q3": float(np.percentile(tts, 75)) if tts else 0.0,
            "fid_mean": float(np.mean(fid)) if fid else 0.0,
            "fid_std": float(np.std(fid)) if fid else 0.0,
        }
    return out


def normalize_backend(b: str) -> str:
    """Collapse 'acp_m6', 'acp_m8' etc. to 'acp'."""
    return "acp" if b.startswith("acp") else b


# ---------------------------------------------------------------------------
# CHART 1 — Success rate heatmaps
# ---------------------------------------------------------------------------

def chart1_success_heatmap(rows: List[dict], output_dir: str):
    """Two-panel heatmap: ODO vs ACP, x=hops, y=distance, color=success rate."""
    for r in rows:
        r["_backend"] = normalize_backend(r["backend"])

    stats = aggregate(rows, lambda r: (r["_backend"],
                                       int(r["inter_node_distance_km"]),
                                       r["hop_distance"]))

    distances = sorted({int(r["inter_node_distance_km"]) for r in rows},
                       reverse=True)
    hops = sorted({r["hop_distance"] for r in rows
                   if r["hop_distance"] > 0})

    if not distances or not hops:
        print("  chart1: no data")
        return

    def build_matrix(backend: str) -> np.ndarray:
        M = np.full((len(distances), len(hops)), np.nan)
        for i, d in enumerate(distances):
            for j, h in enumerate(hops):
                cell = stats.get((backend, d, h))
                if cell and cell["n"] > 0:
                    M[i, j] = cell["rate"] * 100
        return M

    M_odo = build_matrix("odo")
    M_acp = build_matrix("acp")

    # Build per-cell sample size matrix (combined across backends, since either
    # alone can have a small n at deep hops).
    def build_n_matrix(backend: str) -> np.ndarray:
        Mn = np.zeros((len(distances), len(hops)), dtype=int)
        for i, d in enumerate(distances):
            for j, h in enumerate(hops):
                cell = stats.get((backend, d, h))
                if cell:
                    Mn[i, j] = cell["n"]
        return Mn

    N_odo = build_n_matrix("odo")
    N_acp = build_n_matrix("acp")

    # Cells where either backend has fewer than this many samples are flagged.
    LOW_N_THRESHOLD = 30

    cmap = LinearSegmentedColormap.from_list(
        "success", ["#b91c1c", "#fef3c7", "#166534"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    any_low_n = False
    for ax, M, Mn, title in zip(axes, [M_odo, M_acp], [N_odo, N_acp],
                                 ["ODO", "ACP"]):
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(hops)))
        ax.set_xticklabels(hops)
        ax.set_yticks(range(len(distances)))
        ax.set_yticklabels([f"{d} km" for d in distances])
        ax.set_xlabel("Hop distance from QDC")
        ax.set_title(title)
        ax.grid(False)

        # Write percentage in each cell, with low-n indication
        for i in range(len(distances)):
            for j in range(len(hops)):
                val = M[i, j]
                n_here = Mn[i, j]
                if not np.isnan(val):
                    txt_color = "white" if (val < 35 or val > 75) else "black"
                    if n_here < LOW_N_THRESHOLD:
                        any_low_n = True
                        # Hash overlay to flag low-n cells visually
                        ax.add_patch(plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, hatch="///",
                            edgecolor="white", alpha=0.5, linewidth=0,
                        ))
                        # Show value with sample size
                        ax.text(j, i, f"{val:.0f}*", ha="center", va="center",
                                color=txt_color, fontsize=10, fontweight="bold")
                    else:
                        ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                                color=txt_color, fontsize=10, fontweight="bold")
                else:
                    ax.text(j, i, "—", ha="center", va="center",
                            color="#888", fontsize=10)

    axes[0].set_ylabel("Inter-node distance")

    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.03,
                        pad=0.02, shrink=0.85)
    cbar.set_label("Query success rate (%)")

    fig.suptitle("QPQ success rate: ODO vs ACP", fontsize=14, y=1.02)

    if any_low_n:
        fig.text(0.5, -0.02,
                 f"* = cell has fewer than {LOW_N_THRESHOLD} samples",
                 ha="center", fontsize=9, color="#444", style="italic")

    path = os.path.join(output_dir, "chart1_success_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# CHART 2 — TTS small multiples by distance
# ---------------------------------------------------------------------------

def chart2_tts_small_multiples(rows: List[dict], output_dir: str):
    """One panel per distance; each panel shows ODO and ACP TTS (median + IQR) vs hops."""
    for r in rows:
        r["_backend"] = normalize_backend(r["backend"])

    stats = aggregate(rows, lambda r: (r["_backend"],
                                       int(r["inter_node_distance_km"]),
                                       r["hop_distance"]))

    distances = sorted({int(r["inter_node_distance_km"]) for r in rows})
    hops = sorted({r["hop_distance"] for r in rows if r["hop_distance"] > 0})

    if not distances or not hops:
        print("  chart2: no data")
        return

    fig, axes = plt.subplots(1, len(distances), figsize=(4 * len(distances), 4),
                             sharey=True)
    if len(distances) == 1:
        axes = [axes]

    # Global y-max for shared scale
    all_q3 = [stats[k]["tts_q3"] for k in stats
              if stats[k]["n_success"] > 0]
    y_max = max(all_q3) * 1.1 if all_q3 else 100

    x = np.arange(len(hops))
    w = 0.36

    for ax, dist in zip(axes, distances):
        odo_med = [stats.get(("odo", dist, h), {}).get("tts_median", 0)
                   for h in hops]
        odo_q1 = [stats.get(("odo", dist, h), {}).get("tts_q1", 0) for h in hops]
        odo_q3 = [stats.get(("odo", dist, h), {}).get("tts_q3", 0) for h in hops]
        acp_med = [stats.get(("acp", dist, h), {}).get("tts_median", 0)
                   for h in hops]
        acp_q1 = [stats.get(("acp", dist, h), {}).get("tts_q1", 0) for h in hops]
        acp_q3 = [stats.get(("acp", dist, h), {}).get("tts_q3", 0) for h in hops]

        # IQR error bars: distances from median (must be non-negative)
        odo_err = [
            [max(0, m - q1) for m, q1 in zip(odo_med, odo_q1)],
            [max(0, q3 - m) for m, q3 in zip(odo_med, odo_q3)],
        ]
        acp_err = [
            [max(0, m - q1) for m, q1 in zip(acp_med, acp_q1)],
            [max(0, q3 - m) for m, q3 in zip(acp_med, acp_q3)],
        ]

        ax.bar(x - w/2, odo_med, w, yerr=odo_err, label="ODO", color=C_ODO,
               edgecolor="white", linewidth=0.5, capsize=3,
               error_kw={"ecolor": "#333", "linewidth": 1})
        ax.bar(x + w/2, acp_med, w, yerr=acp_err, label="ACP", color=C_ACP,
               edgecolor="white", linewidth=0.5, capsize=3,
               error_kw={"ecolor": "#333", "linewidth": 1})

        # Mark cells with zero successes
        for i, h in enumerate(hops):
            odo_n = stats.get(("odo", dist, h), {}).get("n_success", 0)
            acp_n = stats.get(("acp", dist, h), {}).get("n_success", 0)
            if odo_n == 0:
                ax.text(x[i] - w/2, y_max * 0.02, "×", ha="center",
                        color=C_ODO, fontsize=14, fontweight="bold")
            if acp_n == 0:
                ax.text(x[i] + w/2, y_max * 0.02, "×", ha="center",
                        color=C_ACP, fontsize=14, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(hops)
        ax.set_xlabel("Hop distance from QDC")
        ax.set_title(f"{dist} km links")
        ax.set_ylim(0, y_max)

    axes[0].set_ylabel("Time-to-serve (ms)")
    axes[0].legend(loc="upper left", framealpha=0.9)

    fig.suptitle("QPQ time-to-serve by distance and hop count"
                 "  (× = no successful queries at that cell)",
                 fontsize=13, y=1.02)

    path = os.path.join(output_dir, "chart2_tts_small_multiples.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# CHART 3 — Fidelity lines, ACP only, one line per distance
# ---------------------------------------------------------------------------

def chart3_fidelity_lines(rows: List[dict], output_dir: str):
    """ACP fidelity vs hops, one line per distance."""
    acp_rows = [r for r in rows if normalize_backend(r["backend"]) == "acp"]
    stats = aggregate(acp_rows, lambda r: (int(r["inter_node_distance_km"]),
                                           r["hop_distance"]))

    distances = sorted({int(r["inter_node_distance_km"]) for r in acp_rows})
    hops = sorted({r["hop_distance"] for r in acp_rows
                   if r["hop_distance"] > 0})

    if not distances or not hops:
        print("  chart3: no data")
        return

    fig, ax = plt.subplots(figsize=(7, 4.8))

    for dist in distances:
        y = [stats.get((dist, h), {}).get("fid_mean", np.nan) for h in hops]
        yerr = [stats.get((dist, h), {}).get("fid_std", 0) for h in hops]
        # Filter out zero-data points (where n_success == 0)
        valid = [(h, v, e) for h, v, e in zip(hops, y, yerr)
                 if v > 0 and stats.get((dist, h), {}).get("n_success", 0) > 0]
        if not valid:
            continue
        vh, vy, ve = zip(*valid)
        color = DIST_COLORS.get(dist, "#333")
        ax.errorbar(vh, vy, yerr=ve, marker="o", markersize=7,
                    linewidth=2, capsize=4, color=color,
                    label=f"{dist} km")

    ax.axhline(y=0.8, color=C_THRESH, linestyle="--", linewidth=1.8,
               label=r"$F_{th}$ = 0.8")

    ax.set_xlabel("Hop distance from QDC")
    ax.set_ylabel("End-to-end fidelity")
    ax.set_title("ACP end-to-end fidelity by link distance and hop count")
    ax.set_xticks(hops)
    ax.legend(loc="lower left", framealpha=0.9)

    # Caption
    fig.text(0.5, -0.03, "ODO produces statistically indistinguishable fidelity "
             "(scheduling-only algorithmic difference).",
             ha="center", fontsize=9, color="#444", style="italic")

    path = os.path.join(output_dir, "chart3_fidelity_lines.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# CHART 4 — DB size scaling (success rate + TTS)
# ---------------------------------------------------------------------------

def chart4_dbsize_scaling(rows: List[dict], output_dir: str):
    """Success rate and TTS vs log2(database_size)."""
    for r in rows:
        r["_backend"] = normalize_backend(r["backend"])

    stats = aggregate(rows, lambda r: (r["_backend"], r["database_size_log"]))
    db_sizes = sorted({r["database_size_log"] for r in rows})

    if not db_sizes:
        print("  chart4: no data")
        return

    odo_rate = [stats.get(("odo", n), {}).get("rate", 0) * 100 for n in db_sizes]
    acp_rate = [stats.get(("acp", n), {}).get("rate", 0) * 100 for n in db_sizes]
    odo_tts = [stats.get(("odo", n), {}).get("tts_median", 0) for n in db_sizes]
    odo_tts_q1 = [stats.get(("odo", n), {}).get("tts_q1", 0) for n in db_sizes]
    odo_tts_q3 = [stats.get(("odo", n), {}).get("tts_q3", 0) for n in db_sizes]
    acp_tts = [stats.get(("acp", n), {}).get("tts_median", 0) for n in db_sizes]
    acp_tts_q1 = [stats.get(("acp", n), {}).get("tts_q1", 0) for n in db_sizes]
    acp_tts_q3 = [stats.get(("acp", n), {}).get("tts_q3", 0) for n in db_sizes]

    fig, (ax_rate, ax_tts) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_rate.plot(db_sizes, odo_rate, marker="s", color=C_ODO, linewidth=2,
                 markersize=8, label="ODO")
    ax_rate.plot(db_sizes, acp_rate, marker="o", color=C_ACP, linewidth=2,
                 markersize=8, label="ACP")
    ax_rate.set_xlabel("log₂(database size) = n")
    ax_rate.set_ylabel("Query success rate (%)")
    ax_rate.set_title("Success rate vs database size")
    ax_rate.set_xticks(db_sizes)
    ax_rate.set_ylim(0, 105)
    ax_rate.legend(loc="lower left")

    # Secondary x-axis: pairs per query (= 2*(2n+1))
    ax_rate2 = ax_rate.twiny()
    ax_rate2.set_xlim(ax_rate.get_xlim())
    ax_rate2.set_xticks(db_sizes)
    ax_rate2.set_xticklabels([f"{2*(2*n+1)}" for n in db_sizes])
    ax_rate2.set_xlabel("Bell pairs per query")

    ax_tts.errorbar(db_sizes, odo_tts,
                    yerr=[[max(0, m - q) for m, q in zip(odo_tts, odo_tts_q1)],
                          [max(0, q - m) for m, q in zip(odo_tts, odo_tts_q3)]],
                    marker="s", color=C_ODO, linewidth=2, markersize=8,
                    capsize=4, label="ODO")
    ax_tts.errorbar(db_sizes, acp_tts,
                    yerr=[[max(0, m - q) for m, q in zip(acp_tts, acp_tts_q1)],
                          [max(0, q - m) for m, q in zip(acp_tts, acp_tts_q3)]],
                    marker="o", color=C_ACP, linewidth=2, markersize=8,
                    capsize=4, label="ACP")
    ax_tts.set_xlabel("log₂(database size) = n")
    ax_tts.set_ylabel("Time-to-serve (ms)")
    ax_tts.set_title("Time-to-serve vs database size")
    ax_tts.set_xticks(db_sizes)
    ax_tts.legend(loc="upper left")

    ax_tts2 = ax_tts.twiny()
    ax_tts2.set_xlim(ax_tts.get_xlim())
    ax_tts2.set_xticks(db_sizes)
    ax_tts2.set_xticklabels([f"{2*(2*n+1)}" for n in db_sizes])
    ax_tts2.set_xlabel("Bell pairs per query")

    fig.suptitle(f"QPQ scaling with database size (20 km links)",
                 fontsize=13, y=1.08)

    path = os.path.join(output_dir, "chart4_dbsize_scaling.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# CHART 5 — Failure reason decomposition
# ---------------------------------------------------------------------------

def chart5_failure_decomp(rows: List[dict], output_dir: str):
    """Stacked bars: fraction of outcomes by inferred-from-pair-count category.

    Since the original failure_reason field doesn't capture sim-end timeouts,
    we infer the outcome from how many pairs were delivered:

        0 pairs                  -> never_started
        1 .. (2n+1) - 1          -> round1_incomplete
        exactly (2n+1)           -> round1_done_round2_not_started
        (2n+1)+1 .. (4n+2) - 1   -> round2_incomplete
        exactly (4n+2)           -> all_pairs_delivered (success or rare timeout)

    Where n = database_size_log. This decomposition tells you _how far_ each
    failed query got, which is more diagnostic than a binary success flag.
    """
    for r in rows:
        r["_backend"] = normalize_backend(r["backend"])
        n = r.get("database_size_log", 10)
        pairs_per_round = 2 * n + 1
        pairs_per_query = 2 * pairs_per_round  # 4n + 2
        n_pairs = len(r["pair_arrival_list"])

        if r["success"]:
            r["_outcome"] = "success"
        elif n_pairs == 0:
            r["_outcome"] = "never_started"
        elif n_pairs < pairs_per_round:
            r["_outcome"] = "round1_incomplete"
        elif n_pairs == pairs_per_round:
            r["_outcome"] = "round1_done_round2_not_started"
        elif n_pairs < pairs_per_query:
            r["_outcome"] = "round2_incomplete"
        else:
            # Edge case: all pairs delivered but query still failed.
            # Likely a fidelity rejection or some other late-stage check.
            r["_outcome"] = "all_pairs_delivered_but_failed"

    distances = sorted({int(r["inter_node_distance_km"]) for r in rows})
    hops = sorted({r["hop_distance"] for r in rows if r["hop_distance"] > 0})

    if not distances or not hops:
        print("  chart5: no data")
        return

    outcome_order = [
        "success",
        "round2_incomplete",
        "round1_done_round2_not_started",
        "round1_incomplete",
        "never_started",
        "all_pairs_delivered_but_failed",
    ]
    outcome_colors = {
        "success": "#166534",                          # green
        "round2_incomplete": "#ea580c",                # orange
        "round1_done_round2_not_started": "#facc15",   # yellow
        "round1_incomplete": "#ca8a04",                # darker yellow/amber
        "never_started": "#6b7280",                    # grey
        "all_pairs_delivered_but_failed": "#9333ea",   # purple (rare)
    }

    backends = ["odo", "acp"]

    fig, axes = plt.subplots(len(backends), len(distances),
                             figsize=(3.3 * len(distances), 3.2 * len(backends)),
                             sharey=True, sharex=True)
    if len(backends) == 1:
        axes = axes[np.newaxis, :]
    if len(distances) == 1:
        axes = axes[:, np.newaxis]

    for bi, backend in enumerate(backends):
        for di, dist in enumerate(distances):
            ax = axes[bi, di]
            subset = [r for r in rows
                      if r["_backend"] == backend
                      and int(r["inter_node_distance_km"]) == dist]

            # Count outcomes per hop
            counts = defaultdict(lambda: defaultdict(int))
            for r in subset:
                counts[r["hop_distance"]][r["_outcome"]] += 1

            x = np.arange(len(hops))
            bottom = np.zeros(len(hops))

            for outcome in outcome_order:
                vals = []
                for h in hops:
                    total = sum(counts[h].values())
                    vals.append(counts[h][outcome] / total if total else 0)
                vals = np.array(vals)
                ax.bar(x, vals * 100, bottom=bottom * 100,
                       color=outcome_colors[outcome],
                       label=outcome if (bi == 0 and di == 0) else None,
                       edgecolor="white", linewidth=0.3)
                bottom += vals

            ax.set_xticks(x)
            ax.set_xticklabels(hops)
            if bi == len(backends) - 1:
                ax.set_xlabel("Hop distance")
            if di == 0:
                ax.set_ylabel(f"{backend.upper()}\n% of queries")
            ax.set_title(f"{dist} km" if bi == 0 else "")
            ax.set_ylim(0, 100)

    # Shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    pretty = {
        "success": "Success",
        "round2_incomplete": "Round 2 partial (>R1, <full)",
        "round1_done_round2_not_started": "Round 1 done, Round 2 never started",
        "round1_incomplete": "Round 1 partial",
        "never_started": "Never started (0 pairs)",
        "all_pairs_delivered_but_failed": "All pairs delivered, query failed",
    }
    labels = [pretty.get(l, l) for l in labels]
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.10), frameon=False, fontsize=9)

    fig.suptitle("Query outcome decomposition",
                 fontsize=13, y=1.00)

    path = os.path.join(output_dir, "chart5_failure_decomp.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# CHART 6 — Time to first vs last pair
# ---------------------------------------------------------------------------

def chart6_pair_arrival(rows: List[dict], output_dir: str):
    """Pair-arrival timeline: TTFP, end-of-round-1, end-of-round-2 (medians).

    Only includes successful queries (so all 4n+2 pairs were delivered),
    and only plots cells with at least MIN_N_FOR_CELL successes to avoid
    selection bias from a handful of lucky completions.
    """
    MIN_N_FOR_CELL = 10

    for r in rows:
        r["_backend"] = normalize_backend(r["backend"])

    # Only successful queries — for those, we know the full pair arrival sequence
    # and can split round 1 from round 2 cleanly.
    success_rows = [r for r in rows if r["success"] and r["pair_arrival_list"]]

    distances = sorted({int(r["inter_node_distance_km"]) for r in success_rows})
    hops = sorted({r["hop_distance"] for r in success_rows
                   if r["hop_distance"] > 0})

    if not distances or not hops:
        print("  chart6: no data")
        return

    # For each cell, collect three quantities per query:
    #   ttfp:   first pair (min of full list)
    #   ttlp1:  last pair of round 1 (the (2n+1)-th pair, sorted)
    #   ttlp2:  last pair of round 2 (max of full list)
    per_cell = defaultdict(lambda: {"ttfp": [], "ttlp1": [], "ttlp2": []})
    for r in success_rows:
        pairs = sorted(r["pair_arrival_list"])
        n = r.get("database_size_log", 10)
        pairs_per_round = 2 * n + 1
        if len(pairs) < 2 * pairs_per_round:
            # Successful queries should have exactly 2*(2n+1) pairs.
            # If fewer, something's off — skip rather than mislead.
            continue
        key = (r["_backend"], int(r["inter_node_distance_km"]),
               r["hop_distance"])
        per_cell[key]["ttfp"].append(pairs[0])
        per_cell[key]["ttlp1"].append(pairs[pairs_per_round - 1])
        per_cell[key]["ttlp2"].append(pairs[-1])

    fig, axes = plt.subplots(1, len(distances),
                             figsize=(4.0 * len(distances), 4.6),
                             sharey=True)
    if len(distances) == 1:
        axes = [axes]

    for ax, dist in zip(axes, distances):
        for backend, color, marker in [("odo", C_ODO, "s"),
                                        ("acp", C_ACP, "o")]:
            ttfp_med = []
            ttlp1_med = []
            ttlp2_med = []
            valid_hops = []
            for h in hops:
                cell = per_cell.get((backend, dist, h))
                if cell and len(cell["ttfp"]) >= MIN_N_FOR_CELL:
                    ttfp_med.append(np.median(cell["ttfp"]))
                    ttlp1_med.append(np.median(cell["ttlp1"]))
                    ttlp2_med.append(np.median(cell["ttlp2"]))
                    valid_hops.append(h)

            if not valid_hops:
                continue

            ax.plot(valid_hops, ttfp_med, marker=marker, color=color,
                    linewidth=2.2, markersize=7, linestyle="-",
                    label=f"{backend.upper()} first pair")
            ax.plot(valid_hops, ttlp1_med, marker=marker, color=color,
                    linewidth=1.6, markersize=7, linestyle="--",
                    alpha=0.85,
                    label=f"{backend.upper()} last pair, R1")
            ax.plot(valid_hops, ttlp2_med, marker=marker, color=color,
                    linewidth=1.4, markersize=7, linestyle=":",
                    alpha=0.6,
                    label=f"{backend.upper()} last pair, R2")

        ax.set_xlabel("Hop distance from QDC")
        ax.set_title(f"{dist} km links")
        ax.set_xticks(hops)

    axes[0].set_ylabel("Time from query start (ms)")
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)

    fig.suptitle("Pair arrival timeline: first pair, end of R1, end of R2",
                 fontsize=13, y=1.02)
    fig.text(0.5, -0.03,
             f"Successful queries only; cells with n<{MIN_N_FOR_CELL} suppressed",
             ha="center", fontsize=9, color="#444", style="italic")

    path = os.path.join(output_dir, "chart6_pair_arrival.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate all six QPQ charts")
    parser.add_argument("--sweep-dir", type=str, default="sweep2d_output",
                        help="Directory with primary_sweep.csv and dbsize_sweep.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to write PNGs (default: <sweep-dir>/figures)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.sweep_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    primary_csv = os.path.join(args.sweep_dir, "primary_sweep.csv")
    dbsize_csv = os.path.join(args.sweep_dir, "dbsize_sweep.csv")

    print(f"Reading {primary_csv}")
    primary_rows = load_csv(primary_csv)
    print(f"  {len(primary_rows)} rows\n")

    print(f"Reading {dbsize_csv}")
    dbsize_rows = load_csv(dbsize_csv)
    print(f"  {len(dbsize_rows)} rows\n")

    print("Generating charts...")
    if primary_rows:
        chart1_success_heatmap(primary_rows, output_dir)
        chart2_tts_small_multiples(primary_rows, output_dir)
        chart3_fidelity_lines(primary_rows, output_dir)
        chart5_failure_decomp(primary_rows, output_dir)
        chart6_pair_arrival(primary_rows, output_dir)
    if dbsize_rows:
        chart4_dbsize_scaling(dbsize_rows, output_dir)

    print(f"\nDone. Figures in {output_dir}/")


if __name__ == "__main__":
    main()