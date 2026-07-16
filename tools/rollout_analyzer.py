#!/usr/bin/env python3
"""Offline rollout analyzer for the Megatron RL pipeline.

Loads rollout JSON files collected via --skip-train --perform-rl-step (with the
LANGRL_LOG_DIR environment variable set, which enables the per-iteration rollout
JSON dump), estimates per-level tail factors from the actual rollout
length distribution, and predicts utilization and first-token staleness for all
granularity modes (B/B, G/G, G/B, R/G, R/B, plus the GRPO-unrealizable R/R
reference; G/E, E/E, E/B, R/E for multi-environment runs).

Also recommends matched capacity weights for the E rung (C_i ∝ μ_i · P_i),
shows the utilization cost of the default length-blind allocation, and renders
PNG plots: the rollout-length distribution (all environments + per-env) and,
for each valid mode, the expected first-token lag distribution (all
environments + per-env) with its mean and speed-up over B/B.

This is a thin CLI: the tail-factor math and mode predictions live in
``megatron.rl.rollout_capacity`` (shared with the live OnlineCapacityController);
this file only loads JSON, calls those functions, and prints.

Theory background
-----------------
Each level of the rollout hierarchy has a "tail factor" -- the ratio of the
expected maximum of its children's completions to the expected child completion:

    λ_RG = E[max of K rollout lengths] / E[rollout length]    (within a group)
    λ_GE = E[max of P_i group maxima]  / E[group max]         (within an env)
    λ_EB = E[max_i env_max_i]          / wmean_i E[env_max_i] (across envs)
    λ_GB = λ_GE · λ_EB                                        (single-env split)

Cumulative tails (product along the hierarchy):
    τ_K = λ_RG          (group rung)
    τ_E = τ_K · λ_GE   (environment rung; single-env: τ_E = τ_N)
    τ_N = τ_E · λ_EB   (batch rung)

Utilization and first-token staleness (suspend regime, engine sized to gate):
    U    = 1 / τ_submit
    D    = (τ_consume / τ_submit) · (1 + L)     [B/B: D = L exactly]

Usage
-----
# Step 1 -- collect data on the training cluster (no optimizer needed):
#   LANGRL_LOG_DIR=/path/to/logs torchrun ... train_rl.py \\
#       --skip-train --perform-rl-step --no-load-optim \\
#       --exit-interval 40 [other RL args ...]

# Step 2 -- analyze offline (no GPU; imports the shared core from
#           megatron.rl.rollout_capacity, so run from the repo root):
#   python3 tools/rollout_analyzer.py --rollout-dir /path/to/logs --lag 2
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from megatron.rl.rollout_capacity import (
    capacity_recommendation,
    compute_levels,
    predict_mode,
    valid_modes,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _rollout_length(rollout: dict) -> int:
    """Total token (or character) length of a rollout across all turns."""
    traj = rollout.get("trajectory", [])
    if not traj:
        return 0
    first = traj[0]
    if isinstance(first, list):
        # TokenRollout: trajectory is list[list[int]]
        return sum(len(t) for t in traj)
    else:
        # Rollout: trajectory is list[str]
        return sum(len(t) for t in traj)


def load_batches(rollout_dir: str, rank: int = 0) -> list[list[dict]]:
    """Load rollout JSON files and return a list of batches.

    Each batch is a list of group dicts:
        {"env_id": str, "lengths": [int, ...]}   # one entry per rollout in the group

    Files are matched on rank and sorted by iteration number so batches are in
    the order they were collected.
    """
    pattern = os.path.join(rollout_dir, f"rollouts_rank{rank}_iteration*_*.json")
    files = sorted(glob.glob(pattern), key=lambda p: _iteration_of(p))
    if not files:
        # Fallback: any JSON file in the directory
        files = sorted(glob.glob(os.path.join(rollout_dir, "*.json")))
    if not files:
        raise FileNotFoundError(
            f"No rollout JSON files found in {rollout_dir!r}.\n"
            "Make sure you ran with LANGRL_LOG_DIR pointing to that directory."
        )

    batches = []
    for path in files:
        try:
            with open(path) as f:
                raw = json.load(f)  # [[rollout_dict, ...], ...]  (groups x rollouts)
        except Exception as e:
            print(f"[warn] skipping {path}: {e}", file=sys.stderr)
            continue
        batch = []
        for group in raw:
            if not group:
                continue
            env_id = group[0].get("env_id", "")
            lengths = [_rollout_length(r) for r in group]
            batch.append({"env_id": env_id, "lengths": lengths})
        if batch:
            batches.append(batch)
    return batches


def _iteration_of(path: str) -> int:
    stem = Path(path).stem  # rollouts_rank0_iteration7_math
    for part in stem.split("_"):
        if part.startswith("iteration") and part[9:].isdigit():
            return int(part[9:])
    return 0


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def _bar(frac: float, width: int = 20) -> str:
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


def print_report(lv: dict, lag: int, total_slots: int | None = None) -> None:
    P, K, V, N = lv["P"], lv["K"], lv["V"], lv["N"]
    mu = lv["mu"]
    n_batches = lv["n_batches"]

    print()
    print("=" * 72)
    print(f"  Rollout length profile   "
          f"({n_batches} batches · P={P} groups · K={K} rollouts/group · V={V} env)")
    print("=" * 72)
    print(f"  overall mean length : {mu:.1f}")
    print(f"  lag L               : {lag}   (gate holds {lag+1} batches in flight)")
    print()

    if V > 1:
        print("  Per-environment breakdown:")
        hdr = f"  {'env':<24}  {'P_i':>4}  {'μ_i':>8}  {'std':>7}  {'cv':>5}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for eid in lv["env_ids"]:
            e = lv["envs"][eid]
            print(f"  {eid:<24}  {e['P_i']:>4}  {e['mu_i']:>8.1f}  "
                  f"{e['std_i']:>7.1f}  {e['cv_i']:>5.2f}")
        print()

    print("  Per-level tail factors  (empirical; each estimated at its own level):")
    print(f"    λ_RG = τ_K  = {lv['lambda_RG']:.3f}   rollout lengths within a group")
    if V > 1:
        print(f"    λ_GE        = {lv['lambda_GE']:.3f}   group completions within an environment")
        print(f"    λ_EB        = {lv['lambda_EB']:.3f}   environment completions within a batch")
    print(f"    λ_GB        = {lv['lambda_GB']:.3f}   group completions within a batch  "
          f"{'(= λ_GE · λ_EB)' if V > 1 else ''}")
    print(f"    τ_K         = {lv['tau_K']:.3f}")
    if V > 1:
        print(f"    τ_E         = {lv['tau_E']:.3f}   (τ_K · λ_GE)")
    print(f"    τ_N         = {lv['tau_N']:.3f}   (τ_K · λ_GB)")
    print()

    # ---- mode table --------------------------------------------------------
    modes = valid_modes(V)

    col_w = [8, 10, 18, 18, 12]
    header = (
        f"  {'mode':<{col_w[0]}}  "
        f"{'util':>{col_w[1]}}  "
        f"{'first-staleness':>{col_w[2]}}  "
        f"{'staleness-infl':>{col_w[3]}}  "
        f"{'within-span':>{col_w[4]}}"
    )
    print("  Granularity mode analysis   (suspend regime, engine sized S = (1+L)·N):")
    print(header)
    print("  " + "-" * (sum(col_w) + 4 * 2))

    prev_sub = None
    for sub, con in modes:
        if prev_sub is not None and sub != prev_sub:
            print()
        prev_sub = sub

        p = predict_mode(sub, con, lv, lag)
        tag = ""
        if sub == con:
            tag = "← diagonal (matched)"
        if (sub, con) == ("R", "R"):
            tag = "← GRPO-unrealizable reference"
        if (sub, con) == ("B", "B"):
            tag = f"← phase-locked; staleness exactly {lag}"

        util_str = f"{p['util']*100:.0f}%"
        first_str = f"{p['first']:.2f}"
        infl_str = f"{p['inflation']:.3f}"
        span_str = f"{p['span']:.2f}"
        mode_str = p['mode']
        if (sub, con) == ("R", "R"):
            mode_str = "R/R†"

        print(
            f"  {mode_str:<{col_w[0]}}  "
            f"{util_str:>{col_w[1]}}  "
            f"{first_str:>{col_w[2]}}  "
            f"{infl_str:>{col_w[3]}}  "
            f"{span_str:>{col_w[4]}}"
            + (f"  {tag}" if tag else "")
        )

    print()
    print("  Columns: util = engine utilization; first-staleness = expected versions "
          "since first token was generated;\n"
          "  staleness-inflation = factor over (1+L) baseline; within-span = version "
          "spread within one rollout.")

    # ---- capacity recommendation -------------------------------------------
    if V > 1:
        cap = capacity_recommendation(lv, total_slots)
        print()
        print("  Matched capacity recommendation for the E rung  "
              "(C_i ∝ μ_i · P_i, THEORY.tex §caps):")
        hdr2 = (f"  {'env':<24}  {'P_i':>4}  {'μ_i':>8}  "
                f"{'matched %':>10}  {'blind %':>8}  {'equal %':>8}")
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))
        for eid in lv["env_ids"]:
            c = cap[eid]
            slots_str = ""
            if total_slots is not None:
                slots_str = f"  ({c['matched_slots']} / {c['blind_slots']} slots)"
            print(f"  {eid:<24}  {c['P_i']:>4}  "
                  f"{lv['envs'][eid]['mu_i']:>8.1f}  "
                  f"{c['matched_frac']*100:>9.1f}%  "
                  f"{c['blind_frac']*100:>7.1f}%  "
                  f"{c['equal_frac']*100:>7.1f}%"
                  + slots_str)
        print()
        print("  blind = current default (C_i ∝ P_i, ignores rollout length).")
        print("  matched = length-aware (equalized delivery rates; see THEORY.tex §caps).")
        print("  The matched weights are the recommended initial values for")
        print("  WeightedMultiTask.set_capacity_weights().")
        _print_matched_weights(lv)

    print()
    print("=" * 72)


def _print_matched_weights(lv: dict) -> None:
    """Print matched capacity weights as a ready-to-use Python snippet."""
    envs = lv["envs"]
    total = sum(e["matched_weight"] for e in envs.values())
    weights = [envs[eid]["matched_weight"] / total for eid in lv["env_ids"]]
    print()
    print("  # Snippet for set_capacity_weights() (normalized matched weights):")
    print(f"  # agent.set_capacity_weights([")
    for eid, w in zip(lv["env_ids"], weights):
        print(f"  #     {w:.4f},  # {eid}")
    print(f"  # ])")


# ---------------------------------------------------------------------------
# Speed-up / trade-off summary
# ---------------------------------------------------------------------------

def print_speedup_summary(lv: dict, lag: int) -> None:
    """Print the speed-up of every valid mode over the B/B baseline.

    Speed-up is the utilization ratio U_mode / U_B/B = τ_N / τ_submit: with a
    right-sized engine, generation throughput (hence generation wall-clock per
    batch) scales with utilization.
    """
    base = predict_mode("B", "B", lv, lag)

    print()
    print("  Speed-up over B/B  (speed-up = utilization ratio = generation-"
          "throughput gain):")
    hdr = (f"  {'mode':<8}  {'util':>6}  {'speed-up':>9}  "
           f"{'first-staleness':>16}  {'stale vs B/B':>13}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    prev_sub = None
    for sub, con in valid_modes(lv["V"]):
        if prev_sub is not None and sub != prev_sub:
            print()
        prev_sub = sub

        p = predict_mode(sub, con, lv, lag)
        speedup = p["util"] / base["util"]
        stale_str = (f"×{p['first'] / base['first']:.2f}"
                     if base["first"] > 0 else "n/a")

        mode_str = p["mode"]
        tag = ""
        if (sub, con) == ("B", "B"):
            tag = "← baseline (phase-locked)"
        elif (sub, con) == ("R", "R"):
            mode_str = "R/R*"
            tag = "← not achievable with GRPO"
        elif sub == con:
            tag = "← diagonal (matched)"

        print(f"  {mode_str:<8}  {p['util']*100:>5.0f}%  ×{speedup:>7.2f}  "
              f"{p['first']:>16.2f}  {stale_str:>13}"
              + (f"  {tag}" if tag else ""))

    print()
    print("  * R/R needs a per-rollout training unit (e.g. a critic baseline); "
          "GRPO's\n    group-relative advantage cannot realize it.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# Chart tokens (light mode) and fixed-order categorical series slots.
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_SURFACE = "#fcfcfb"
_SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
           "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]

# What one histogram sample is, per consumption granularity.
_CONSUME_UNIT = {
    "R": "one sample per rollout",
    "G": "one sample per group (max of its K rollouts)",
    "E": "one sample per environment unit, weighted by its P_i groups",
    "B": "one sample per batch (straggler shared by all its groups)",
}


def _env_color(idx: int, V: int) -> str:
    # Fixed slot per environment, never cycled; past 8 envs identity is
    # carried by the panel title alone.
    return _SERIES[idx] if V <= len(_SERIES) else _SERIES[0]


def _short(env_id: str, n: int = 30) -> str:
    return env_id if len(env_id) <= n else env_id[: n - 1] + "…"


def _style_axis(ax) -> None:
    ax.set_facecolor(_SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_AXIS)
    ax.spines["bottom"].set_color(_AXIS)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _mean_line(ax, mean: float, label: str) -> None:
    ax.axvline(mean, color=_INK, linestyle="--", linewidth=1.2)
    ax.annotate(label, xy=(mean, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -2), textcoords="offset points",
                ha="left", va="top", fontsize=8, color=_INK)


def _hist(ax, np, values, color, bins, weights=None) -> float:
    """Density histogram; returns the (weighted) sample mean."""
    vals = np.asarray(values, dtype=float)
    ax.hist(vals, bins=bins, weights=weights, density=True,
            color=color, edgecolor=_SURFACE, linewidth=0.5)
    return float(np.average(vals, weights=weights))


def _shared_bins(np, values):
    lo, hi = float(min(values)), float(max(values))
    if hi <= lo:
        hi = lo + 1.0
    n = max(15, min(60, int(len(values) ** 0.5)))
    return np.linspace(lo, hi, n + 1)


def _point_mass(ax, x: float, color: str) -> None:
    """Degenerate distribution: all mass at x (the B/B phase-locked lag)."""
    ax.plot([x, x], [0.0, 1.0], color=color, linewidth=6, solid_capstyle="butt")
    ax.set_ylim(0, 1.15)
    ax.set_xlim(0, max(2.0 * x, x + 1.0))


def _panel_figure(plt, lv: dict, suptitle: str):
    """One full-width all-env panel on top; per-env small multiples below (V>1)."""
    V = lv["V"]
    if V > 1:
        ncols = min(3, V)
        nrows = -(-V // ncols)
        fig = plt.figure(figsize=(3.6 * ncols, 3.0 + 2.3 * nrows), dpi=150,
                         layout="constrained")
        gs = fig.add_gridspec(nrows + 1, ncols,
                              height_ratios=[1.35] + [1.0] * nrows)
        ax_all = fig.add_subplot(gs[0, :])
        env_axes = [fig.add_subplot(gs[1 + i // ncols, i % ncols])
                    for i in range(V)]
    else:
        fig = plt.figure(figsize=(7.5, 4.2), dpi=150, layout="constrained")
        ax_all = fig.add_subplot(1, 1, 1)
        env_axes = []
    fig.patch.set_facecolor(_SURFACE)
    fig.suptitle(suptitle, color=_INK, fontsize=12, fontweight="bold")
    return fig, ax_all, env_axes


def plot_length_distributions(plt, np, lv: dict, plot_dir: str) -> str:
    """Histogram of rollout lengths: all environments + per-env panels."""
    s = lv["samples"]
    V = lv["V"]
    fig, ax_all, env_axes = _panel_figure(plt, lv, "Rollout length distribution")
    bins = _shared_bins(np, s["lengths"])

    mean = _hist(ax_all, np, s["lengths"], _SERIES[0], bins)
    _style_axis(ax_all)
    ax_all.set_title(f"all environments — {len(s['lengths'])} rollouts, "
                     f"{lv['n_batches']} batches", loc="left",
                     fontsize=9, color=_INK_2)
    _mean_line(ax_all, mean, f"μ = {mean:.0f}")
    ax_all.set_xlabel("rollout length", fontsize=8, color=_MUTED)
    ax_all.set_ylabel("density", fontsize=8, color=_MUTED)

    for i, (eid, ax) in enumerate(zip(lv["env_ids"], env_axes)):
        mean_i = _hist(ax, np, s["env_lengths"][eid], _env_color(i, V), bins)
        _style_axis(ax)
        ax.set_title(_short(eid), loc="left", fontsize=9, color=_INK)
        _mean_line(ax, mean_i, f"μᵢ = {mean_i:.0f}")
        ax.set_xlabel("rollout length", fontsize=8, color=_MUTED)

    path = os.path.join(plot_dir, "rollout_lengths.png")
    fig.savefig(path, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def _lag_samples_all(con: str, lv: dict):
    """(values, weights) of realized consume-unit completions, batch-wide."""
    s = lv["samples"]
    if con == "R":
        return s["lengths"], None
    if con == "G":
        return s["group_maxima"], None
    if con == "E":
        vals, wts = [], []
        for eid in lv["env_ids"]:
            ems = s["env_maxima"][eid]
            vals.extend(ems)
            wts.extend([lv["envs"][eid]["P_i"]] * len(ems))  # unit spans P_i groups
        return vals, wts
    return s["batch_maxima"], None  # "B": every group shares the straggler


def _lag_samples_env(con: str, lv: dict, eid: str):
    """Realized consume-unit completions restricted to one environment."""
    s = lv["samples"]
    if con == "R":
        return s["env_lengths"][eid]
    if con == "G":
        return s["env_group_maxima"][eid]
    if con == "E":
        return s["env_maxima"][eid]
    return s["batch_maxima"]  # "B": the global straggler gates every env


def plot_lag_distributions(plt, np, lv: dict, lag: int, plot_dir: str) -> list[str]:
    """One figure per valid mode: expected first-token lag distribution.

    Realized per-unit lag is (1+L)/τ_s · T̂_c/μ with T̂_c the consume-unit
    completion (THEORY.tex §dist / §rr); B/B is a point mass at exactly L.
    """
    V = lv["V"]
    base_util = predict_mode("B", "B", lv, lag)["util"]
    paths = []

    for sub, con in valid_modes(V):
        p = predict_mode(sub, con, lv, lag)
        speedup = p["util"] / base_util
        is_rr = (sub, con) == ("R", "R")
        is_bb = (sub, con) == ("B", "B")
        mode_label = "R/R*" if is_rr else p["mode"]

        fig, ax_all, env_axes = _panel_figure(
            plt, lv, f"Expected lag distribution — {mode_label}   (L = {lag})")
        panels = [(ax_all, "all environments", _SERIES[0], None)] + [
            (ax, _short(eid), _env_color(i, V), eid)
            for i, (eid, ax) in enumerate(zip(lv["env_ids"], env_axes))
        ]

        if is_bb:
            for ax, name, color, _ in panels:
                _point_mass(ax, lag, color)
                _style_axis(ax)
                ax.set_title(name, loc="left", fontsize=9, color=_INK_2)
                _mean_line(ax, lag, f"mean = {lag:.2f} (exact)")
                ax.set_xlabel("first-token staleness (policy versions)",
                              fontsize=8, color=_MUTED)
            ax_all.set_ylabel("probability mass", fontsize=8, color=_MUTED)
            ax_all.set_title("all environments — phase-locked: deterministic "
                             "lag, zero spread", loc="left", fontsize=9,
                             color=_INK_2)
        else:
            tau_s = {"R": 1.0, "G": lv["tau_K"], "E": lv["tau_E"],
                     "B": lv["tau_N"]}[sub]
            scale = (1 + lag) / (tau_s * lv["mu"])
            vals, wts = _lag_samples_all(con, lv)
            scaled_all = [v * scale for v in vals]
            bins = _shared_bins(np, scaled_all)

            mean_all = _hist(ax_all, np, scaled_all, _SERIES[0], bins, weights=wts)
            _style_axis(ax_all)
            ax_all.set_title(f"all environments — {_CONSUME_UNIT[con]}",
                             loc="left", fontsize=9, color=_INK_2)
            _mean_line(ax_all, mean_all, f"mean = {mean_all:.2f}")
            ax_all.set_xlabel("first-token staleness (policy versions)",
                              fontsize=8, color=_MUTED)
            ax_all.set_ylabel("density", fontsize=8, color=_MUTED)

            for ax, name, color, eid in panels[1:]:
                scaled_i = [v * scale for v in _lag_samples_env(con, lv, eid)]
                mean_i = _hist(ax, np, scaled_i, color, bins)
                _style_axis(ax)
                ax.set_title(name, loc="left", fontsize=9, color=_INK)
                _mean_line(ax, mean_i, f"mean = {mean_i:.2f}")
                ax.set_xlabel("first-token staleness (policy versions)",
                              fontsize=8, color=_MUTED)

        note = (f"mean lag = {p['first']:.2f} versions\n"
                f"speed-up over B/B: ×{speedup:.2f}")
        ax_all.text(0.98, 0.95, note, transform=ax_all.transAxes,
                    ha="right", va="top", fontsize=9, color=_INK_2,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=_SURFACE,
                              edgecolor=_GRID))

        footnotes = []
        if is_rr:
            footnotes.append("* not achievable with GRPO — needs a per-rollout "
                             "training unit; GRPO's group-relative advantage "
                             "cannot realize it")
        if "E" in (sub, con):
            footnotes.append("E-rung values are the streaming-ideal lower bound "
                             "(balanced-E, matched capacity; THEORY.tex §caps)")
        if footnotes:
            fig.text(0.01, -0.01, "\n".join(footnotes), ha="left", va="top",
                     fontsize=8, color=_INK_2)

        path = os.path.join(plot_dir, f"lag_distribution_{sub}_{con}.png")
        fig.savefig(path, facecolor=_SURFACE, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


def save_plots(lv: dict, lag: int, plot_dir: str) -> None:
    """Render the length and per-mode lag distribution PNGs into plot_dir."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"[warn] plots skipped (matplotlib unavailable: {e})", file=sys.stderr)
        return

    os.makedirs(plot_dir, exist_ok=True)
    paths = [plot_length_distributions(plt, np, lv, plot_dir)]
    paths += plot_lag_distributions(plt, np, lv, lag, plot_dir)

    print()
    print(f"  Plots written to {plot_dir}:")
    for path in paths:
        print(f"    {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--rollout-dir", required=True,
        help="Directory containing rollout JSON files written when LANGRL_LOG_DIR is set.",
    )
    p.add_argument(
        "--rank", type=int, default=0,
        help="Rank whose files to load (default: 0).",
    )
    p.add_argument(
        "--lag", type=int, default=2,
        help="rl_generation_lag L (default: 2).",
    )
    p.add_argument(
        "--total-slots", type=int, default=None,
        help="Total engine KV slots S. When given, capacity recommendation "
             "also shows absolute slot counts. Default: (1+L)·N (right-sized).",
    )
    p.add_argument(
        "--max-batches", type=int, default=None,
        help="Cap the number of batches analyzed (default: all).",
    )
    p.add_argument(
        "--trade-offs", action="store_true",
        help="Print a concise speed-up / trade-off summary in addition to the main table.",
    )
    p.add_argument(
        "--plot-dir", default=None,
        help="Directory for PNG distribution plots (default: <rollout-dir>/plots).",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating distribution plots.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    print(f"Loading rollout data from {args.rollout_dir!r} (rank {args.rank}) ...",
          file=sys.stderr)
    batches = load_batches(args.rollout_dir, rank=args.rank)
    if args.max_batches:
        batches = batches[: args.max_batches]
    print(f"  loaded {len(batches)} batches.", file=sys.stderr)

    lv = compute_levels(batches)

    total_slots = args.total_slots
    if total_slots is None:
        total_slots = (1 + args.lag) * lv["N"]

    print_report(lv, lag=args.lag, total_slots=total_slots)

    if args.trade_offs:
        print_speedup_summary(lv, lag=args.lag)

    if not args.no_plots:
        plot_dir = args.plot_dir or os.path.join(args.rollout_dir, "plots")
        save_plots(lv, lag=args.lag, plot_dir=plot_dir)


if __name__ == "__main__":
    main()
