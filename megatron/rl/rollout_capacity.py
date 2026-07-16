# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Rollout capacity: shared tail-factor computation core plus an online controller.

The computation core (``compute_levels`` / ``predict_mode`` /
``capacity_recommendation`` / ``valid_modes``) has no Megatron dependencies and
estimates the per-level tail factors (``tau_K``, ``tau_E``, ``tau_N``,
``lambda_RG``, ``lambda_GE``, ``lambda_EB``) of the batch ⊃ env ⊃ group ⊃
rollout hierarchy and the utilization / staleness predictions of each
granularity mode (see THEORY.tex). It is shared with the offline
``tools/rollout_analyzer.py`` CLI.

``OnlineCapacityController`` runs incrementally alongside live rollout collection:
it is fed one group at a time as groups exit the consumer (``update_group``) and
finalized once per batch (``finalize_batch``), maintaining a rolling window and
per-env length EMAs. After warmup it pushes matched capacity weights
(``w_i ∝ ema_mu_i · P_i``) to a ``WeightedMultiTask`` agent, and — when
``diagnostics`` is set — computes the full tail-factor picture and per-mode
predictions for wandb logging each step.
"""
import statistics
from collections import defaultdict


# ---------------------------------------------------------------------------
# Granularity modes
# ---------------------------------------------------------------------------

def valid_modes(V: int) -> list[tuple[str, str]]:
    """All valid (submission, consumption) modes, in display order.

    Submission coarse-to-fine, consumption fine-to-coarse within each submission
    row; consumption is never finer than submission. R/R is the GRPO-unrealizable
    reference; the E rung exists only for V > 1 (single-env E coincides with B).
    """
    if V > 1:
        return [
            ("B", "B"),
            ("E", "E"), ("E", "B"),
            ("G", "G"), ("G", "E"), ("G", "B"),
            ("R", "R"), ("R", "G"), ("R", "E"), ("R", "B"),
        ]
    return [("B", "B"), ("G", "G"), ("G", "B"), ("R", "R"), ("R", "G"), ("R", "B")]


# ---------------------------------------------------------------------------
# Tail-factor estimation from empirical batch data
# ---------------------------------------------------------------------------

def compute_levels(batches: list[list[dict]]) -> dict:
    """Estimate per-level tail factors from collected batch data.

    Args:
        batches: list of batches, each a list of group dicts
            ``{"env_id": str, "lengths": [int, ...]}`` (one entry per rollout).

    Returns a dict with:
        mu          -- overall mean rollout length
        tau_K       -- cumulative group tail  (= lambda_RG)
        tau_E       -- cumulative env tail    (= tau_K * lambda_GE)
        tau_N       -- cumulative batch tail  (= tau_E * lambda_EB)
        lambda_RG   -- rollout-within-group edge
        lambda_GE   -- group-within-env edge
        lambda_EB   -- env-within-batch edge
        lambda_GB   -- group-within-batch edge (= tau_N / tau_K)
        envs        -- per-env stats dict keyed by env_id
        P           -- groups per batch (inferred)
        K           -- rollouts per group (inferred)
        V           -- number of distinct environments
        n_batches   -- number of batches analyzed
        samples     -- raw sample arrays behind the estimates (for plotting):
            lengths / group_maxima / batch_maxima, plus per-env dicts
            env_lengths / env_group_maxima / env_maxima (per-batch env max).
    """
    if not batches:
        raise ValueError("No batches to analyze.")

    # Infer batch shape from the first batch
    P = len(batches[0])
    K = len(batches[0][0]["lengths"])

    # Per-env group count (P_i): count groups per env_id in first batch
    env_group_counts: dict[str, int] = defaultdict(int)
    for group in batches[0]:
        env_group_counts[group["env_id"]] += 1
    env_ids_ordered = list(env_group_counts.keys())
    V = len(env_ids_ordered)

    # Accumulators
    all_lengths: list[float] = []       # every rollout length
    group_maxima: list[float] = []      # max length per group
    batch_maxima: list[float] = []      # max group-max per batch
    env_maxima: dict[str, list[float]] = defaultdict(list)  # per-env max per batch

    env_lengths: dict[str, list[float]] = defaultdict(list)  # for per-env mean
    env_group_maxima: dict[str, list[float]] = defaultdict(list)  # per-env group maxima

    for batch in batches:
        b_group_maxima_by_env: dict[str, list[float]] = defaultdict(list)
        b_all_group_maxima: list[float] = []

        for group in batch:
            eid = group["env_id"]
            gmax = max(group["lengths"])
            group_maxima.append(gmax)
            all_lengths.extend(group["lengths"])
            env_lengths[eid].extend(group["lengths"])
            env_group_maxima[eid].append(gmax)
            b_group_maxima_by_env[eid].append(gmax)
            b_all_group_maxima.append(gmax)

        batch_maxima.append(max(b_all_group_maxima))
        for eid, gmaxes in b_group_maxima_by_env.items():
            env_maxima[eid].append(max(gmaxes))

    mu = statistics.fmean(all_lengths)
    mean_group_max = statistics.fmean(group_maxima)
    tau_K = mean_group_max / mu

    # tau_N: mean of batch maxima (each batch_max is max over P group maxima)
    tau_N = statistics.fmean(batch_maxima) / mu

    # tau_E: weighted mean of per-env expected env-maxima
    # weight by n_i / N (= P_i * K / (P * K) = P_i / P)
    N = P * K
    tau_E_num = 0.0
    for eid in env_ids_ordered:
        Pi = env_group_counts[eid]
        ni = Pi * K
        mean_env_max = statistics.fmean(env_maxima[eid]) if env_maxima[eid] else mu
        tau_E_num += (ni / N) * mean_env_max
    tau_E = tau_E_num / mu

    lambda_RG = tau_K
    lambda_GE = tau_E / tau_K if tau_K > 0 else 1.0
    lambda_EB = tau_N / tau_E if tau_E > 0 else 1.0
    lambda_GB = tau_N / tau_K if tau_K > 0 else 1.0

    # Per-env stats
    envs = {}
    for eid in env_ids_ordered:
        Pi = env_group_counts[eid]
        lens = env_lengths[eid]
        mu_i = statistics.fmean(lens) if lens else 0.0
        std_i = statistics.pstdev(lens) if len(lens) > 1 else 0.0
        cv_i = std_i / mu_i if mu_i > 0 else 0.0
        envs[eid] = {
            "P_i": Pi,
            "n_i": Pi * K,
            "mu_i": mu_i,
            "std_i": std_i,
            "cv_i": cv_i,
            # matched capacity weight (proportional to mu_i * P_i)
            "matched_weight": mu_i * Pi,
        }

    return {
        "mu": mu,
        "tau_K": tau_K,
        "tau_E": tau_E,
        "tau_N": tau_N,
        "lambda_RG": lambda_RG,
        "lambda_GE": lambda_GE,
        "lambda_EB": lambda_EB,
        "lambda_GB": lambda_GB,
        "envs": envs,
        "P": P,
        "K": K,
        "V": V,
        "N": N,
        "n_batches": len(batches),
        "env_ids": env_ids_ordered,
        "samples": {
            "lengths": all_lengths,
            "group_maxima": group_maxima,
            "batch_maxima": batch_maxima,
            "env_lengths": {eid: env_lengths[eid] for eid in env_ids_ordered},
            "env_group_maxima": {eid: env_group_maxima[eid] for eid in env_ids_ordered},
            "env_maxima": {eid: env_maxima[eid] for eid in env_ids_ordered},
        },
    }


# ---------------------------------------------------------------------------
# Theoretical predictions (suspend regime, engine sized to gate)
# ---------------------------------------------------------------------------

def _tau(rung: str, lv: dict) -> float:
    return {"R": 1.0, "G": lv["tau_K"], "E": lv["tau_E"], "B": lv["tau_N"]}[rung]


def predict_mode(submission: str, consumption: str, lv: dict, lag: int) -> dict:
    """Predict utilization and staleness for one (submission, consumption) mode."""
    tau_s = _tau(submission, lv)
    tau_c = _tau(consumption, lv)
    util = 1.0 / tau_s
    if submission == "B":
        # B/B phase lock: staleness = L exactly
        first = float(lag)
        avg = float(lag)
        last = float(lag)
        span = 0.0
    else:
        span = (1 + lag) / tau_s
        first = (tau_c / tau_s) * (1 + lag)
        avg = first - span / 2
        last = first - span
    inflation = tau_c / tau_s
    return {
        "mode": f"{submission}/{consumption}",
        "util": util,
        "first": first,
        "avg": avg,
        "last": last,
        "inflation": inflation,
        "span": span,
    }


# ---------------------------------------------------------------------------
# Capacity weight recommendation
# ---------------------------------------------------------------------------

def capacity_recommendation(lv: dict, total_slots: int | None = None) -> dict:
    """Compute matched and blind capacity allocations for each environment.

    Matched:  C_i ∝ μ_i · P_i   (THEORY.tex eq. caps)
    Blind:    C_i ∝ P_i          (current default, length-unaware)
    Equal:    C_i = S / V
    """
    envs = lv["envs"]
    if not envs or lv["V"] <= 1:
        return {}

    total_matched = sum(e["matched_weight"] for e in envs.values())
    total_blind = sum(e["P_i"] for e in envs.values())
    V = lv["V"]

    rec = {}
    for eid, e in envs.items():
        rec[eid] = {
            "P_i": e["P_i"],
            "mu_i": e["mu_i"],
            "matched_frac": e["matched_weight"] / total_matched,
            "blind_frac": e["P_i"] / total_blind,
            "equal_frac": 1.0 / V,
        }
        if total_slots is not None:
            rec[eid]["matched_slots"] = round(total_slots * rec[eid]["matched_frac"])
            rec[eid]["blind_slots"] = round(total_slots * rec[eid]["blind_frac"])
    return rec


# ---------------------------------------------------------------------------
# Online capacity controller
# ---------------------------------------------------------------------------

class OnlineCapacityController:
    """Incremental capacity controller run alongside live rollout collection.

    Fed one group at a time as groups exit the consumer (``update_group``) and
    finalized once per batch (``finalize_batch``). Maintains a rolling window of
    batch records and per-env length EMAs; after warmup, pushes matched capacity
    weights (``w_i ∝ ema_mu_i · P_i``) to the agent when it exposes
    ``set_capacity_weights``. With ``diagnostics=True`` it also computes the full
    tail-factor picture and per-mode predictions for wandb logging.
    """

    def __init__(
        self,
        agent,               # WeightedMultiTask instance, or None for single-env
        lag: int,            # rl_generation_lag
        ema_alpha: float = 0.05,
        window_size: int = 200,
        diagnostics: bool = False,
    ):
        self.agent = agent
        self.lag = lag
        self.ema_alpha = ema_alpha
        self.window_size = window_size
        self.diagnostics = diagnostics

        # Rolling window of past batch records; each record is a list of
        # {"env_id": str, "lengths": list[int]} group dicts.
        self._window: list[list[dict]] = []
        self._ema_mu: dict[str, float] = {}       # per env_id EMA of mean length
        self._batches_seen: int = 0
        # Don't push weights before the EMA has meaningfully converged.
        self._warmup_batches: int = max(10, int(1 / ema_alpha))
        self._current_batch: list[dict] = []      # accumulator for in-progress batch

    def update_group(self, group) -> None:
        """Record one RolloutGroup as it exits the consumer (batch still filling).

        Intentionally incremental: runs while subsequent rollouts of the same
        batch are still being generated, hiding the bookkeeping behind GPU work.
        """
        env_id = group.rollouts[0].env_id if group.rollouts else ""
        lengths = [sum(len(t) for t in r.trajectory) for r in group.rollouts]
        self._current_batch.append({"env_id": env_id, "lengths": lengths})

    def finalize_batch(self) -> dict:
        """Fold the completed batch into the window/EMA and return wandb metrics."""
        batch = self._current_batch
        self._current_batch = []
        if not batch:
            return {}

        # Roll the window.
        self._window.append(batch)
        if len(self._window) > self.window_size:
            self._window.pop(0)

        # Update per-env EMA of mean rollout length.
        alpha = self.ema_alpha
        batch_lengths_by_env: dict[str, list[float]] = defaultdict(list)
        for record in batch:
            batch_lengths_by_env[record["env_id"]].extend(record["lengths"])
        for eid, lens in batch_lengths_by_env.items():
            if not lens:
                continue
            mu_batch = statistics.fmean(lens)
            if eid in self._ema_mu:
                self._ema_mu[eid] = (1 - alpha) * self._ema_mu[eid] + alpha * mu_batch
            else:
                self._ema_mu[eid] = mu_batch

        self._batches_seen += 1

        metrics: dict = {"capacity/batches_seen": self._batches_seen}
        for eid, mu in self._ema_mu.items():
            metrics[f"capacity/ema_mu/{eid}"] = mu

        # Push matched capacity weights once the EMA has warmed up.
        if (
            self._batches_seen >= self._warmup_batches
            and self.agent is not None
            and hasattr(self.agent, "set_capacity_weights")
        ):
            weights = self._compute_matched_weights()
            ordered_ids = self._get_env_ids_ordered()
            if ordered_ids and all(eid in weights for eid in ordered_ids):
                self.agent.set_capacity_weights([weights[eid] for eid in ordered_ids])
                for eid in ordered_ids:
                    metrics[f"capacity/weight/{eid}"] = weights[eid]

        # Optional diagnostics: full tail-factor picture + per-mode predictions.
        if self.diagnostics and len(self._window) >= 2:
            import time

            t0 = time.monotonic()
            lv = compute_levels(self._window)
            levels_ms = (time.monotonic() - t0) * 1000.0

            t1 = time.monotonic()
            predictions = {
                f"{sub}/{con}": predict_mode(sub, con, lv, self.lag)
                for sub, con in valid_modes(lv["V"])
            }
            predict_ms = (time.monotonic() - t1) * 1000.0

            metrics["capacity/tau_K"] = lv["tau_K"]
            metrics["capacity/tau_E"] = lv["tau_E"]
            metrics["capacity/tau_N"] = lv["tau_N"]
            metrics["capacity/lambda_EB"] = lv["lambda_EB"]
            metrics["capacity/diagnostics_levels_ms"] = levels_ms
            metrics["capacity/diagnostics_predict_ms"] = predict_ms
            for mode, p in predictions.items():
                metrics[f"capacity/util/{mode}"] = p["util"]
                metrics[f"capacity/staleness/first/{mode}"] = p["first"]

        return metrics

    def _compute_matched_weights(self) -> dict[str, float]:
        """Matched weights ``w_i ∝ ema_mu_i · P_i``, normalized to sum to 1."""
        if self.agent is not None and hasattr(self.agent, "groups_per_agent"):
            total_groups = len(self._window[-1]) if self._window else 0
            groups_per_agent = self.agent.groups_per_agent(total_groups)
            ordered_ids = self._get_env_ids_ordered()
            raw = {
                eid: self._ema_mu.get(eid, 0.0) * P_i
                for eid, P_i in zip(ordered_ids, groups_per_agent)
            }
        else:
            # No per-agent group counts: weight by mean length alone.
            raw = dict(self._ema_mu)
        total = sum(raw.values())
        if total <= 0:
            return {}
        return {eid: v / total for eid, v in raw.items()}

    def _get_env_ids_ordered(self) -> list[str]:
        """env_id per agent, in the index order expected by set_capacity_weights."""
        if self.agent is not None and hasattr(self.agent, "env_id_for_agent"):
            ordered = [
                self.agent.env_id_for_agent(i) for i in range(len(self.agent.agents))
            ]
            if all(eid is not None for eid in ordered):
                return ordered
        # Best-effort fallback: first-seen order across the window.
        ordered = []
        seen = set()
        for batch in self._window:
            for record in batch:
                eid = record["env_id"]
                if eid not in seen:
                    seen.add(eid)
                    ordered.append(eid)
        return ordered
