import json
import math
import os
import random
from collections import deque
from typing import Dict, Optional


def _percentile(sorted_vals, p: float) -> float:
    """p-th percentile (p in [0,1]) of an already-sorted ascending list."""
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 1:
        return float(sorted_vals[-1])
    idx = int(math.ceil(p * (len(sorted_vals) - 1)))
    return float(sorted_vals[idx])


class DynamicMaxLen:
    def __init__(
        self,
        alpha_hi: float,
        alpha_min: Optional[float] = None,
        scale: float = 4096.0,
        ema_decay: float = 0.9,
        warmup_frac: float = 0.5,
        controller: str = "length",
        bandit_arms: Optional[list[float]] = None,
        bandit_token_cost: float = 0.05,
        bandit_exploration: float = 0.1,
        bandit_min_samples_per_arm: int = 16,
        reward_warmup_samples: int = 512,
        reward_percentile: float = 0.9,
        reward_cap_margin: float = 1.1,
        reward_min_cap: int = 256,
        reward_window: int = 4096,
        reward_refresh_frac: float = 0.05,
        reward_protect_delta: float = 0.03,
        reward_protect_min_samples: int = 32,
        reward_protect_percentile: float = 0.99,
        state_path: Optional[str] = None,
    ) -> None:
        self.alpha_hi = float(alpha_hi)
        self.alpha_min = float(alpha_hi if alpha_min is None else alpha_min)
        self.scale = max(1e-6, float(scale))
        self.beta = float(ema_decay)
        self.warmup_frac = float(warmup_frac)
        self.controller = controller
        # Tighter range than the original [0.75..3.0]: with a free large arm + saturated
        # reward the bandit always drifted to 3.0. These keep the cap near the typical length
        # so the arm choice actually trades truncation vs budget.
        # Range extended past 1.5 so long-generation envs (e.g. code_gen, whose utility was
        # still rising at the 1.5 cap) can find an interior optimum instead of railing at it.
        self.bandit_arms = list(bandit_arms or [0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0])
        self.bandit_token_cost = float(bandit_token_cost)
        self.bandit_exploration = max(0.0, float(bandit_exploration))
        self.bandit_min_samples_per_arm = max(0, int(bandit_min_samples_per_arm))
        # "reward" controller: observe UNCAPPED until reward_warmup_samples per env, then set the
        # cap to the length that retains reward_percentile of the reward MASS (not the max length).
        self.reward_warmup_samples = max(1, int(reward_warmup_samples))
        self.reward_percentile = min(1.0, max(0.0, float(reward_percentile)))
        self.reward_cap_margin = max(1.0, float(reward_cap_margin))
        self.reward_min_cap = max(1, int(reward_min_cap))
        self.reward_window = max(self.reward_warmup_samples, int(reward_window))
        self.reward_refresh_frac = min(1.0, max(0.0, float(reward_refresh_frac)))
        # Reward-protection guard: if UNCAPPED-phase reward (warmup/refresh/protected) exceeds
        # CAPPED-phase reward by more than this, the cap is suppressing reward (e.g. multi-turn
        # workplace_assistant collapsed 0.76->0.34) -> revert that env to ~uncapped. Without this,
        # the reward-mass percentile gets stuck in a local optimum: it sees moderate reward at the
        # short capped length and never discovers uncapping recovers more.
        self.reward_protect_delta = max(0.0, float(reward_protect_delta))
        self.reward_protect_min_samples = max(1, int(reward_protect_min_samples))
        # When the guard fires, don't go FULLY uncapped -- cap at this percentile of the env's TRUE
        # uncapped completion lengths, trimming only the runaway tail while preserving the
        # reward-bearing bulk (controls the "crazy long" rollouts without damaging reward).
        self.reward_protect_percentile = min(1.0, max(0.0, float(reward_protect_percentile)))
        self.state_path = state_path
        self._ema: Dict[str, float] = {}
        self._reward_ema: Dict[str, float] = {}
        self._bandit: Dict[str, Dict[float, dict[str, float]]] = {}
        self._arm_cursor: Dict[str, int] = {}
        self._window: Dict[str, deque] = {}        # env -> deque[(length, reward)] for "reward"
        self._reward_cap: Dict[str, int] = {}      # env -> last established cap (for metrics/log)
        # env -> deque of TRUE uncapped completion lengths (warmup/refresh only) for the P99
        # protected-cap (the generous tail-trim used when the guard fires).
        self._uncapped_lengths: Dict[str, deque] = {}
        # reward-protection: per-env reward EMA + count, split by whether the rollout was generated
        # capped vs uncapped (warmup/refresh/protected). Keyed [env] -> [mean, n].
        self._reward_capped: Dict[str, list] = {}
        self._reward_uncapped: Dict[str, list] = {}
        self._last_selection: Dict[str, dict] = {}

    def alpha(self, length: float) -> float:
        if self.alpha_min == self.alpha_hi:
            return self.alpha_hi
        return self.alpha_min + (self.alpha_hi - self.alpha_min) * math.exp(-length / self.scale)

    def select(self, env_id: str, max_seq: int) -> tuple[int, dict]:
        max_seq = int(max_seq)
        if self.controller == "reward":
            return self._select_reward(env_id, max_seq)
        length = self._ema.get(env_id)
        if length is None or length <= 0:
            cap = max(1, min(int(round(int(max_seq) * self.warmup_frac)), int(max_seq)))
            meta = self._meta(env_id, max_seq, cap, None, None, self.alpha(0.0), None)
            self._last_selection[env_id] = meta
            return cap, meta
        if self.controller == "bandit":
            arm = self._select_arm(env_id)
            cap = max(1, min(int(round(arm * length)), int(max_seq)))
            stats = self._arm_stats(env_id)[arm]
            meta = self._meta(env_id, max_seq, cap, arm, stats, arm, length)
            self._last_selection[env_id] = meta
            return cap, meta
        alpha = self.alpha(length)
        cap = max(1, min(int(round(alpha * length)), int(max_seq)))
        meta = self._meta(env_id, max_seq, cap, None, None, alpha, length)
        self._last_selection[env_id] = meta
        return cap, meta

    def cap(self, env_id: str, max_seq: int) -> int:
        return self.select(env_id, max_seq)[0]

    # ----- "reward" controller: warmup uncapped, then cap where the reward actually is -----
    def _select_reward(self, env_id: str, max_seq: int) -> tuple[int, dict]:
        win = self._window.get(env_id)
        n = 0 if win is None else len(win)
        # Warmup: generate UNCAPPED (cap == ceiling) so we observe the true length<->reward
        # relationship without truncating anything.
        if n < self.reward_warmup_samples:
            return self._finish_reward(env_id, max_seq, max_seq, n, None, "warmup")
        Lstar = self._reward_cap_length(env_id)
        p90_cap = max(self.reward_min_cap,
                      min(int(math.ceil(Lstar * self.reward_cap_margin)), max_seq))
        rf = self.reward_refresh_frac
        r = random.random()
        # Reward-protection. NOT protected: cap aggressively at the reward-mass percentile (P90),
        # with an occasional UNCAPPED refresh to keep the true-length estimate honest. PROTECTED
        # (capping is suppressing reward, e.g. multi-turn workplace_assistant): mostly use a
        # GENEROUS cap at P99 of the env's true uncapped lengths -- trims only the runaway tail,
        # preserves the reward-bearing bulk -- plus a small uncapped refresh (keeps P99 + reward
        # estimates fresh) and a small P90 re-test (so the guard can flip back if the policy drifts).
        if self._cap_hurts(env_id):
            if r < rf:
                phase, cap = "refresh", max_seq                  # pure uncapped: refresh true tail
            elif r < 2 * rf:
                phase, cap = "capped", p90_cap                   # re-test the aggressive cap
            else:
                L99 = self._protected_cap_length(env_id)
                if L99 == float("inf"):
                    phase, cap = "protected", max_seq            # not enough samples yet -> uncapped
                else:
                    phase = "protected"
                    cap = max(self.reward_min_cap, min(
                        int(math.ceil(max(Lstar, L99) * self.reward_cap_margin)), max_seq))
        else:
            if r < rf:
                phase, cap = "refresh", max_seq
            else:
                phase, cap = "capped", p90_cap
        if phase in ("capped", "protected"):
            self._reward_cap[env_id] = cap
        return self._finish_reward(env_id, max_seq, cap, n, Lstar, phase)

    def _protected_cap_length(self, env_id: str) -> float:
        """The generous tail-trim length used when the guard fires: the reward_protect_percentile
        (P99) of the env's TRUE uncapped completion lengths (from warmup/refresh, NOT the capped/
        protected generations whose lengths are truncated). float('inf') until enough samples."""
        lens = self._uncapped_lengths.get(env_id)
        if not lens or len(lens) < self.reward_protect_min_samples:
            return float("inf")
        return _percentile(sorted(lens), self.reward_protect_percentile)

    def _cap_hurts(self, env_id: str) -> bool:
        """True iff CAPPING this env earns meaningfully less reward than running it UNCAPPED
        (both estimated with >= reward_protect_min_samples). The guard against truncating
        reward-bearing completions (e.g. multi-turn envs)."""
        rc, ru = self._reward_capped.get(env_id), self._reward_uncapped.get(env_id)
        if rc is None or ru is None:
            return False
        if rc[1] < self.reward_protect_min_samples or ru[1] < self.reward_protect_min_samples:
            return False
        return ru[0] > rc[0] + self.reward_protect_delta

    def _finish_reward(self, env_id, max_seq, cap, n, Lstar, phase):
        meta = self._meta(env_id, max_seq, cap, None, None, 0.0, self._ema.get(env_id))
        meta["phase"] = phase
        meta["n_obs"] = int(n)
        meta["reward_cap_length"] = None if Lstar is None else float(Lstar)
        self._last_selection[env_id] = meta
        return cap, meta

    def _reward_cap_length(self, env_id: str) -> float:
        """Smallest length L such that completions of length <= L earned reward_percentile of the
        total reward mass in the window. If there is ~no reward signal yet, fall back to the 95th
        percentile of observed lengths (cut only the extreme tail, don't truncate the bulk)."""
        win = self._window.get(env_id)
        if not win:
            return float("inf")
        obs = sorted(((int(l), max(0.0, float(r))) for (l, r) in win), key=lambda t: t[0])
        total = sum(r for _, r in obs)
        if total <= 1e-9:
            return _percentile([l for l, _ in obs], 0.95)
        target = self.reward_percentile * total
        acc = 0.0
        for l, r in obs:
            acc += r
            if acc >= target:
                return float(l)
        return float(obs[-1][0])

    def update(self, env_id: str, generated_length: int, reward=None, metadata=None, ceiling=None) -> None:
        if generated_length is None or generated_length <= 0:
            return
        prev = self._ema.get(env_id)
        self._ema[env_id] = float(generated_length) if prev is None else self.beta * prev + (1 - self.beta) * float(generated_length)
        if reward is not None:
            prev_r = self._reward_ema.get(env_id)
            self._reward_ema[env_id] = float(reward) if prev_r is None else self.beta * prev_r + (1 - self.beta) * float(reward)
        if self.controller == "reward":
            win = self._window.get(env_id)
            if win is None or win.maxlen != self.reward_window:
                win = deque(win or (), maxlen=self.reward_window)
                self._window[env_id] = win
            win.append((int(generated_length), float(reward) if reward is not None else 0.0))
            # reward-protection bookkeeping: bucket this rollout's reward by whether it was
            # generated capped vs uncapped (warmup/refresh/protected), so _cap_hurts can tell if
            # the cap is suppressing reward. EMA-smoothed (decaying) -> tracks the current policy.
            phase = metadata.get("phase") if metadata is not None else None
            if reward is not None and phase is not None:
                uncapped = phase in ("warmup", "refresh", "protected")
                bucket = self._reward_uncapped if uncapped else self._reward_capped
                cur = bucket.get(env_id)
                if cur is None:
                    bucket[env_id] = [float(reward), 1.0]
                else:
                    cur[0] = self.beta * cur[0] + (1.0 - self.beta) * float(reward)
                    cur[1] += 1.0
            # TRUE uncapped lengths come ONLY from warmup/refresh (cap == ceiling); protected
            # generations are P99-capped so their lengths are truncated/biased -- exclude them.
            if phase in ("warmup", "refresh"):
                ul = self._uncapped_lengths.get(env_id)
                if ul is None or ul.maxlen != self.reward_window:
                    ul = deque(ul or (), maxlen=self.reward_window)
                    self._uncapped_lengths[env_id] = ul
                ul.append(int(generated_length))
        if self.controller == "bandit" and metadata and metadata.get("arm") is not None and reward is not None:
            arm = float(metadata["arm"])
            # Penalize the RESERVED generation budget (the cap = arm * length_ema), which
            # reserves KV cache and caps rollout parallelism, relative to the env's TYPICAL
            # generated length -- NOT the global ceiling. The old ceiling-normalized penalty
            # (cost * generated_len / max_seq) was ~1e-4 here and inert, so the bandit always
            # drifted to the largest arm (== no cap). cap/length_scale ~= arm, so this makes
            # the cost depend on the chosen arm: the bandit now prefers the smallest cap that
            # doesn't truncate -- a too-small cap costs reward (truncation), a too-large cap
            # costs budget.
            length_scale = self._ema.get(env_id) or float(generated_length) or 1.0
            cap = float(metadata.get("cap") or (arm * length_scale))
            utility = float(reward) - self.bandit_token_cost * (cap / max(1.0, length_scale))
            stats = self._arm_stats(env_id)[arm]
            # EMA (decaying) utility, NOT a cumulative sample mean: RL rewards are
            # NON-STATIONARY (the policy changes), and a 1/n sample mean lets early
            # exploration rewards dominate forever -> the bandit gets stuck on a stale arm
            # (observed: mcqa railing at the max cap with stale util 0.36 while reward_ema had
            # collapsed to 0.02). The EMA tracks recent utility so the arm choice adapts.
            if stats["n"] == 0.0:
                stats["mean"] = utility
            else:
                stats["mean"] = self.beta * stats["mean"] + (1.0 - self.beta) * utility
            stats["n"] = stats["n"] + 1.0

    def mean(self, env_id: str):
        return self._ema.get(env_id)

    def reward_mean(self, env_id: str):
        return self._reward_ema.get(env_id)

    def metrics(self, prefix="rl/dynamic_maxlen"):
        out = {}
        for env_id, selection in self._last_selection.items():
            env = env_id.replace(":", "_").replace("/", "_").replace(" ", "_")
            if env_id in self._ema:
                out[f"{prefix}/{env}/length_ema"] = float(self._ema[env_id])
            if env_id in self._reward_ema:
                out[f"{prefix}/{env}/reward_ema"] = float(self._reward_ema[env_id])
            out[f"{prefix}/{env}/cap"] = float(selection.get("cap", 0))
            out[f"{prefix}/{env}/alpha_or_arm"] = float(selection.get("alpha", 0))
            if selection.get("arm") is not None:
                out[f"{prefix}/{env}/arm"] = float(selection["arm"])
                out[f"{prefix}/{env}/arm_utility"] = float(selection.get("arm_utility", 0))
                out[f"{prefix}/{env}/arm_n"] = float(selection.get("arm_n", 0))
            if self.controller == "reward":
                out[f"{prefix}/{env}/n_obs"] = float(selection.get("n_obs", 0))
                # phase as a number so it's plottable: 0=warmup,1=capped,2=refresh,3=protected
                out[f"{prefix}/{env}/phase"] = {
                    "warmup": 0.0, "capped": 1.0, "refresh": 2.0, "protected": 3.0
                }.get(selection.get("phase"), -1.0)
                if selection.get("reward_cap_length") is not None:
                    out[f"{prefix}/{env}/reward_cap_length"] = float(selection["reward_cap_length"])
                # reward-protection observability: capped vs uncapped reward + the guard verdict
                rc, ru = self._reward_capped.get(env_id), self._reward_uncapped.get(env_id)
                if rc is not None:
                    out[f"{prefix}/{env}/reward_capped"] = float(rc[0])
                if ru is not None:
                    out[f"{prefix}/{env}/reward_uncapped"] = float(ru[0])
                out[f"{prefix}/{env}/cap_protected"] = 1.0 if self._cap_hurts(env_id) else 0.0
        return out

    def save_state(self):
        if not self.state_path:
            return
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        data = {
            "ema": self._ema,
            "reward_ema": self._reward_ema,
            "bandit": {env: {str(a): s for a, s in arms.items()} for env, arms in self._bandit.items()},
            "arm_cursor": self._arm_cursor,
            # persist the observation window so a chained restart resumes the cap estimate
            # (and stays past warmup) instead of re-observing uncapped from scratch.
            "window": {env: list(win) for env, win in self._window.items()},
            "reward_cap": self._reward_cap,
            "reward_capped": self._reward_capped,
            "reward_uncapped": self._reward_uncapped,
            "uncapped_lengths": {env: list(dq) for env, dq in self._uncapped_lengths.items()},
        }
        tmp = f"{self.state_path}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.state_path)

    def load_state(self):
        if not self.state_path or not os.path.exists(self.state_path):
            return
        with open(self.state_path) as f:
            data = json.load(f)
        self._ema = {k: float(v) for k, v in data.get("ema", {}).items()}
        self._reward_ema = {k: float(v) for k, v in data.get("reward_ema", {}).items()}
        self._bandit = {
            env: {float(a): {"n": float(s["n"]), "mean": float(s["mean"])} for a, s in arms.items()}
            for env, arms in data.get("bandit", {}).items()
        }
        self._arm_cursor = {k: int(v) for k, v in data.get("arm_cursor", {}).items()}
        self._window = {
            env: deque(((int(l), float(r)) for l, r in obs), maxlen=self.reward_window)
            for env, obs in data.get("window", {}).items()
        }
        self._reward_cap = {k: int(v) for k, v in data.get("reward_cap", {}).items()}
        self._reward_capped = {k: [float(v[0]), float(v[1])] for k, v in data.get("reward_capped", {}).items()}
        self._reward_uncapped = {k: [float(v[0]), float(v[1])] for k, v in data.get("reward_uncapped", {}).items()}
        self._uncapped_lengths = {
            env: deque((int(x) for x in xs), maxlen=self.reward_window)
            for env, xs in data.get("uncapped_lengths", {}).items()
        }

    def _meta(self, env_id, max_seq, cap, arm, stats, alpha, mean):
        return {
            "controller": self.controller,
            "arm": arm,
            "arm_n": 0 if stats is None else int(stats["n"]),
            "arm_utility": 0.0 if stats is None else float(stats["mean"]),
            "cap": int(cap),
            "ceiling": int(max_seq),
            "mean": mean,
            "alpha": float(alpha),
            "reward_mean": self._reward_ema.get(env_id),
        }

    def _arm_stats(self, env_id):
        if env_id not in self._bandit:
            self._bandit[env_id] = {float(a): {"n": 0.0, "mean": 0.0} for a in self.bandit_arms}
        return self._bandit[env_id]

    def _select_arm(self, env_id):
        stats = self._arm_stats(env_id)
        under = [float(a) for a in self.bandit_arms if stats[float(a)]["n"] < self.bandit_min_samples_per_arm]
        if under:
            cursor = self._arm_cursor.get(env_id, 0)
            arm = under[cursor % len(under)]
            self._arm_cursor[env_id] = cursor + 1
            return arm
        best_arm, best_score = float(self.bandit_arms[0]), -float("inf")
        for arm in self.bandit_arms:
            arm = float(arm)
            uncertainty = self.bandit_exploration / math.sqrt(stats[arm]["n"] + 1.0)
            score = random.gauss(stats[arm]["mean"], uncertainty) if uncertainty > 0 else stats[arm]["mean"]
            if score > best_score:
                best_arm, best_score = arm, score
        return best_arm


_INSTANCE: Optional[DynamicMaxLen] = None


def configure(alpha_hi, alpha_min=None, scale=4096.0, ema_decay=0.9, warmup_frac=0.5, controller="length",
              bandit_arms=None, bandit_token_cost=0.05, bandit_exploration=0.1, bandit_min_samples_per_arm=16,
              reward_warmup_samples=512, reward_percentile=0.9, reward_cap_margin=1.1, reward_min_cap=256,
              reward_window=4096, reward_refresh_frac=0.05, reward_protect_delta=0.03,
              reward_protect_min_samples=32, reward_protect_percentile=0.99, state_path=None):
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = DynamicMaxLen(alpha_hi, alpha_min, scale, ema_decay, warmup_frac, controller,
                                  bandit_arms, bandit_token_cost, bandit_exploration,
                                  bandit_min_samples_per_arm,
                                  reward_warmup_samples, reward_percentile, reward_cap_margin,
                                  reward_min_cap, reward_window, reward_refresh_frac,
                                  reward_protect_delta, reward_protect_min_samples,
                                  reward_protect_percentile, state_path)
        _INSTANCE.load_state()
    return _INSTANCE


def get_instance():
    return _INSTANCE


def configure_from_args(args):
    alpha = getattr(args, "rl_dynamic_maxlen_alpha", None)
    if alpha is None or alpha <= 0:
        return None
    state_path = getattr(args, "rl_dynamic_maxlen_state_path", None)
    if not state_path and getattr(args, "save", None):
        state_path = os.path.join(args.save, "dynamic_maxlen_state.json")
    return configure(
        alpha_hi=alpha,
        alpha_min=getattr(args, "rl_dynamic_maxlen_alpha_min", None),
        scale=getattr(args, "rl_dynamic_maxlen_alpha_scale", 4096.0),
        ema_decay=getattr(args, "rl_dynamic_maxlen_ema_decay", 0.9),
        warmup_frac=getattr(args, "rl_dynamic_maxlen_warmup_frac", 0.5),
        controller=getattr(args, "rl_dynamic_maxlen_controller", "length"),
        bandit_arms=_parse_arms(getattr(args, "rl_dynamic_maxlen_bandit_arms", None)),
        bandit_token_cost=getattr(args, "rl_dynamic_maxlen_bandit_token_cost", 0.0),
        bandit_exploration=getattr(args, "rl_dynamic_maxlen_bandit_exploration", 0.25),
        bandit_min_samples_per_arm=getattr(args, "rl_dynamic_maxlen_bandit_min_samples_per_arm", 1),
        reward_warmup_samples=getattr(args, "rl_dynamic_maxlen_reward_warmup_samples", 512),
        reward_percentile=getattr(args, "rl_dynamic_maxlen_reward_percentile", 0.9),
        reward_cap_margin=getattr(args, "rl_dynamic_maxlen_reward_cap_margin", 1.1),
        reward_min_cap=getattr(args, "rl_dynamic_maxlen_reward_min_cap", 256),
        reward_window=getattr(args, "rl_dynamic_maxlen_reward_window", 4096),
        reward_refresh_frac=getattr(args, "rl_dynamic_maxlen_reward_refresh_frac", 0.05),
        reward_protect_delta=getattr(args, "rl_dynamic_maxlen_reward_protect_delta", 0.03),
        reward_protect_min_samples=getattr(args, "rl_dynamic_maxlen_reward_protect_min_samples", 32),
        reward_protect_percentile=getattr(args, "rl_dynamic_maxlen_reward_protect_percentile", 0.99),
        state_path=state_path,
    )


def _parse_arms(arms):
    if arms is None or arms == "":
        return None
    return [float(x) for x in str(arms).split(",") if x.strip()]
