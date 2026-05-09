from __future__ import annotations

"""
parameter_framework.py

Parameter-selection framework for FWSP shortlisting.

Goal
----
Given a shortlist problem instance (synthetic or OBD-derived), this script:
1. Diagnoses instance difficulty from estimated arm means.
2. Produces a rule-based recommendation.
3. Optionally performs a small empirical search over candidate parameters.
4. Scores candidate settings by balancing:
      - best-arm inclusion probability,
      - stop fraction,
      - sample usage.

This is designed to sit *on top of* the existing FWSP implementation rather than
modifying the algorithm itself.

Typical usage
-------------
# Rule-based recommendation only
python parameter_framework.py --npz obd_arms.npz --mode rule

# Rule + empirical search (recommended)
python parameter_framework.py --npz obd_arms.npz --mode search \
    --k-grid 3 --kkeep-grid 6,8,10 --delta-grid 0.05,0.1,0.2 \
    --budget-grid 50000,100000,200000 --burnin-grid 20,50 --n-trials 10

Notes
-----
* For Bernoulli / OBD settings, sigma^2 is recomputed as theta*(1-theta)
  for consistency with the reward model.
* The script is robust to either version of fwsp_shortlisting_learning:
  - older version returning only tau / S_hat_Z / S_hat_p
  - newer fixed-confidence-friendly version returning stopped_by_rule etc.
"""

import argparse
import itertools
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from learning_ver import fwsp_shortlisting_learning


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass
class InstanceStats:
    K_keep: int
    k: int
    best_second_gap: float
    relevant_gap_best_to_outside: float
    boundary_gap_k_kplus1: Optional[float]
    avg_mean: float
    avg_var: float
    min_mean: float
    max_mean: float
    difficulty_score: float
    regime: str


@dataclass
class RuleRecommendation:
    K_keep: int
    k: int
    delta: float
    N_max: int
    burn_in: int
    reward_dist: str
    rationale: List[str]


@dataclass
class ConfigSummary:
    K_keep: int
    k: int
    delta: float
    N_max: int
    burn_in: int
    reward_dist: str
    n_trials: int
    success_rate: float
    stop_fraction: float
    mean_pulls_used: float
    median_pulls_used: float
    mean_tau_among_stopped: Optional[float]
    success_among_stopped: Optional[float]
    score: float


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)



def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]



def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]



def overlap_rate_shortlist(S_hat: Sequence[int], true_theta: np.ndarray, k: int) -> float:
    true_topk = set(np.argsort(-true_theta)[:k])
    return len(set(S_hat) & true_topk) / float(k)



def _safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------
# Instance loading / diagnosis
# ---------------------------------------------------------------------


def load_instance_from_npz(npz_path: str, K_keep: int, reward_dist: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    d = np.load(npz_path)
    true_theta = d["true_theta"][:K_keep].astype(float).copy()

    if reward_dist == "bernoulli":
        sigma2 = true_theta * (1.0 - true_theta)
    else:
        if "sigma2" in d:
            sigma2 = d["sigma2"][:K_keep].astype(float).copy()
        else:
            sigma2 = np.ones_like(true_theta)

    extra = {
        "item_ids": d["item_ids"][:K_keep].copy() if "item_ids" in d else np.arange(K_keep),
        "n_impr": d["n_impr"][:K_keep].copy() if "n_impr" in d else np.zeros(K_keep, dtype=int),
    }
    return true_theta, sigma2, extra



def compute_instance_stats(theta: np.ndarray, sigma2: np.ndarray, k: int) -> InstanceStats:
    theta = np.asarray(theta, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    theta_sorted = np.sort(theta)[::-1]
    K_keep = len(theta_sorted)

    best_second_gap = float(theta_sorted[0] - theta_sorted[1]) if K_keep >= 2 else 0.0
    relevant_gap_best_to_outside = float(theta_sorted[0] - theta_sorted[k]) if K_keep > k else best_second_gap
    boundary_gap = float(theta_sorted[k - 1] - theta_sorted[k]) if K_keep > k else None

    avg_mean = float(np.mean(theta_sorted))
    avg_var = float(np.mean(sigma2))
    min_mean = float(np.min(theta_sorted))
    max_mean = float(np.max(theta_sorted))

    effective_gap = boundary_gap if boundary_gap is not None else max(best_second_gap, 1e-12)
    difficulty_score = float((math.log(K_keep + 1.0) * max(avg_var, 1e-12)) / max(effective_gap, 1e-12))

    if effective_gap <= 5e-4 or difficulty_score >= 120:
        regime = "hard"
    elif effective_gap <= 15e-4 or difficulty_score >= 40:
        regime = "moderate"
    else:
        regime = "easy"

    return InstanceStats(
        K_keep=K_keep,
        k=k,
        best_second_gap=best_second_gap,
        relevant_gap_best_to_outside=relevant_gap_best_to_outside,
        boundary_gap_k_kplus1=boundary_gap,
        avg_mean=avg_mean,
        avg_var=avg_var,
        min_mean=min_mean,
        max_mean=max_mean,
        difficulty_score=difficulty_score,
        regime=regime,
    )


# ---------------------------------------------------------------------
# Rule-based recommendation
# ---------------------------------------------------------------------


def recommend_by_rules(stats: InstanceStats, reward_dist: str) -> RuleRecommendation:
    rationale: List[str] = []

    if stats.regime == "easy":
        delta = 0.05
        N_max = 50_000
        burn_in = 5 if reward_dist == "gaussian" else 20
        rationale.append("Instance classified as easy: use stricter confidence and moderate budget.")
    elif stats.regime == "moderate":
        delta = 0.10
        N_max = 100_000
        burn_in = 5 if reward_dist == "gaussian" else 50
        rationale.append("Instance classified as moderate: use balanced confidence and larger budget.")
    else:
        delta = 0.10
        N_max = 200_000 if reward_dist == "bernoulli" else 100_000
        burn_in = 5 if reward_dist == "gaussian" else 50
        rationale.append("Instance classified as hard: prefer moderate confidence and larger budget.")
        rationale.append("For hard instances, capped-budget evaluation may be more practical than strict stopping.")

    if reward_dist == "bernoulli":
        rationale.append("Bernoulli rewards detected: use sigma^2 = theta*(1-theta) for consistency.")

    return RuleRecommendation(
        K_keep=stats.K_keep,
        k=stats.k,
        delta=delta,
        N_max=N_max,
        burn_in=burn_in,
        reward_dist=reward_dist,
        rationale=rationale,
    )


# ---------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------


def _extract_trial_outcome(out: Dict[str, Any], true_theta: np.ndarray, k: int) -> Dict[str, Any]:
    best_arm = int(np.argmax(true_theta))

    # Newer version
    if "stopped_by_rule" in out:
        stopped = bool(out["stopped_by_rule"])
        pulls_used = int(out["N_counts"].sum())
        shortlist = out["S_hat_Z"] if stopped else out["S_hat_p"]
        tau = out["tau"]
    else:
        # Older version
        stopped = out.get("tau") is not None and out.get("S_hat_Z") is not None
        pulls_used = int(out["N_counts"].sum())
        shortlist = out["S_hat_Z"] if stopped else out["S_hat_p"]
        tau = out.get("tau")

    success = best_arm in shortlist
    overlap = overlap_rate_shortlist(shortlist, true_theta, k)

    return {
        "stopped": stopped,
        "pulls_used": pulls_used,
        "tau": tau,
        "success": bool(success),
        "overlap": float(overlap),
        "shortlist": tuple(int(x) for x in shortlist),
    }



def evaluate_configuration(
    true_theta: np.ndarray,
    sigma2: np.ndarray,
    k: int,
    delta: float,
    N_max: int,
    burn_in: int,
    reward_dist: str,
    n_trials: int,
    base_seed: int,
    check_every: int,
    use_active_shortlist_grads: bool,
) -> ConfigSummary:
    records: List[Dict[str, Any]] = []

    for t in range(n_trials):
        seed = base_seed + t
        out = fwsp_shortlisting_learning(
            sigma2=sigma2,
            k=k,
            N_rounds=N_max,
            true_theta=true_theta,
            delta=delta,
            use_active_shortlist_grads=use_active_shortlist_grads,
            burn_in=burn_in,
            seed=seed,
            verbose=False,
            check_every=check_every,
            reward_dist=reward_dist,
            record_trajectory=False,
        )
        records.append(_extract_trial_outcome(out, true_theta, k))

    success_rate = float(np.mean([r["success"] for r in records]))
    stop_fraction = float(np.mean([r["stopped"] for r in records]))
    pulls_used = np.array([r["pulls_used"] for r in records], dtype=float)
    taus = [r["tau"] for r in records if r["stopped"] and r["tau"] is not None]
    success_when_stopped = [r["success"] for r in records if r["stopped"]]

    # Composite score: prioritize success, then stop fraction, then fewer pulls.
    # This is intentionally transparent rather than overly tuned.
    normalized_cost = float(np.mean(pulls_used) / max(N_max, 1))
    score = (
        1000.0 * success_rate
        + 200.0 * stop_fraction
        - 100.0 * normalized_cost
    )

    return ConfigSummary(
        K_keep=len(true_theta),
        k=k,
        delta=delta,
        N_max=N_max,
        burn_in=burn_in,
        reward_dist=reward_dist,
        n_trials=n_trials,
        success_rate=success_rate,
        stop_fraction=stop_fraction,
        mean_pulls_used=float(np.mean(pulls_used)),
        median_pulls_used=float(np.median(pulls_used)),
        mean_tau_among_stopped=_safe_mean(taus),
        success_among_stopped=_safe_mean(success_when_stopped),
        score=float(score),
    )


# ---------------------------------------------------------------------
# Search framework
# ---------------------------------------------------------------------


def run_empirical_search(
    npz_path: str,
    reward_dist: str,
    k_grid: Sequence[int],
    kkeep_grid: Sequence[int],
    delta_grid: Sequence[float],
    budget_grid: Sequence[int],
    burnin_grid: Sequence[int],
    n_trials: int,
    base_seed: int,
    check_every: int,
    use_active_shortlist_grads: bool,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    for K_keep, k in itertools.product(kkeep_grid, k_grid):
        if k >= K_keep:
            continue

        true_theta, sigma2, extra = load_instance_from_npz(npz_path, K_keep=K_keep, reward_dist=reward_dist)
        stats = compute_instance_stats(true_theta, sigma2, k)
        rule_rec = recommend_by_rules(stats, reward_dist)

        for delta, N_max, burn_in in itertools.product(delta_grid, budget_grid, burnin_grid):
            summary = evaluate_configuration(
                true_theta=true_theta,
                sigma2=sigma2,
                k=k,
                delta=delta,
                N_max=N_max,
                burn_in=burn_in,
                reward_dist=reward_dist,
                n_trials=n_trials,
                base_seed=base_seed,
                check_every=check_every,
                use_active_shortlist_grads=use_active_shortlist_grads,
            )

            rec = {
                "instance_stats": asdict(stats),
                "rule_recommendation": asdict(rule_rec),
                "config_summary": asdict(summary),
                "top_item_ids": [int(x) for x in extra["item_ids"].tolist()],
                "top_true_theta": [float(x) for x in true_theta.tolist()],
            }
            results.append(rec)
            print(
                f"[search] K_keep={K_keep:>2}, k={k}, delta={delta:.3f}, N_max={N_max:>6}, "
                f"burn_in={burn_in:>3} | success={summary.success_rate:.3f}, "
                f"stop={summary.stop_fraction:.3f}, mean_pulls={summary.mean_pulls_used:,.0f}, "
                f"score={summary.score:.2f}"
            )

    # Ranking rule: highest score first, then higher success, then lower mean pulls.
    results.sort(
        key=lambda r: (
            -r["config_summary"]["score"],
            -r["config_summary"]["success_rate"],
            r["config_summary"]["mean_pulls_used"],
        )
    )

    best = results[0] if results else None
    return {"all_results": results, "best_result": best}


# ---------------------------------------------------------------------
# Presentation / reporting
# ---------------------------------------------------------------------


def print_instance_report(stats: InstanceStats, rule_rec: RuleRecommendation) -> None:
    print("=" * 72)
    print("Instance Diagnosis")
    print("=" * 72)
    print(f"K_keep                 : {stats.K_keep}")
    print(f"k                      : {stats.k}")
    print(f"best-second gap        : {stats.best_second_gap:.6g}")
    print(f"best-outside gap       : {stats.relevant_gap_best_to_outside:.6g}")
    print(f"boundary gap k,k+1     : {stats.boundary_gap_k_kplus1 if stats.boundary_gap_k_kplus1 is not None else 'N/A'}")
    print(f"avg mean               : {stats.avg_mean:.6g}")
    print(f"avg variance           : {stats.avg_var:.6g}")
    print(f"difficulty score       : {stats.difficulty_score:.4f}")
    print(f"regime                 : {stats.regime}")
    print()
    print("Rule-based recommendation")
    print("-" * 72)
    print(f"delta                  : {rule_rec.delta}")
    print(f"N_max                  : {rule_rec.N_max}")
    print(f"burn_in                : {rule_rec.burn_in}")
    print(f"reward_dist            : {rule_rec.reward_dist}")
    for i, line in enumerate(rule_rec.rationale, start=1):
        print(f"  {i}. {line}")
    print()



def print_best_result(best: Dict[str, Any]) -> None:
    if not best:
        print("No search result available.")
        return
    cfg = best["config_summary"]
    print("=" * 72)
    print("Best Empirical Configuration")
    print("=" * 72)
    print(f"K_keep                 : {cfg['K_keep']}")
    print(f"k                      : {cfg['k']}")
    print(f"delta                  : {cfg['delta']}")
    print(f"N_max                  : {cfg['N_max']}")
    print(f"burn_in                : {cfg['burn_in']}")
    print(f"success_rate           : {cfg['success_rate']:.3f}")
    print(f"stop_fraction          : {cfg['stop_fraction']:.3f}")
    print(f"mean_pulls_used        : {cfg['mean_pulls_used']:.1f}")
    print(f"median_pulls_used      : {cfg['median_pulls_used']:.1f}")
    print(f"mean_tau_among_stopped : {cfg['mean_tau_among_stopped']}")
    print(f"success_among_stopped  : {cfg['success_among_stopped']}")
    print(f"score                  : {cfg['score']:.2f}")
    print()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter-selection framework for FWSP shortlisting")
    parser.add_argument("--npz", default="obd_arms.npz", help="Path to NPZ instance file")
    parser.add_argument("--mode", choices=["rule", "search"], default="search")
    parser.add_argument("--reward-dist", choices=["bernoulli", "gaussian"], default="bernoulli")

    parser.add_argument("--K-keep", type=int, default=8, help="K_keep used in rule mode")
    parser.add_argument("--k", type=int, default=3, help="shortlist size used in rule mode")

    parser.add_argument("--kkeep-grid", default="6,8,10", help="comma-separated K_keep grid")
    parser.add_argument("--k-grid", default="3", help="comma-separated k grid")
    parser.add_argument("--delta-grid", default="0.05,0.1,0.2", help="comma-separated delta grid")
    parser.add_argument("--budget-grid", default="50000,100000,200000", help="comma-separated budget grid")
    parser.add_argument("--burnin-grid", default="20,50", help="comma-separated burn-in grid")

    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-every", type=int, default=50)
    parser.add_argument("--full-gradient", action="store_true", help="use full rho-mu gradient instead of active-shortlist gradient")
    args = parser.parse_args()

    if args.mode == "rule":
        true_theta, sigma2, extra = load_instance_from_npz(args.npz, K_keep=args.K_keep, reward_dist=args.reward_dist)
        stats = compute_instance_stats(true_theta, sigma2, args.k)
        rec = recommend_by_rules(stats, args.reward_dist)
        print_instance_report(stats, rec)

        out = {
            "mode": "rule",
            "instance_stats": asdict(stats),
            "rule_recommendation": asdict(rec),
            "top_item_ids": [int(x) for x in extra["item_ids"].tolist()],
            "top_true_theta": [float(x) for x in true_theta.tolist()],
        }
        save_path = os.path.join(RESULTS_DIR, f"parameter_framework_rule_kkeep{args.K_keep}_k{args.k}.json")
        save_json(save_path, out)
        print(f"Saved: {save_path}")
        return

    search_out = run_empirical_search(
        npz_path=args.npz,
        reward_dist=args.reward_dist,
        k_grid=parse_int_list(args.k_grid),
        kkeep_grid=parse_int_list(args.kkeep_grid),
        delta_grid=parse_float_list(args.delta_grid),
        budget_grid=parse_int_list(args.budget_grid),
        burnin_grid=parse_int_list(args.burnin_grid),
        n_trials=args.n_trials,
        base_seed=args.seed,
        check_every=args.check_every,
        use_active_shortlist_grads=not args.full_gradient,
    )

    best = search_out["best_result"]
    if best:
        print_best_result(best)

    save_path = os.path.join(RESULTS_DIR, "parameter_framework_search_results.json")
    save_json(save_path, search_out)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
