import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import csv
import json
from datetime import datetime

from learning_ver import (
    fwsp_shortlisting_learning,
    all_k_subsets,
    complement_indices,
    evidence_min_over_j,
    beta_threshold,
    argmax_lex,
)


# ============================================================
# 1. Naive fixed-confidence baseline
# ============================================================
def naive_fixed_confidence(
    true_theta: np.ndarray,
    sigma2: np.ndarray,
    k: int,
    delta: float,
    N_max: int,
    burn_in: int = 5,
    check_every: int = 10,
    reward_dist: str = "gaussian",
    seed: int = None,
):
    """
    Uniform / round-robin fixed-confidence baseline.

    IMPORTANT:
    - If stopping rule triggers: valid fixed-confidence stop
    - If N_max exhausted before stopping: timeout, no guarantee
    """
    rng = np.random.default_rng(seed)
    K = len(true_theta)

    if reward_dist == "gaussian":
        def sample(i):
            return float(rng.normal(true_theta[i], np.sqrt(sigma2[i])))
    elif reward_dist == "bernoulli":
        def sample(i):
            return float(rng.binomial(1, true_theta[i]))
    else:
        raise ValueError(f"Unknown reward_dist: {reward_dist!r}")

    all_S = all_k_subsets(K, k)
    Sc_map = {S: complement_indices(K, S) for S in all_S}

    N_counts = np.zeros(K, dtype=int)
    sum_rewards = np.zeros(K, dtype=float)
    theta_hat = np.zeros(K, dtype=float)

    # burn-in
    for i in range(K):
        for _ in range(burn_in):
            y = sample(i)
            N_counts[i] += 1
            sum_rewards[i] += y
            theta_hat[i] = sum_rewards[i] / N_counts[i]

    tau = None
    S_hat_Z = None
    stopped_by_rule = False
    timed_out = False

    n_round = 0
    while int(N_counts.sum()) < N_max:
        i = n_round % K

        y = sample(i)
        N_counts[i] += 1
        sum_rewards[i] += y
        theta_hat[i] = sum_rewards[i] / N_counts[i]

        n_round += 1

        if delta is not None and n_round % check_every == 0:
            Z_S = {
                S: evidence_min_over_j(theta_hat, sigma2, N_counts, S, Sc_map)
                for S in all_S
            }
            Z_best_S, Z_best_val = argmax_lex(Z_S)

            if Z_best_val >= beta_threshold(int(N_counts.sum()), delta):
                tau = int(N_counts.sum())
                S_hat_Z = Z_best_S
                stopped_by_rule = True
                break

    if not stopped_by_rule:
        timed_out = True

    status = "stopped" if stopped_by_rule else "timeout"

    return {
        "tau": tau,
        "status": status,
        "stopped_by_rule": stopped_by_rule,
        "timed_out": timed_out,
        "guarantee_valid": stopped_by_rule,
        "theta_hat": theta_hat,
        "N_counts": N_counts,
        "S_hat_Z": S_hat_Z,   # valid only if stopped_by_rule=True
    }


# ============================================================
# 2. Multi-trial runner for fixed-confidence experiments
# ============================================================
def run_trials_fixed_confidence(
    true_theta: np.ndarray,
    sigma2: np.ndarray,
    k: int,
    delta: float,
    N_max: int,
    n_trials: int,
    burn_in: int = 5,
    check_every_fwsp: int = 50,
    check_every_naive: int = 50,
    reward_dist: str = "gaussian",
    use_active_shortlist_grads: bool = True,
    base_seed: int = 0,
):
    """
    Runs FWSP and naive baseline under the SAME fixed-confidence protocol:
    - stop if evidence threshold reached
    - otherwise mark as timeout at N_max
    Adds trial-level progress bar and ETA.
    """
    import time
    from tqdm import tqdm

    best_arm = int(np.argmax(true_theta))

    fwsp_results = []
    naive_results = []

    start_all = time.time()

    trial_iter = tqdm(
        range(n_trials),
        desc="Fixed-confidence trials",
        unit="trial",
        dynamic_ncols=True,
    )

    for trial in trial_iter:
        trial_start = time.time()
        seed = base_seed + trial

        fwsp_out = fwsp_shortlisting_learning(
            sigma2=sigma2,
            k=k,
            N_rounds=N_max,
            true_theta=true_theta,
            delta=delta,
            use_active_shortlist_grads=use_active_shortlist_grads,
            burn_in=burn_in,
            seed=seed,
            verbose=False,
            check_every=check_every_fwsp,
            reward_dist=reward_dist,
            record_trajectory=False,
        )

        if fwsp_out["stopped_by_rule"]:
            fwsp_success = best_arm in fwsp_out["S_hat_Z"]
            fwsp_final_shortlist = fwsp_out["S_hat_Z"]
        else:
            fwsp_success = None
            fwsp_final_shortlist = None

        fwsp_results.append({
            "tau": fwsp_out["tau"],
            "status": fwsp_out["status"],
            "stopped_by_rule": fwsp_out["stopped_by_rule"],
            "timed_out": fwsp_out["timed_out"],
            "guarantee_valid": fwsp_out["guarantee_valid"],
            "success_when_stopped": fwsp_success,
            "final_shortlist_used": fwsp_final_shortlist,
            "shortlist_source": fwsp_out["shortlist_source"],
            "p": fwsp_out["p"],
            "S_hat_Z": fwsp_out["S_hat_Z"],
        })

        naive_out = naive_fixed_confidence(
            true_theta=true_theta,
            sigma2=sigma2,
            k=k,
            delta=delta,
            N_max=N_max,
            burn_in=burn_in,
            check_every=check_every_naive,
            reward_dist=reward_dist,
            seed=seed,
        )

        if naive_out["stopped_by_rule"]:
            naive_success = best_arm in naive_out["S_hat_Z"]
            naive_final_shortlist = naive_out["S_hat_Z"]
        else:
            naive_success = None
            naive_final_shortlist = None

        naive_results.append({
            "tau": naive_out["tau"],
            "status": naive_out["status"],
            "stopped_by_rule": naive_out["stopped_by_rule"],
            "timed_out": naive_out["timed_out"],
            "guarantee_valid": naive_out["guarantee_valid"],
            "success_when_stopped": naive_success,
            "final_shortlist_used": naive_final_shortlist,
            "S_hat_Z": naive_out["S_hat_Z"],
        })

        trial_elapsed = time.time() - trial_start
        done = trial + 1
        avg_per_trial = (time.time() - start_all) / done
        eta = avg_per_trial * (n_trials - done)

        trial_iter.set_postfix({
            "last_s": f"{trial_elapsed:.1f}",
            "avg_s": f"{avg_per_trial:.1f}",
            "eta_s": f"{eta:.1f}",
        })

    total_elapsed = time.time() - start_all
    if total_elapsed < 60:
        print(f"\nActual total time: {total_elapsed:.1f} seconds")
    else:
        print(f"\nActual total time: {total_elapsed/60:.1f} minutes")

    return {
        "fwsp": fwsp_results,
        "naive": naive_results,
    }

def _safe_float(x):
    return None if x is None else float(x)


def _safe_int(x):
    return None if x is None else int(x)


def make_result_prefix(
    root: str,
    exp_name: str,
    reward_dist: str,
    k: int,
    delta: float,
    n_max: int,
):
    """
    Standardized filename prefix.
    Example:
      results/fc_gap_gaussian_k2_d010_n80000
      results/obd_capped_bernoulli_k5_d010_n50000
    """
    d_str = str(delta).replace(".", "")
    return os.path.join(
        root,
        f"{exp_name}_{reward_dist}_k{k}_d{d_str}_n{n_max}"
    )


def summarize_status_table(results_list):
    """
    Trial-level status counts.
    """
    n_total = len(results_list)
    n_stopped = sum(int(r["stopped_by_rule"]) for r in results_list)
    n_timeout = sum(int(r["timed_out"]) for r in results_list)
    n_guarantee_valid = sum(int(r["guarantee_valid"]) for r in results_list)

    return {
        "n_total": n_total,
        "n_stopped_by_rule": n_stopped,
        "n_timeout": n_timeout,
        "n_guarantee_valid": n_guarantee_valid,
        "stop_fraction": n_stopped / n_total if n_total > 0 else 0.0,
        "timeout_fraction": n_timeout / n_total if n_total > 0 else 0.0,
    }


def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_records_csv(path: str, records: list):
    if len(records) == 0:
        return
    keys = sorted(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)

# ============================================================
# 3. Summary for fixed-confidence experiments
# ============================================================
def summarize_fixed_confidence(results_list):
    """
    Summarize ONLY proper fixed-confidence outputs.
    Timeout runs are counted in stop_fraction / timeout_fraction,
    but do not contribute to tau|stopped or success|stopped.
    """
    n_total = len(results_list)
    stopped_runs = [r for r in results_list if r["stopped_by_rule"]]
    timeout_runs = [r for r in results_list if r["timed_out"]]

    stop_fraction = len(stopped_runs) / n_total if n_total > 0 else 0.0
    timeout_fraction = len(timeout_runs) / n_total if n_total > 0 else 0.0

    taus = [r["tau"] for r in stopped_runs if r["tau"] is not None]
    successes = [
        r["success_when_stopped"]
        for r in stopped_runs
        if r["success_when_stopped"] is not None
    ]

    mean_tau = float(np.mean(taus)) if len(taus) > 0 else None
    median_tau = float(np.median(taus)) if len(taus) > 0 else None
    min_tau = float(np.min(taus)) if len(taus) > 0 else None
    max_tau = float(np.max(taus)) if len(taus) > 0 else None
    success_given_stop = float(np.mean(successes)) if len(successes) > 0 else None

    return {
        "n_total": n_total,
        "n_stopped": len(stopped_runs),
        "n_timeout": len(timeout_runs),
        "stop_fraction": stop_fraction,
        "timeout_fraction": timeout_fraction,
        "mean_tau_given_stop": mean_tau,
        "median_tau_given_stop": median_tau,
        "min_tau_given_stop": min_tau,
        "max_tau_given_stop": max_tau,
        "success_given_stop": success_given_stop,
    }

def print_fixed_confidence_summary(name: str, summary: dict, delta: float):
    print("=" * 64)
    print(f"{name} — Fixed-Confidence Summary")
    print("=" * 64)
    print(f"Total trials:                  {summary['n_total']}")
    print(f"Stopped by rule:               {summary['n_stopped']}")
    print(f"Timed out:                     {summary['n_timeout']}")
    print(f"Stop fraction:                 {summary['stop_fraction']:.3f}")
    print(f"Timeout fraction:              {summary['timeout_fraction']:.3f}")

    if summary["mean_tau_given_stop"] is None:
        print("Mean tau | stopped:            None")
        print("Median tau | stopped:          None")
        print("Min tau | stopped:             None")
        print("Max tau | stopped:             None")
    else:
        print(f"Mean tau | stopped:            {summary['mean_tau_given_stop']:.1f}")
        print(f"Median tau | stopped:          {summary['median_tau_given_stop']:.1f}")
        print(f"Min tau | stopped:             {summary['min_tau_given_stop']:.1f}")
        print(f"Max tau | stopped:             {summary['max_tau_given_stop']:.1f}")

    if summary["success_given_stop"] is None:
        print("Success rate | stopped:        None")
    else:
        print(
            f"Success rate | stopped:        {summary['success_given_stop']:.3f} "
            f"(empirical target ~ {1-delta:.2f})"
        )

    print("Note: only 'stopped by rule' runs carry fixed-confidence semantics.")

def save_fixed_confidence_outputs(
    results_dict: dict,
    fwsp_summary: dict,
    naive_summary: dict,
    save_prefix: str,
    meta: dict,
):
    """
    Save machine-readable outputs for report writing.
    """
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

    # 1) save summaries JSON
    summary_obj = {
        "meta": meta,
        "fwsp_summary": fwsp_summary,
        "naive_summary": naive_summary,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(save_prefix + "_summary.json", summary_obj)

    # 2) save FWSP trial table
    fwsp_rows = []
    for i, r in enumerate(results_dict["fwsp"]):
        fwsp_rows.append({
            "trial": i,
            "status": r["status"],
            "stopped_by_rule": r["stopped_by_rule"],
            "timed_out": r["timed_out"],
            "guarantee_valid": r["guarantee_valid"],
            "tau": _safe_int(r["tau"]),
            "success_when_stopped": r["success_when_stopped"],
            "shortlist_source": r["shortlist_source"],
            "S_hat_Z": None if r["S_hat_Z"] is None else str(tuple(int(x) for x in r["S_hat_Z"])),
        })
    save_records_csv(save_prefix + "_fwsp_trials.csv", fwsp_rows)

    # 3) save naive trial table
    naive_rows = []
    for i, r in enumerate(results_dict["naive"]):
        naive_rows.append({
            "trial": i,
            "status": r["status"],
            "stopped_by_rule": r["stopped_by_rule"],
            "timed_out": r["timed_out"],
            "guarantee_valid": r["guarantee_valid"],
            "tau": _safe_int(r["tau"]),
            "success_when_stopped": r["success_when_stopped"],
            "S_hat_Z": None if r["S_hat_Z"] is None else str(tuple(int(x) for x in r["S_hat_Z"])),
        })
    save_records_csv(save_prefix + "_naive_trials.csv", naive_rows)

# ============================================================
# 5. Plotting for fixed-confidence
# ============================================================
def plot_fixed_confidence_tau_hist(results_list, title="Tau distribution | stopped runs", save_path=None):
    taus = [r["tau"] for r in results_list if r["stopped_by_rule"] and r["tau"] is not None]

    plt.figure(figsize=(7, 4))
    if len(taus) > 0:
        plt.hist(taus, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Stopping time τ")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


# ============================================================
# 6. Example main
# ============================================================
if __name__ == "__main__":
    # Example: small synthetic Gaussian fixed-confidence validation
    true_theta = np.array([0.50, 0.45, 0.35, 0.20, 0.10], dtype=float)
    sigma2 = np.ones_like(true_theta)

    k = 2
    delta = 0.1
    N_max = 50000
    n_trials = 20

    out = run_trials_fixed_confidence(
        true_theta=true_theta,
        sigma2=sigma2,
        k=k,
        delta=delta,
        N_max=N_max,
        n_trials=n_trials,
        burn_in=5,
        check_every_fwsp=50,
        check_every_naive=50,
        reward_dist="gaussian",
        base_seed=0,
    )

    fwsp_summary = summarize_fixed_confidence(out["fwsp"])
    naive_summary = summarize_fixed_confidence(out["naive"])

    print_fixed_confidence_summary("FWSP", fwsp_summary, delta)
    print_fixed_confidence_summary("Naive", naive_summary, delta)

    plot_fixed_confidence_tau_hist(
        out["fwsp"],
        title="FWSP tau distribution | stopped runs",
        save_path="fwsp_fixed_confidence_tau.png",
    )

    plot_fixed_confidence_tau_hist(
        out["naive"],
        title="Naive tau distribution | stopped runs",
        save_path="naive_fixed_confidence_tau.png",
    )