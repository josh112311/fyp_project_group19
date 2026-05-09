import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fixed_confidence_experiments import (
    run_trials_fixed_confidence,
    summarize_fixed_confidence,
    print_fixed_confidence_summary,
    plot_fixed_confidence_tau_hist,
    make_result_prefix,
    save_fixed_confidence_outputs,
    save_records_csv,
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Experiment 1: Synthetic fixed-confidence sanity check
# ============================================================
def experiment_1_sanity_fixed(n_trials=20, seed=0):
    true_theta = np.array([0.50, 0.40, 0.30, 0.20, 0.10], dtype=float)
    sigma2 = np.ones_like(true_theta)

    k = 2
    delta = 0.10
    N_max = 50_000
    check_every = 50

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Fixed-Confidence Sanity Check")
    print("=" * 70)
    print(f"theta = {true_theta}")
    print(f"k = {k}, delta = {delta}, N_max = {N_max:,}, trials = {n_trials}")
    print(f"check_every (FWSP = Naive) = {check_every}")

    out = run_trials_fixed_confidence(
        true_theta=true_theta,
        sigma2=sigma2,
        k=k,
        delta=delta,
        N_max=N_max,
        n_trials=n_trials,
        burn_in=5,
        check_every_fwsp=check_every,
        check_every_naive=check_every,
        reward_dist="gaussian",
        use_active_shortlist_grads=True,
        base_seed=seed,
    )

    fwsp_summary = summarize_fixed_confidence(out["fwsp"])
    naive_summary = summarize_fixed_confidence(out["naive"])

    print_fixed_confidence_summary("FWSP", fwsp_summary, delta)
    print_fixed_confidence_summary("Naive", naive_summary, delta)

    prefix = make_result_prefix(
        RESULTS_DIR, "exp1_fc_sanity", "gaussian", k, delta, N_max
    )

    plot_fixed_confidence_tau_hist(
        out["fwsp"],
        title="FWSP tau distribution | stopped runs",
        save_path=prefix + "_fwsp_tau.png",
    )
    plot_fixed_confidence_tau_hist(
        out["naive"],
        title="Naive tau distribution | stopped runs",
        save_path=prefix + "_naive_tau.png",
    )

    save_fixed_confidence_outputs(
        results_dict=out,
        fwsp_summary=fwsp_summary,
        naive_summary=naive_summary,
        save_prefix=prefix,
        meta={
            "experiment": "exp1_fc_sanity",
            "reward_dist": "gaussian",
            "k": k,
            "delta": delta,
            "N_max": N_max,
            "n_trials": n_trials,
            "check_every": check_every,
            "theta": true_theta.tolist(),
        },
    )

    return out


# ============================================================
# Experiment 2: Delta sensitivity (fixed-confidence core study)
# ============================================================
def experiment_2_delta_sensitivity(n_trials=20, seed=0):
    true_theta = np.array([0.50, 0.45, 0.35, 0.20, 0.10], dtype=float)
    sigma2 = np.ones_like(true_theta)

    k = 2
    N_max = 80_000
    deltas = [0.20, 0.10, 0.05]
    check_every = 50

    fwsp_stop = []
    fwsp_tau = []
    naive_stop = []
    naive_tau = []

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Delta Sensitivity (Fixed-Confidence)")
    print("=" * 70)

    all_summaries = []

    for delta in deltas:
        print(f"\n--- delta = {delta} ---")

        out = run_trials_fixed_confidence(
            true_theta=true_theta,
            sigma2=sigma2,
            k=k,
            delta=delta,
            N_max=N_max,
            n_trials=n_trials,
            burn_in=5,
            check_every_fwsp=check_every,
            check_every_naive=check_every,
            reward_dist="gaussian",
            use_active_shortlist_grads=True,
            base_seed=seed,
        )

        fwsp_summary = summarize_fixed_confidence(out["fwsp"])
        naive_summary = summarize_fixed_confidence(out["naive"])

        print_fixed_confidence_summary("FWSP", fwsp_summary, delta)
        print_fixed_confidence_summary("Naive", naive_summary, delta)

        fwsp_stop.append(fwsp_summary["stop_fraction"])
        fwsp_tau.append(np.nan if fwsp_summary["mean_tau_given_stop"] is None else fwsp_summary["mean_tau_given_stop"])
        naive_stop.append(naive_summary["stop_fraction"])
        naive_tau.append(np.nan if naive_summary["mean_tau_given_stop"] is None else naive_summary["mean_tau_given_stop"])

        all_summaries.append({
            "delta": delta,
            "fwsp_stop_fraction": fwsp_summary["stop_fraction"],
            "fwsp_mean_tau_given_stop": fwsp_summary["mean_tau_given_stop"],
            "fwsp_success_given_stop": fwsp_summary["success_given_stop"],
            "naive_stop_fraction": naive_summary["stop_fraction"],
            "naive_mean_tau_given_stop": naive_summary["mean_tau_given_stop"],
            "naive_success_given_stop": naive_summary["success_given_stop"],
        })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(deltas, fwsp_stop, "o-", label="FWSP")
    ax.plot(deltas, naive_stop, "s--", label="Naive")
    ax.set_xlabel("delta")
    ax.set_ylabel("Stop fraction")
    ax.set_title("Stop Fraction vs delta")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(deltas, fwsp_tau, "o-", label="FWSP")
    ax.plot(deltas, naive_tau, "s--", label="Naive")
    ax.set_xlabel("delta")
    ax.set_ylabel("Mean tau | stopped")
    ax.set_title("Mean tau | stopped vs delta")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    prefix = make_result_prefix(
        RESULTS_DIR,
        "exp2_fc_delta_sweep",
        "gaussian",
        k,
        "multi",  
        N_max
    )
    fig.savefig(prefix + "_plot.png", dpi=150)
    plt.close(fig)

    save_records_csv(prefix + "_summary.csv", all_summaries)
    print(f"\nSaved: {prefix}_plot.png")


# ============================================================
# Experiment 3: Gap sensitivity (fixed-confidence core study)
# ============================================================
def experiment_3_gap_sensitivity(n_trials=20, seed=0):
    k = 2
    delta = 0.10
    N_max = 80_000
    gaps = [0.30, 0.20, 0.15, 0.10, 0.05]
    check_every = 50

    fwsp_tau = []
    naive_tau = []
    fwsp_stop = []
    naive_stop = []

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Gap Sensitivity (Fixed-Confidence)")
    print("=" * 70)

    all_summaries = []

    for gap in gaps:
        theta3 = 0.50 - gap
        true_theta = np.array([0.50, 0.45, theta3, 0.20, 0.10], dtype=float)
        sigma2 = np.ones_like(true_theta)

        print(f"\n--- gap = {gap:.2f}, theta = {true_theta} ---")

        out = run_trials_fixed_confidence(
            true_theta=true_theta,
            sigma2=sigma2,
            k=k,
            delta=delta,
            N_max=N_max,
            n_trials=n_trials,
            burn_in=5,
            check_every_fwsp=check_every,
            check_every_naive=check_every,
            reward_dist="gaussian",
            use_active_shortlist_grads=True,
            base_seed=seed,
        )

        fwsp_summary = summarize_fixed_confidence(out["fwsp"])
        naive_summary = summarize_fixed_confidence(out["naive"])

        print_fixed_confidence_summary("FWSP", fwsp_summary, delta)
        print_fixed_confidence_summary("Naive", naive_summary, delta)

        fwsp_stop.append(fwsp_summary["stop_fraction"])
        naive_stop.append(naive_summary["stop_fraction"])
        fwsp_tau.append(np.nan if fwsp_summary["mean_tau_given_stop"] is None else fwsp_summary["mean_tau_given_stop"])
        naive_tau.append(np.nan if naive_summary["mean_tau_given_stop"] is None else naive_summary["mean_tau_given_stop"])

        all_summaries.append({
            "gap": gap,
            "theta3": theta3,
            "fwsp_stop_fraction": fwsp_summary["stop_fraction"],
            "fwsp_mean_tau_given_stop": fwsp_summary["mean_tau_given_stop"],
            "fwsp_success_given_stop": fwsp_summary["success_given_stop"],
            "naive_stop_fraction": naive_summary["stop_fraction"],
            "naive_mean_tau_given_stop": naive_summary["mean_tau_given_stop"],
            "naive_success_given_stop": naive_summary["success_given_stop"],
        })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(gaps, fwsp_stop, "o-", label="FWSP")
    ax.plot(gaps, naive_stop, "s--", label="Naive")
    ax.set_xlabel("Gap")
    ax.set_ylabel("Stop fraction")
    ax.set_title("Stop Fraction vs Gap")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.legend()

    ax = axes[1]
    ax.plot(gaps, fwsp_tau, "o-", label="FWSP")
    ax.plot(gaps, naive_tau, "s--", label="Naive")
    ax.set_xlabel("Gap")
    ax.set_ylabel("Mean tau | stopped")
    ax.set_title("Mean tau | stopped vs Gap")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.legend()

    fig.tight_layout()
    prefix = make_result_prefix(
        RESULTS_DIR, "exp3_fc_gap", "gaussian", k, delta, N_max
    )
    fig.savefig(prefix + "_plot.png", dpi=150)
    plt.close(fig)

    save_records_csv(prefix + "_summary.csv", all_summaries)
    print(f"\nSaved: {prefix}_plot.png")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all", help="1, 2, 3, or all")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    exp = args.exp.lower()

    if exp in ("1", "all"):
        experiment_1_sanity_fixed(n_trials=args.n_trials, seed=args.seed)

    if exp in ("2", "all"):
        experiment_2_delta_sensitivity(n_trials=args.n_trials, seed=args.seed)

    if exp in ("3", "all"):
        experiment_3_gap_sensitivity(n_trials=args.n_trials, seed=args.seed)

    print("\nDone. Fixed-confidence results saved to results/")


if __name__ == "__main__":
    main()