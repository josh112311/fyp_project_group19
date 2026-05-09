"""
Variance validation experiment.

This script compares:
- correct Bernoulli variance sigma^2 = theta * (1 - theta)
- incorrect default variance sigma^2 = 1

Its purpose is methodological validation:
to show that the variance specification strongly affects whether the fixed-confidence stopping rule can trigger.
"""

import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from learning_ver import fwsp_shortlisting_learning

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def main():
    true_theta = np.array([0.050, 0.042, 0.035, 0.020, 0.015, 0.010], dtype=float)
    k = 3
    delta = 0.1
    N_max = 50_000
    burn_in = 20
    best_arm = 0

    sigma2_correct = true_theta * (1.0 - true_theta)
    sigma2_wrong = np.ones_like(true_theta)

    print("=" * 70)
    print("OBD Variance Debug Experiment")
    print("=" * 70)
    print("Compare correct Bernoulli variance vs wrong sigma^2 = 1")
    print()

    print(f"{'Arm':>4}  {'CTR':>8}  {'sigma2_correct':>14}  {'sigma2_wrong':>12}")
    for i in range(len(true_theta)):
        print(f"{i+1:>4}  {true_theta[i]:>7.3%}  {sigma2_correct[i]:>14.6f}  {sigma2_wrong[i]:>12.1f}")

    print("\n" + "-" * 70)
    print("Single run with correct sigma2")
    print("-" * 70)

    t0 = time.time()
    out_correct = fwsp_shortlisting_learning(
        sigma2=sigma2_correct,
        k=k,
        N_rounds=N_max,
        true_theta=true_theta,
        delta=delta,
        use_active_shortlist_grads=True,
        burn_in=burn_in,
        seed=42,
        reward_dist="bernoulli",
        verbose=False,
        check_every=50,
    )
    t1 = time.time()

    print(f"Stopped by rule? {out_correct['stopped_by_rule']}")
    print(f"tau = {out_correct['tau']}")
    print(f"time = {t1 - t0:.1f}s")
    if out_correct["stopped_by_rule"]:
        print(f"Shortlist = {tuple(i+1 for i in out_correct['S_hat_Z'])}")
    else:
        print(f"Fallback shortlist = {tuple(i+1 for i in out_correct['S_hat_p'])}")

    print("\n" + "-" * 70)
    print("Single run with wrong sigma2 = 1")
    print("-" * 70)

    t0 = time.time()
    out_wrong = fwsp_shortlisting_learning(
        sigma2=sigma2_wrong,
        k=k,
        N_rounds=5_000,
        true_theta=true_theta,
        delta=delta,
        use_active_shortlist_grads=True,
        burn_in=burn_in,
        seed=42,
        reward_dist="bernoulli",
        verbose=False,
        check_every=50,
    )
    t1 = time.time()

    print(f"Stopped by rule? {out_wrong['stopped_by_rule']}")
    print(f"tau = {out_wrong['tau']}")
    print(f"time = {t1 - t0:.1f}s")
    print(f"n_used = {int(out_wrong['N_counts'].sum()):,}")

    # trajectory comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Correct vs Wrong Bernoulli Variance", fontsize=14, fontweight="bold")

    if out_correct["t_traj"].size > 0:
        ax = axes[0, 0]
        for i in range(len(true_theta)):
            ax.plot(out_correct["t_traj"], out_correct["p_traj"][:, i], label=f"Arm {i+1}")
        ax.set_xscale("log")
        ax.set_title("Correct sigma2: Allocation")
        ax.set_xlabel("Total pulls")
        ax.set_ylabel("Allocation")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=7)

        ax = axes[0, 1]
        for i in range(len(true_theta)):
            ax.plot(out_correct["t_traj"], out_correct["theta_traj"][:, i], label=f"Arm {i+1}")
            ax.axhline(true_theta[i], linestyle="--", linewidth=1, color=f"C{i%10}", alpha=0.4)
        ax.set_xscale("log")
        ax.set_title("Correct sigma2: Mean estimates")
        ax.set_xlabel("Total pulls")
        ax.set_ylabel(r"$\hat{\theta}_i$")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=7)

    if out_wrong["t_traj"].size > 0:
        ax = axes[1, 0]
        for i in range(len(true_theta)):
            ax.plot(out_wrong["t_traj"], out_wrong["p_traj"][:, i], label=f"Arm {i+1}")
        ax.set_xscale("log")
        ax.set_title("Wrong sigma2=1: Allocation")
        ax.set_xlabel("Total pulls")
        ax.set_ylabel("Allocation")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=7)

        ax = axes[1, 1]
        for i in range(len(true_theta)):
            ax.plot(out_wrong["t_traj"], out_wrong["theta_traj"][:, i], label=f"Arm {i+1}")
            ax.axhline(true_theta[i], linestyle="--", linewidth=1, color=f"C{i%10}", alpha=0.4)
        ax.set_xscale("log")
        ax.set_title("Wrong sigma2=1: Mean estimates")
        ax.set_xlabel("Total pulls")
        ax.set_ylabel(r"$\hat{\theta}_i$")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=7)

    prefix = os.path.join(RESULTS_DIR, "obd_variance_debug_k3_d01_n50000")

    save_json(prefix + "_summary.json", {
        "experiment": "variance_debug_single_run",
        "true_theta": [float(x) for x in true_theta.tolist()],
        "sigma2_correct": [float(x) for x in sigma2_correct.tolist()],
        "sigma2_wrong": [float(x) for x in sigma2_wrong.tolist()],

        "correct_run": {
            "stopped_by_rule": bool(out_correct["stopped_by_rule"]),
            "timed_out": bool(out_correct["timed_out"]),
            "tau": None if out_correct["tau"] is None else int(out_correct["tau"]),
            "shortlist_source": out_correct["shortlist_source"],
            "guarantee_valid": bool(out_correct["guarantee_valid"]),
            "S_hat_Z": None if out_correct["S_hat_Z"] is None else list(map(int, out_correct["S_hat_Z"]))
        },

        "wrong_run": {
            "stopped_by_rule": bool(out_wrong["stopped_by_rule"]),
            "timed_out": bool(out_wrong["timed_out"]),
            "tau": None if out_wrong["tau"] is None else int(out_wrong["tau"]),
            "shortlist_source": out_wrong["shortlist_source"],
            "guarantee_valid": bool(out_wrong["guarantee_valid"]),
            "S_hat_Z": None if out_wrong["S_hat_Z"] is None else list(map(int, out_wrong["S_hat_Z"]))
        }
    })

    print(f"Saved debug summary to: {prefix}_summary.json")

    fig.tight_layout()
    prefix = os.path.join(RESULTS_DIR, "obd_variance_debug_k3_d01_n50000")
    fig.savefig(prefix + "_trajectory.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved: {prefix}_trajectory.png")
    print("\nDone.")


if __name__ == "__main__":
    main()