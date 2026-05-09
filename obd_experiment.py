"""
OBD practical evaluation script.

This script evaluates FWSP on a real-data-inspired Bernoulli bandit instance
constructed from the ZOZOTOWN Open Bandit Dataset.

Important:
- This is a capped-budget practical evaluation, not a strict formal fixed-confidence benchmark.
- Bernoulli variances are recomputed as theta * (1 - theta) for consistency with the reward model.
- The optional epsilon-greedy baseline is included as a reward-oriented comparison method.
"""

import argparse
import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

from learning_ver import (
    fwsp_shortlisting_learning,
    run_trials,
    plot_experiment_results,
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


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


def make_obd_prefix(k: int, delta: float, n_max: int, k_keep: int):
    d_str = str(delta).replace(".", "")
    return os.path.join(
        RESULTS_DIR,
        f"obd_capped_kkeep{k_keep}_k{k}_d{d_str}_n{n_max}"
    )


def summarize_capped_budget(results: dict):
    fwsp = results["fwsp"]
    naive = results["naive"]
    egreedy = results["egreedy"]

    fwsp_stop_mask = fwsp["stopped_flags"].astype(bool)

    summary = {
        "fwsp_overall_success": float(np.mean(fwsp["successes"])),
        "fwsp_stop_fraction": float(np.mean(fwsp["stopped_flags"])),
        "fwsp_mean_pulls": float(np.mean(fwsp["pulls_used"])),
        "fwsp_median_pulls": float(np.median(fwsp["pulls_used"])),
        "fwsp_success_among_stopped": (
            float(np.mean(fwsp["successes"][fwsp_stop_mask]))
            if np.any(fwsp_stop_mask) else None
        ),
        "fwsp_mean_tau_among_stopped": (
            float(np.mean(fwsp["pulls_used"][fwsp_stop_mask]))
            if np.any(fwsp_stop_mask) else None
        ),
        "fwsp_mean_overlap": float(np.mean(fwsp["overlap_rates"])),
        "fwsp_mean_ranking_error": float(np.mean(fwsp["ranking_errors"])),

        "naive_overall_success": float(np.mean(naive["successes"])),
        "naive_mean_tau": float(np.mean(naive["taus"])),
        "naive_median_tau": float(np.median(naive["taus"])),
        "naive_mean_overlap": float(np.mean(naive["overlap_rates"])),
        "naive_mean_ranking_error": float(np.mean(naive["ranking_errors"])),

        "egreedy_overall_success": float(np.mean(egreedy["successes"])),
        "egreedy_mean_tau": float(np.mean(egreedy["taus"])),
        "egreedy_mean_overlap": float(np.mean(egreedy["overlap_rates"])),
        "egreedy_mean_ranking_error": float(np.mean(egreedy["ranking_errors"])),
    }
    return summary

def load_instance(npz_path: str, K_keep: int):
    d = np.load(npz_path)
    true_theta = d["true_theta"][:K_keep].copy()
    sigma2 = d["sigma2"][:K_keep].copy()
    item_ids = d["item_ids"][:K_keep].copy()
    n_impr = d["n_impr"][:K_keep].copy()
    return true_theta, sigma2, item_ids, n_impr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="obd_arms.npz")
    parser.add_argument("--K-keep", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--N-max", type=int, default=200_000)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--burn-in", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-single", action="store_true")
    parser.add_argument("--verbose-single", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()

    true_theta, _, item_ids, n_impr = load_instance(args.npz, args.K_keep)

    # IMPORTANT: Bernoulli rewards => use Bernoulli variance
    sigma2 = true_theta * (1.0 - true_theta)

    K = len(true_theta)
    best_arm = int(np.argmax(true_theta))
    relevant_gap = true_theta[args.k - 1] - true_theta[args.k] if K > args.k else 0.0

    print("=" * 70)
    print("OBD Experiment (Capped-Budget Practical Evaluation)")
    print("=" * 70)
    print(f"K_keep = {K}, k = {args.k}, delta = {args.delta}")
    print(f"N_max = {args.N_max:,}, n_trials = {args.n_trials}, burn_in = {args.burn_in}")
    print()

    print(f"{'rank':>4}  {'item_id':>8}  {'CTR':>10}  {'impressions':>12}")
    for r in range(K):
        marker = " <- best" if r == best_arm else ""
        print(f"{r+1:>4}  {int(item_ids[r]):>8}  {true_theta[r]:>9.4%}  {int(n_impr[r]):>12,}{marker}")

    best_second_gap = true_theta[0] - true_theta[1] if K >= 2 else 0.0
    relevant_gap = true_theta[args.k - 1] - true_theta[args.k] if K > args.k else 0.0

    print(f"\nBest-second gap θ(1) - θ(2):   {best_second_gap:.4%}")
    print(f"Relevant gap θ(k) - θ(k+1):    {relevant_gap:.4%}")

    if args.run_single:
        print("\n" + "-" * 70)
        print("Single run")
        print("-" * 70)

        out = fwsp_shortlisting_learning(
            sigma2=sigma2,
            k=args.k,
            N_rounds=args.N_max,
            true_theta=true_theta,
            delta=args.delta,
            use_active_shortlist_grads=True,
            burn_in=args.burn_in,
            seed=args.seed,
            reward_dist="bernoulli",
            verbose=args.verbose_single,
            check_every=50,
        )

        stopped = out["stopped_by_rule"]
        n_used = int(out["N_counts"].sum())

        print(f"Stopped by rule? {stopped}")
        print(f"tau = {out['tau']}")
        print(f"pulls used = {n_used:,}")
        print(f"guarantee valid = {out['guarantee_valid']}")
        print(f"shortlist source = {out['shortlist_source']}")

        if stopped:
            S_final = out["S_hat_Z"]
            print(f"Fixed-confidence shortlist = {tuple(i+1 for i in S_final)}")
        else:
            S_final = out["S_hat_p"]
            print(f"Fallback shortlist (diagnostic only) = {tuple(i+1 for i in S_final)}")

        print(f"Best arm in shortlist? {best_arm in S_final}")
        print(f"Allocation p = {np.round(out['p'], 4)}")

        if out["t_traj"].size > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax = axes[0]
            for i in range(K):
                ax.plot(out["t_traj"], out["p_traj"][:, i], label=f"Arm {i+1}")
            ax.set_xscale("log")
            ax.set_xlabel("Total pulls")
            ax.set_ylabel("Allocation proportion")
            ax.set_title("FWSP on OBD: Allocation Trajectory")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.legend(fontsize=7)

            ax = axes[1]
            for i in range(K):
                ax.plot(out["t_traj"], out["theta_traj"][:, i], label=f"Arm {i+1}")
                ax.axhline(true_theta[i], linestyle="--", linewidth=1, color=f"C{i%10}", alpha=0.4)
            ax.set_xscale("log")
            ax.set_xlabel("Total pulls")
            ax.set_ylabel(r"Empirical CTR $\hat{\theta}_i$")
            ax.set_title("FWSP on OBD: CTR Estimates")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.legend(fontsize=7)

            fig.tight_layout()
            fig.savefig(os.path.join(RESULTS_DIR, "obd_single_run.png"), dpi=150)
            plt.close(fig)
            print(f"Saved: {RESULTS_DIR}/obd_single_run.png")

    print("\n" + "-" * 70)
    print("Multi-trial capped-budget evaluation")
    print("-" * 70)

    results = run_trials(
        true_theta=true_theta,
        sigma2=sigma2,
        k=args.k,
        delta=args.delta,
        N_max=args.N_max,
        n_trials=args.n_trials,
        burn_in=args.burn_in,
        use_active_shortlist_grads=True,
        base_seed=args.seed,
        verbose=True,
        reward_dist="bernoulli",
        run_naive=True,
        run_egreedy=True,
        epsilon=args.epsilon,
        n_jobs=-1,
        check_every_fwsp=50,
        check_every_naive=50,
    )

    fwsp = results["fwsp"]
    naive = results["naive"]
    egreedy = results["egreedy"]
    summary = summarize_capped_budget(results)

    print("\nEpsilon-greedy (capped-budget):")
    print(f"  Overall success rate:              {egreedy['successes'].mean():.3f}")
    print(f"  Mean pulls used:                   {egreedy['taus'].mean():,.0f}")
    print(f"  Mean top-k overlap:                {egreedy['overlap_rates'].mean():.3f}")
    print(f"  Mean ranking error:                {egreedy['ranking_errors'].mean():.3f}")

    print("\nFWSP (capped-budget):")
    print(f"  Overall success rate:              {summary['fwsp_overall_success']:.3f}")
    print(f"  Stop fraction:                     {summary['fwsp_stop_fraction']:.3f}")
    print(f"  Mean pulls used:                   {summary['fwsp_mean_pulls']:,.0f}")
    print(f"  Median pulls used:                 {summary['fwsp_median_pulls']:,.0f}")
    print(f"  Success rate among stopped runs:   {summary['fwsp_success_among_stopped']}")
    print(f"  Mean tau among stopped runs:       {summary['fwsp_mean_tau_among_stopped']}")
    print(f"  Mean top-k overlap:                {fwsp['overlap_rates'].mean():.3f}")
    print(f"  Mean ranking error:                {fwsp['ranking_errors'].mean():.3f}")

    print("\nNaive (capped-budget):")
    print(f"  Overall success rate:              {summary['naive_overall_success']:.3f}")
    print(f"  Mean tau:                          {summary['naive_mean_tau']:,.0f}")
    print(f"  Median tau:                        {summary['naive_median_tau']:,.0f}")
    print(f"  Mean top-k overlap:                {naive['overlap_rates'].mean():.3f}")
    print(f"  Mean ranking error:                {naive['ranking_errors'].mean():.3f}")


    prefix = make_obd_prefix(args.k, args.delta, args.N_max, args.K_keep)

    plot_experiment_results(
        results,
        true_theta,
        args.k,
        save_prefix=prefix,
    )

    # save summary json
    save_json(prefix + "_summary.json", {
        "meta": {
            "K_keep": args.K_keep,
            "k": args.k,
            "delta": args.delta,
            "N_max": args.N_max,
            "n_trials": args.n_trials,
            "burn_in": args.burn_in,
            "reward_dist": "bernoulli",
            "epsilon": args.epsilon,
            "best_second_gap": float(best_second_gap),
            "relevant_gap": float(relevant_gap),
        },
        "summary": summary,
        "top_item_ids": [int(x) for x in item_ids.tolist()],
        "top_true_theta": [float(x) for x in true_theta.tolist()],
        "top_impressions": [int(x) for x in n_impr.tolist()],
    })

    # save per-trial csv
    rows = []
    for i in range(len(fwsp["pulls_used"])):
        rows.append({
            "trial": i,
            "fwsp_pulls_used": int(fwsp["pulls_used"][i]),
            "fwsp_success": bool(fwsp["successes"][i]),
            "fwsp_stopped": bool(fwsp["stopped_flags"][i]),
            "fwsp_shortlist_source": fwsp["shortlist_sources"][i],
            "naive_tau": int(naive["taus"][i]),
            "naive_success": bool(naive["successes"][i]),
            "fwsp_overlap_rate": float(fwsp["overlap_rates"][i]),
            "fwsp_ranking_error": float(fwsp["ranking_errors"][i]),
            "naive_overlap_rate": float(naive["overlap_rates"][i]),
            "naive_ranking_error": float(naive["ranking_errors"][i]),
            "egreedy_tau": int(egreedy["taus"][i]),
            "egreedy_success": bool(egreedy["successes"][i]),
            "egreedy_overlap_rate": float(egreedy["overlap_rates"][i]),
            "egreedy_ranking_error": float(egreedy["ranking_errors"][i]),
        })
    save_records_csv(prefix + "_trials.csv", rows)

    print(f"\nSaved capped-budget OBD plots and summaries with prefix:\n  {prefix}")


if __name__ == "__main__":
    main()