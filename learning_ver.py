"""
Learning FWSP (one-hot per round) for Gaussian shortlisting.

This implementation augments the FWSP (Frank–Wolfe Self-Play) idea with:
  • A posterior-sample plug-in for value/gradient calls (to encourage exploration).
  • Counts-based allocation p_n (implementing the 1/(n+1) FW step implicitly).
  • A fixed-confidence stopping rule using a time-adaptive threshold β(n, δ).
  • Simple trajectory logging and plotting utilities (log-scale x-axis).

Posterior model (improper flat Gaussian prior, known σ_i^2):
  θ_i | data ~ N( θ̂_i,  σ_i^2 / N_i ), with N_i ≥ 1 ensured by burn-in.

Notes on logic:
  • Each FWSP round draws ONE posterior sample θ̃(n) and uses it consistently
	in all calls to scenario_value_and_grad(·) for shortlist/scenario/gradient choices.
  • The fixed-confidence stopping rule uses the sample mean θ̂(n), not θ̃(n).
  • The allocation gradient can be taken either over the active shortlist only
	or over all (ρ, μ)-weighted shortlists (controlled by use_active_shortlist_grads).
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import multiprocessing as mp
import os
import time

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def all_k_subsets(K: int, k: int):
	"""Return all sorted k-subsets of {0, …, K-1} as tuples."""
	return [tuple(sorted(c)) for c in combinations(range(K), k)]


def complement_indices(K: int, S: Tuple[int, ...]):
	"""Return the sorted complement of S within {0, …, K-1}."""
	return tuple(sorted(set(range(K)) - set(S)))


def sort_by_theta_desc(theta: np.ndarray, idxs: Tuple[int, ...]):
	"""Return idxs sorted by descending theta values."""
	return tuple(sorted(idxs, key=lambda i: -theta[i]))


def beta_threshold(n: int, delta: float) -> float:
	"""Fixed-confidence threshold β(n, δ) = log( log(n+1)/δ + 1 )."""
	return float(np.log(np.log(n + 1.0) / float(delta) + 1.0))


def argmax_lex(d: Dict):
	"""
	Return (key, value) for the maximum value in dict d;
	break ties by the lexicographically smallest key.
	"""
	best_k, best_v = None, None
	for k, v in d.items():
		if (best_v is None) or (v > best_v) or (v == best_v and k < best_k):
			best_k, best_v = k, v
	return best_k, best_v


# ----------------------------------------------------------------------
# Exact S-local scenario oracle (values + gradients)
# ----------------------------------------------------------------------

def scenario_value_and_grad(theta_vec: np.ndarray,
							sigma2: np.ndarray,
							p_like: np.ndarray,
							S: Tuple[int, ...],
							j: int):
	"""
	Compute the exact S-local scenario value and its gradient for Gaussian noise.

	Alternative set (S-local):
		Alt_{S→j} = { ϑ : ϑ_j ≥ ϑ_i, ∀ i∈S }.

	With weights w_i = p_like[i] / σ_i^2, the inner problem reduces to
		min_t (1/2) [ w_j (t - θ_j)^2 + Σ_{i∈S} w_i (θ_i - min{θ_i, t})^2 ].

	Implementation:
	 • Scan prefixes of S sorted by θ.
	 • For each prefix m, compute candidate t_m and clamp to its interval.
	 • Evaluate Q(t) and take the best.
	 • Return (t*, payoff, ∇_p Γ, m*), where ∇_p Γ uses the envelope theorem.

	Args
	----
	theta_vec : (K,) array
		Mean vector θ.
	sigma2 : (K,) array
		Known per-arm variances σ_i^2.
	p_like : (K,) array
		“Allocation-like” weights (e.g., p or N/n).
	S : tuple[int]
		Shortlist indices.
	j : int
		Outside contender index (j ∉ S).

	Returns
	-------
	t_star : float
		Optimal threshold t*.
	payoff : float
		Γ(p_like, S→j ; θ).
	grad : (K,) array
		Gradient w.r.t. p_like: grad[i] = (θ_i - ϑ_i^*)^2 / (2 σ_i^2) on active coordinates.
	m_star : int
		Number of shortlist arms above t* (active in the pooling).
	"""
	theta = np.asarray(theta_vec, dtype=float)
	sigma2 = np.asarray(sigma2, dtype=float)
	p_like = np.asarray(p_like, dtype=float)
	K = len(theta)

	w = p_like / sigma2
	wj = w[j]
	tj = theta[j]

	S_arr = np.fromiter(S, dtype=int, count=len(S))
	order = np.argsort(-theta[S_arr], kind="stable")
	S_sorted_arr = S_arr[order]
	thetas_S = theta[S_sorted_arr]
	w_S = w[S_sorted_arr]
	k = len(S_sorted_arr)

	W_pref = np.concatenate(([0.0], np.cumsum(w_S)))
	T_pref = np.concatenate(([0.0], np.cumsum(w_S * thetas_S)))

	denom = wj + W_pref
	t_cand = np.where(denom > 0.0, (wj * tj + T_pref) / np.where(denom > 0.0, denom, 1.0), tj)

	upper = np.concatenate((thetas_S, [np.inf]))[::-1]
	upper = np.concatenate(([np.inf], thetas_S))
	lower = np.concatenate((thetas_S, [-np.inf]))
	t_clamped = np.minimum(np.maximum(t_cand, lower), upper)

	diffs = thetas_S[None, :] - np.minimum(thetas_S[None, :], t_clamped[:, None])
	Q_S = (w_S[None, :] * diffs * diffs).sum(axis=1)
	Q = 0.5 * (wj * (t_clamped - tj) ** 2 + Q_S)
	Q = np.where(denom > 0.0, Q, np.inf)

	m_best = int(np.argmin(Q))
	best_Q = float(Q[m_best])
	best_t = float(t_clamped[m_best])

	if not np.isfinite(best_Q):
		best_Q, best_t = 0.0, float(tj)

	t_star = best_t
	payoff = best_Q

	grad = np.zeros(K, dtype=float)
	grad[j] = (t_star - tj) ** 2 / (2.0 * sigma2[j])
	above_mask = thetas_S > t_star
	if above_mask.any():
		active_idx = S_sorted_arr[above_mask]
		grad[active_idx] = (theta[active_idx] - t_star) ** 2 / (2.0 * sigma2[active_idx])

	m_star = int(above_mask.sum())
	return t_star, payoff, grad, m_star


# ----------------------------------------------------------------------
# Evidence (fixed-confidence) using counts and sample means
# ----------------------------------------------------------------------

# For a given shortlist S, calculate how strong the current data is in supporting it.
def evidence_min_over_j(theta_hat: np.ndarray, # Sample mean estimate
						sigma2: np.ndarray,
						N_counts: np.ndarray, # Number of times each arm is sampled
						S: Tuple[int, ...],
						# Sc_map： For a shortlist S, which arms are outside the shortlist?
						Sc_map: Dict[Tuple[int, ...], Tuple[int, ...]]) -> float:
	"""
	Compute the shortlist evidence Z_n(S) = min_{j∈S^c} (1/2) Σ_i [N_i/σ_i^2] (θ̂_i - ϑ_i)^2.

	Implementation detail:
	 • Call scenario_value_and_grad with p_like = N / n to obtain (1/n)*Z_n(S),
		then multiply by n.
	"""
	n_total = int(N_counts.sum())
	if n_total <= 0:
		return 0.0

	p_like = N_counts / float(n_total)
	vals = []
	for j in Sc_map[S]:
		'''If outsider j wants to challenge shortlist S, how easy/difficult is this 
		scenario to distinguish under the current configuration?'''
		_, payoff_p, _, _ = scenario_value_and_grad(theta_hat, sigma2, p_like, S, j)
		vals.append(n_total * payoff_p)
	return min(vals)


# ----------------------------------------------------------------------
# Learning FWSP (posterior-sample θ̃ for value/gradient calls)
# ----------------------------------------------------------------------

def fwsp_shortlisting_learning(
    sigma2: np.ndarray,
    k: int,
    N_rounds: int,
    true_theta: np.ndarray,
    delta: Optional[float] = None,
    use_active_shortlist_grads: bool = True,
    burn_in: int = 1,
    seed: Optional[int] = None,
    verbose: bool = False,
    check_every: int = 100,
    traj_every: int = 10,
    reward_dist: str = "gaussian",
    record_trajectory: bool = True,
):
    """
    Fixed-confidence friendly FWSP.

    IMPORTANT semantics:
    - N_rounds is interpreted as TOTAL pull budget, including burn-in.
    - If stopping rule triggers, this run is a valid fixed-confidence stop.
    - If the total pull budget is exhausted before stopping, this run is marked as timeout.
    - We still compute S_hat_p at the end for diagnostics, but timeout runs
      should NOT be interpreted as fixed-confidence outputs.
    """

    sigma2 = np.asarray(sigma2, dtype=float)
    K = len(sigma2)
    assert 1 <= k < K

    if true_theta is None:
        raise ValueError("true_theta must be provided.")

    rng = np.random.default_rng(seed)
    true_theta = np.asarray(true_theta, dtype=float)
    if len(true_theta) != K:
        raise ValueError("true_theta must have length K.")

    def sample(i: int) -> float:
        if reward_dist == "gaussian":
            return float(rng.normal(true_theta[i], np.sqrt(sigma2[i])))
        elif reward_dist == "bernoulli":
            return float(rng.binomial(1, true_theta[i]))
        else:
            raise ValueError(f"Unknown reward_dist: {reward_dist!r}")

    all_S = all_k_subsets(K, k)
    Sc_map = {S: complement_indices(K, S) for S in all_S}

    N_counts = np.zeros(K, dtype=int)
    sum_rewards = np.zeros(K, dtype=float)
    theta_hat = np.zeros(K, dtype=float)

    t_traj, p_traj, theta_traj = [], [], []

    def _append_state():
        if not record_trajectory:
            return
        n_now = int(N_counts.sum())
        p_now = (N_counts / n_now) if n_now > 0 else np.full(K, 1.0 / K)
        t_traj.append(n_now)
        p_traj.append(p_now.copy())
        theta_traj.append(theta_hat.copy())

    # burn-in
    for i in range(K):
        for _ in range(burn_in):
            y = sample(i)
            N_counts[i] += 1
            sum_rewards[i] += y
            theta_hat[i] = sum_rewards[i] / N_counts[i]
            _append_state()

    total_burnin_pulls = K * burn_in
    remaining_budget = N_rounds - total_burnin_pulls
    if remaining_budget < 0:
        raise ValueError(
            f"N_rounds={N_rounds} is smaller than required burn-in budget {total_burnin_pulls}."
        )

    # initialize rho and mu
    rho_counts = Counter({S: 1.0 for S in all_S})
    mu_counts = {S: Counter({j: 1.0 for j in Sc_map[S]}) for S in all_S}

    def rho_from_counts():
        tot = sum(rho_counts.values())
        return {S: rho_counts[S] / tot for S in all_S}

    def mu_from_counts(S):
        tot = sum(mu_counts[S].values())
        return {j: mu_counts[S][j] / tot for j in Sc_map[S]}

    rho = rho_from_counts()
    mu = {S: mu_from_counts(S) for S in all_S}

    tau = None
    S_hat_Z = None
    stopped_by_rule = False
    timed_out = False

    run_start = time.time()

    round_iter = tqdm(
        range(remaining_budget),
        desc="FWSP single run",
        unit="round",
        dynamic_ncols=True,
        disable=not verbose,
    )

    for n in round_iter:
        n_total = int(N_counts.sum())
        p = N_counts / float(n_total)

        # posterior draw
        post_std = np.sqrt(sigma2 / np.maximum(N_counts, 1))
        theta_draw = theta_hat + rng.normal(0.0, 1.0, size=K) * post_std

        # shortlist best response
        h_vals = {}
        payoff_cache = {S: {} for S in all_S}
        grad_cache = {S: {} for S in all_S}

        for S in all_S:
            F_val = 0.0
            for j in Sc_map[S]:
                _, payoff, grad, _ = scenario_value_and_grad(theta_draw, sigma2, p, S, j)
                payoff_cache[S][j] = payoff
                grad_cache[S][j] = grad
                F_val += mu[S][j] * payoff
            h_vals[S] = F_val

        S_n, _ = argmax_lex(h_vals)
        rho_counts[S_n] += 1.0
        rho = rho_from_counts()

        # scenario best response
        j_n = min(Sc_map[S_n], key=lambda jj: (payoff_cache[S_n][jj], jj))
        mu_counts[S_n][j_n] += 1.0
        mu[S_n] = mu_from_counts(S_n)

        # allocation best response
        g = np.zeros(K, dtype=float)
        if use_active_shortlist_grads:
            for j, wj in mu[S_n].items():
                if wj == 0.0:
                    continue
                g += wj * grad_cache[S_n][j]
        else:
            for S in all_S:
                rS = rho[S]
                if rS == 0.0:
                    continue
                for j, wj in mu[S].items():
                    if wj == 0.0:
                        continue
                    g += rS * wj * grad_cache[S][j]

        max_g = g.max()
        i_candidates = np.flatnonzero(g == max_g)
        i_n = int(i_candidates.min())

        # sample chosen arm
        y = sample(i_n)
        N_counts[i_n] += 1
        sum_rewards[i_n] += y
        theta_hat[i_n] = sum_rewards[i_n] / N_counts[i_n]

        if record_trajectory and (n % traj_every == 0 or n == remaining_budget - 1):
            _append_state()

        # Check after every completed block of `check_every` post-burn-in rounds,
        # matching the naive baseline timing convention.
        # fixed-confidence stop check
        if delta is not None and (n + 1) % check_every == 0:
            Z_S = {
                S: evidence_min_over_j(theta_hat, sigma2, N_counts, S, Sc_map)
                for S in all_S
            }
            Z_best_S, Z_best_val = argmax_lex(Z_S)

            if Z_best_val >= beta_threshold(int(N_counts.sum()), delta):
                tau = int(N_counts.sum())
                S_hat_Z = Z_best_S
                stopped_by_rule = True
                _append_state()
                break

        if verbose and (n + 1) % max(1, check_every) == 0:
            elapsed = time.time() - run_start
            done = n + 1
            avg_per_round = elapsed / done
            eta = avg_per_round * (remaining_budget - done)

            round_iter.set_postfix({
                "elapsed_s": f"{elapsed:.1f}",
                "eta_s": f"{eta:.1f}",
                "pulls": int(N_counts.sum()),
            })

    if not stopped_by_rule:
        timed_out = True

    p_final = N_counts / max(1, int(N_counts.sum()))

    # diagnostic fallback shortlist (NOT fixed-confidence output if timed_out)
    perS_min_p = {}
    for S in all_S:
        vals = []
        for j in Sc_map[S]:
            _, payoff, _, _ = scenario_value_and_grad(theta_hat, sigma2, p_final, S, j)
            vals.append(payoff)
        perS_min_p[S] = min(vals)
    S_hat_p, _ = argmax_lex(perS_min_p)

    rho_final = rho_from_counts()
    mu_final = {S: mu_from_counts(S) for S in all_S}

    if record_trajectory:
        t_traj = np.asarray(t_traj, dtype=int)
        p_traj = np.vstack(p_traj) if len(p_traj) else np.zeros((0, K))
        theta_traj = np.vstack(theta_traj) if len(theta_traj) else np.zeros((0, K))
    else:
        t_traj = np.zeros(0, dtype=int)
        p_traj = np.zeros((0, K))
        theta_traj = np.zeros((0, K))

    status = "stopped" if stopped_by_rule else "timeout"
    shortlist_source = "S_hat_Z" if stopped_by_rule else "S_hat_p"

    return {
        "tau": tau,
        "status": status,
        "stopped_by_rule": stopped_by_rule,
        "timed_out": timed_out,
        "guarantee_valid": stopped_by_rule,
        "shortlist_source": shortlist_source,
        "theta_hat": theta_hat,
        "N_counts": N_counts,
        "p": p_final,
        "rho": rho_final,
        "mu": mu_final,
        "S_hat_Z": S_hat_Z,   # valid only if stopped_by_rule=True
        "S_hat_p": S_hat_p,   # diagnostic only
        "perS_min_p": perS_min_p,
        "t_traj": t_traj,
        "p_traj": p_traj,
        "theta_traj": theta_traj,
    }

# ----------------------------------------------------------------------
# Plotting helpers (log-scale x-axis)
# ----------------------------------------------------------------------

def plot_trajectories(t_traj: np.ndarray,
					 p_traj: np.ndarray,
					 theta_traj: np.ndarray,
					 true_theta: np.ndarray):
	"""
	Plot trajectories:
	 1) Allocation proportions p_i vs total pulls (log-scale x-axis).
	 2) Sample means θ̂_i vs total pulls (log-scale x-axis).
		If true means are provided, draw matching-color dashed reference lines.
	"""
	if t_traj.size == 0:	
		print("No trajectory data to plot.")
		return

	K = p_traj.shape[1]

	# Allocation trajectories
	plt.figure()
	for i in range(K):
		plt.plot(t_traj, p_traj[:, i], label=f"arm {i+1}")
	plt.xscale("log")
	plt.xlabel("Total pulls")
	plt.ylabel("Allocation proportion p_i")
	plt.title("FWSP Learning: Allocation Trajectories")
	plt.grid(True, which="both", linestyle="--", linewidth=0.5)
	plt.legend()
	plt.tight_layout()

	# Sample-mean trajectories
	plt.figure()
	for i in range(K):
		plt.plot(t_traj, theta_traj[:, i], label=f"arm {i+1}")
	if true_theta is not None:
		for i in range(len(true_theta)):
			plt.axhline(true_theta[i], linestyle="--", linewidth=1, color=f"C{i}", alpha=0.5)
	plt.xscale("log")
	plt.xlabel("Total pulls")
	plt.ylabel(r"Sample mean $\hat{\theta}_i$")
	plt.title("FWSP Learning: Sample-Mean Trajectories")
	plt.grid(True, which="both", linestyle="--", linewidth=0.5)
	plt.legend()
	plt.tight_layout()
	plt.show()

def topk_overlap_and_error(S_hat, true_theta: np.ndarray, k: int):
    """
    Shortlist-quality metric.

    We define:
    - overlap_rate = |S_hat ∩ S_true_topk| / k
    - ranking_error = 1 - overlap_rate

    This is NOT a full ranking loss over all arms.
    Instead, it measures top-k shortlist quality,
    aligned with the project goal.
    """
    true_topk = tuple(sorted(np.argsort(-true_theta)[:k]))
    S_hat = tuple(sorted(S_hat))

    overlap_count = len(set(S_hat) & set(true_topk))
    overlap_rate = overlap_count / float(k)
    ranking_error = 1.0 - overlap_rate

    return overlap_count, overlap_rate, ranking_error

# ----------------------------------------------------------------------
# Naive baseline: uniform sampling, return top-k empirical means
# ----------------------------------------------------------------------

def naive_topk_baseline(
    true_theta: np.ndarray,
    sigma2: np.ndarray,
    k: int,
    delta: float,
    N_max: int,
    burn_in: int = 1,
    seed: Optional[int] = None,
    reward_dist: str = "gaussian",
    check_every: int = 10,
):
    """
    Naive capped-budget baseline.

    Semantics:
    - If the stopping rule triggers, S_hat is the evidence-based shortlist.
    - If the stopping rule does not trigger before N_max, fall back to empirical top-k.
    - Suitable for capped-budget evaluation, not strict fixed-confidence analysis.
    """
    rng = np.random.default_rng(seed)
    K = len(true_theta)
    sigma2 = np.asarray(sigma2, dtype=float)

    def draw(i: int) -> float:
        if reward_dist == "gaussian":
            return float(rng.normal(true_theta[i], np.sqrt(sigma2[i])))
        elif reward_dist == "bernoulli":
            return float(rng.binomial(1, true_theta[i]))
        else:
            raise ValueError(f"Unknown reward_dist: {reward_dist!r}")

    N_counts = np.zeros(K, dtype=int)
    sum_rewards = np.zeros(K, dtype=float)
    theta_hat = np.zeros(K, dtype=float)

    best_arm = int(np.argmax(true_theta))

    # Burn-in
    for i in range(K):
        for _ in range(burn_in):
            y = draw(i)
            N_counts[i] += 1
            sum_rewards[i] += y
            theta_hat[i] = sum_rewards[i] / N_counts[i]

    total_burnin_pulls = K * burn_in
    remaining_budget = N_max - total_burnin_pulls
    if remaining_budget < 0:
        raise ValueError(
            f"N_max={N_max} is smaller than required burn-in budget {total_burnin_pulls}."
        )

    all_S = all_k_subsets(K, k)
    Sc_map = {S: complement_indices(K, S) for S in all_S}

    tau = None
    S_hat = None

    for n_round in range(remaining_budget):
        arm = n_round % K
        y = draw(arm)
        N_counts[arm] += 1
        sum_rewards[arm] += y
        theta_hat[arm] = sum_rewards[arm] / N_counts[arm]

        if n_round % check_every != 0:
            continue

        n_total = int(N_counts.sum())

        Z_best_val = -np.inf
        Z_best_S = None
        for S in all_S:
            Z_val = evidence_min_over_j(theta_hat, sigma2, N_counts, S, Sc_map)
            if Z_val > Z_best_val:
                Z_best_val = Z_val
                Z_best_S = S

        if Z_best_val >= beta_threshold(n_total, delta):
            tau = n_total
            S_hat = Z_best_S
            break

    if S_hat is None:
        ranked = np.argsort(-theta_hat)[:k]
        S_hat = tuple(sorted(ranked))
        tau = int(N_counts.sum())

    success = best_arm in S_hat
    overlap_count, overlap_rate, ranking_error = topk_overlap_and_error(
        S_hat, true_theta, k
    )

    return {
        "tau": tau,
        "S_hat": S_hat,
        "success": success,
        "N_counts": N_counts,
        "theta_hat": theta_hat,
        "overlap_count": overlap_count,
        "overlap_rate": overlap_rate,
        "ranking_error": ranking_error,
    }

def epsilon_greedy_topk_baseline(
    true_theta: np.ndarray,
    sigma2: np.ndarray,
    k: int,
    N_max: int,
    epsilon: float = 0.1,
    burn_in: int = 1,
    seed: Optional[int] = None,
    reward_dist: str = "gaussian",
):
    """
    Epsilon-greedy capped-budget baseline.

    Semantics:
    - With probability epsilon: explore uniformly at random
    - Otherwise: exploit the arm with the highest empirical mean
    - At the end of the budget, return empirical top-k shortlist

    IMPORTANT:
    - This is a practical capped-budget baseline only.
    - It does not implement a fixed-confidence stopping rule.
    - The returned tau therefore means "total pulls used", not "formal stopping time".
    """
    rng = np.random.default_rng(seed)
    K = len(true_theta)
    sigma2 = np.asarray(sigma2, dtype=float)

    def draw(i: int) -> float:
        if reward_dist == "gaussian":
            return float(rng.normal(true_theta[i], np.sqrt(sigma2[i])))
        elif reward_dist == "bernoulli":
            return float(rng.binomial(1, true_theta[i]))
        else:
            raise ValueError(f"Unknown reward_dist: {reward_dist!r}")

    N_counts = np.zeros(K, dtype=int)
    sum_rewards = np.zeros(K, dtype=float)
    theta_hat = np.zeros(K, dtype=float)

    best_arm = int(np.argmax(true_theta))

    # Burn-in
    for i in range(K):
        for _ in range(burn_in):
            y = draw(i)
            N_counts[i] += 1
            sum_rewards[i] += y
            theta_hat[i] = sum_rewards[i] / N_counts[i]

    total_burnin_pulls = K * burn_in
    remaining_budget = N_max - total_burnin_pulls
    if remaining_budget < 0:
        raise ValueError(
            f"N_max={N_max} is smaller than required burn-in budget {total_burnin_pulls}."
        )

    for _ in range(remaining_budget):
        # explore
        if rng.random() < epsilon:
            arm = int(rng.integers(0, K))
        else:
            # exploit current empirical best
            best_val = np.max(theta_hat)
            candidates = np.flatnonzero(theta_hat == best_val)
            arm = int(candidates.min())

        y = draw(arm)
        N_counts[arm] += 1
        sum_rewards[arm] += y
        theta_hat[arm] = sum_rewards[arm] / N_counts[arm]

    ranked = np.argsort(-theta_hat)[:k]
    S_hat = tuple(sorted(ranked))

    success = best_arm in S_hat
    overlap_count, overlap_rate, ranking_error = topk_overlap_and_error(
        S_hat, true_theta, k
    )

    return {
        "tau": int(N_counts.sum()),
        "S_hat": S_hat,
        "success": success,
        "N_counts": N_counts,
        "theta_hat": theta_hat,
        "overlap_count": overlap_count,
        "overlap_rate": overlap_rate,
        "ranking_error": ranking_error,
    }

# ----------------------------------------------------------------------
# Multi-trial simulation
# ----------------------------------------------------------------------

def _trial_worker_mp(args):
    """
    Module-level worker for multiprocessing.Pool.

    Defined at module scope (not as a closure) so it is picklable on every
    start method (fork, forkserver, spawn). Replaces the previous
    joblib.Parallel call, which crashed on Python 3.11.0a7 because that
    alpha is missing subprocess._USE_VFORK.

    Args:
        args: (trial_idx, params_dict) tuple.

    Returns:
        Per-trial result dict — same shape as the previous _run_one_trial_local.
    """
    trial_idx, params = args
    true_theta = np.asarray(params["true_theta"], dtype=float)
    sigma2     = np.asarray(params["sigma2"], dtype=float)
    k          = params["k"]
    delta      = params["delta"]
    N_max      = params["N_max"]
    burn_in    = params["burn_in"]
    base_seed  = params["base_seed"]
    use_active_shortlist_grads = params["use_active_shortlist_grads"]
    reward_dist = params["reward_dist"]
    run_naive   = params["run_naive"]
    run_egreedy = params["run_egreedy"]
    epsilon     = params["epsilon"]
    check_every_fwsp  = params["check_every_fwsp"]
    check_every_naive = params["check_every_naive"]

    best_arm = int(np.argmax(true_theta))
    seed = base_seed + trial_idx

    out = fwsp_shortlisting_learning(
        record_trajectory=False,
        sigma2=sigma2,
        k=k,
        N_rounds=N_max,
        true_theta=true_theta,
        delta=delta,
        use_active_shortlist_grads=use_active_shortlist_grads,
        burn_in=burn_in,
        seed=seed,
        verbose=False,
        reward_dist=reward_dist,
        check_every=check_every_fwsp,
    )

    stopped = out["stopped_by_rule"]
    S_final = out["S_hat_Z"] if stopped else out["S_hat_p"]

    overlap_count, overlap_rate, ranking_error = topk_overlap_and_error(
        S_final, true_theta, k
    )

    result = {
        "stopped": stopped,
        "pulls_used": out["tau"] if stopped else int(out["N_counts"].sum()),
        "success": best_arm in S_final,
        "alloc": out["p"].copy(),
        "shortlist": S_final,
        "shortlist_source": out["shortlist_source"],
        "overlap_count": overlap_count,
        "overlap_rate": overlap_rate,
        "ranking_error": ranking_error,
    }

    if run_naive:
        nb = naive_topk_baseline(
            true_theta=true_theta,
            sigma2=sigma2,
            k=k,
            delta=delta,
            N_max=N_max,
            burn_in=burn_in,
            seed=seed,
            reward_dist=reward_dist,
            check_every=check_every_naive,
        )
        result["naive_tau"] = nb["tau"]
        result["naive_success"] = nb["success"]

        nb_overlap_count, nb_overlap_rate, nb_ranking_error = topk_overlap_and_error(
            nb["S_hat"], true_theta, k
        )
        result["naive_overlap_rate"] = nb_overlap_rate
        result["naive_ranking_error"] = nb_ranking_error
    else:
        result["naive_tau"] = 0
        result["naive_success"] = False
        result["naive_overlap_rate"] = 0.0
        result["naive_ranking_error"] = 1.0

    if run_egreedy:
        eg = epsilon_greedy_topk_baseline(
            true_theta=true_theta,
            sigma2=sigma2,
            k=k,
            N_max=N_max,
            epsilon=epsilon,
            burn_in=burn_in,
            seed=seed,
            reward_dist=reward_dist,
        )
        result["egreedy_tau"] = eg["tau"]
        result["egreedy_success"] = eg["success"]
        result["egreedy_overlap_rate"] = eg["overlap_rate"]
        result["egreedy_ranking_error"] = eg["ranking_error"]
    else:
        result["egreedy_tau"] = 0
        result["egreedy_success"] = False
        result["egreedy_overlap_rate"] = 0.0
        result["egreedy_ranking_error"] = 1.0

    return result


def run_trials(
    true_theta: np.ndarray,
    sigma2: np.ndarray,
    k: int,
    delta: float,
    N_max: int,
    n_trials: int,
    burn_in: int = 5,
    use_active_shortlist_grads: bool = True,
    base_seed: int = 0,
    verbose: bool = False,
    reward_dist: str = "gaussian",
    run_naive: bool = True,
    run_egreedy: bool = False,
    epsilon: float = 0.1,
    n_jobs: int = -1,
    check_every_fwsp: int = 50,
    check_every_naive: int = 50,
):
    """
    Capped-budget practical evaluation runner.

    This runner compares:
    - FWSP shortlisting
    - naive baseline
    - epsilon-greedy baseline (optional)

    Important semantics:
    - FWSP may stop via fixed-confidence rule
    - Otherwise fallback to S_hat_p
    - epsilon-greedy has NO stopping rule
    """
    n_cores = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    print(f"\nRunning {n_trials} trials on {n_cores} CPU cores (n_jobs={n_jobs})")
    print(f"FWSP check_every = {check_every_fwsp}, Naive check_every = {check_every_naive}")

    # Build a picklable params dict shared across workers (read-only).
    params = {
        "true_theta": np.asarray(true_theta, dtype=float),
        "sigma2": np.asarray(sigma2, dtype=float),
        "k": k,
        "delta": delta,
        "N_max": N_max,
        "burn_in": burn_in,
        "base_seed": base_seed,
        "use_active_shortlist_grads": use_active_shortlist_grads,
        "reward_dist": reward_dist,
        "run_naive": run_naive,
        "run_egreedy": run_egreedy,
        "epsilon": epsilon,
        "check_every_fwsp": check_every_fwsp,
        "check_every_naive": check_every_naive,
    }
    args_iter = [(i, params) for i in range(n_trials)]

    start_time = time.time()

    if n_cores <= 1:
        # Serial path — useful for debugging and for environments where
        # multiprocessing is unavailable (e.g. some Jupyter kernels).
        results_list = [
            _trial_worker_mp(a)
            for a in tqdm(args_iter, desc="trials", total=n_trials)
        ]
    else:
        # Parallel path. We use the default start method (fork on Linux,
        # spawn on macOS/Windows). This avoids joblib's loky+subprocess
        # machinery that crashes on Python 3.11.0a7 due to missing
        # subprocess._USE_VFORK.
        with mp.Pool(processes=n_cores) as pool:
            results_list = list(tqdm(
                pool.imap(_trial_worker_mp, args_iter, chunksize=1),
                total=n_trials,
                desc="trials",
            ))

    total_elapsed = time.time() - start_time
    if total_elapsed < 60:
        print(f"\nActual total time: {total_elapsed:.1f} seconds")
    else:
        print(f"\nActual total time: {total_elapsed/60:.1f} minutes")

    fwsp_pulls_used = np.array([r["pulls_used"] for r in results_list])
    fwsp_succ = np.array([r["success"] for r in results_list])
    fwsp_stopped = np.array([r["stopped"] for r in results_list])
    fwsp_allocs = np.vstack([r["alloc"] for r in results_list])
    fwsp_shorts = [r["shortlist"] for r in results_list]
    fwsp_shortlist_sources = [r["shortlist_source"] for r in results_list]
    fwsp_overlap = np.array([r["overlap_rate"] for r in results_list])
    fwsp_rankerr = np.array([r["ranking_error"] for r in results_list])

    naive_taus = np.array([r["naive_tau"] for r in results_list])
    naive_succ = np.array([r["naive_success"] for r in results_list])
    naive_overlap = np.array([r["naive_overlap_rate"] for r in results_list])
    naive_rankerr = np.array([r["naive_ranking_error"] for r in results_list])

    egreedy_taus = np.array([r["egreedy_tau"] for r in results_list])
    egreedy_succ = np.array([r["egreedy_success"] for r in results_list])
    egreedy_overlap = np.array([r["egreedy_overlap_rate"] for r in results_list])
    egreedy_rankerr = np.array([r["egreedy_ranking_error"] for r in results_list])

    print(f"  Stop fraction (guarantee-valid): {fwsp_stopped.mean():.3f}")
    print(f"  Overall success rate (capped-budget): {fwsp_succ.mean():.3f}")
    print(f"  Success rate among stopped runs: "
          f"{fwsp_succ[fwsp_stopped].mean():.3f}" if np.any(fwsp_stopped) else
          "  Success rate among stopped runs: None")
    print(f"  Mean pulls used: {fwsp_pulls_used.mean():,.0f}")

    return {
        "fwsp": {
            "pulls_used": fwsp_pulls_used,
            "successes": fwsp_succ,
            "stopped_flags": fwsp_stopped,
            "allocations": fwsp_allocs,
            "shortlists": fwsp_shorts,
            "shortlist_sources": fwsp_shortlist_sources,
            "overlap_rates": fwsp_overlap,
            "ranking_errors": fwsp_rankerr,
        },
        "naive": {
            "taus": naive_taus,
            "successes": naive_succ,
            "overlap_rates": naive_overlap,
            "ranking_errors": naive_rankerr,
        },
        "egreedy": {
            "taus": egreedy_taus,
            "successes": egreedy_succ,
            "overlap_rates": egreedy_overlap,
            "ranking_errors": egreedy_rankerr,
        },
    }


# ----------------------------------------------------------------------
# Comprehensive plotting
# ----------------------------------------------------------------------

def plot_experiment_results(
    results: dict,
    true_theta: np.ndarray,
    k: int,
    save_prefix: str = "fwsp",
):
    """
    Generate and save three diagnostic plots from multi-trial results:
      1. Stopping time / pulls-used distributions (FWSP vs naive)
      2. Mean allocation proportions across trials
      3. Arm selection frequency in final shortlists
    """
    K = len(true_theta)
    fwsp = results["fwsp"]
    naive = results["naive"]

    # 1. Stopping time distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    upper = max(fwsp["pulls_used"].max(), naive["taus"].max()) * 1.1 if len(naive["taus"]) > 0 else fwsp["pulls_used"].max() * 1.1
    bins = np.linspace(0, upper, 40)

    ax.hist(fwsp["pulls_used"], bins=bins, alpha=0.6, label="FWSP", edgecolor="black")
    if len(naive["taus"]) > 0 and np.any(naive["taus"] > 0):
        ax.hist(naive["taus"], bins=bins, alpha=0.6, label="Naive (round-robin)", edgecolor="black")
        ax.axvline(naive["taus"].mean(), color="C1", linestyle="--", linewidth=1.5,
                   label=f"Naive mean = {naive['taus'].mean():.0f}")

    ax.axvline(fwsp["pulls_used"].mean(), color="C0", linestyle="--", linewidth=1.5,
               label=f"FWSP mean = {fwsp['pulls_used'].mean():.0f}")

    ax.set_xlabel("Pulls used / stopping time")
    ax.set_ylabel("Frequency")
    ax.set_title("Pulls Used Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_stopping_times.png", dpi=150)
    plt.close(fig)

    # 2. Mean allocation proportions
    mean_alloc = fwsp["allocations"].mean(axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(1, K + 1), mean_alloc, color=[f"C{i}" for i in range(K)],
                  edgecolor="black", alpha=0.8)
    for bar, val in zip(bars, mean_alloc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Arm")
    ax.set_ylabel("Mean allocation proportion")
    ax.set_title("Average FWSP Allocation Across Trials")
    ax.set_xticks(range(1, K + 1))
    ax.set_xticklabels([f"Arm {i+1}\n(θ={true_theta[i]:.2f})" for i in range(K)], fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_allocation.png", dpi=150)
    plt.close(fig)

    # 3. Arm selection frequency
    arm_freq = np.zeros(K, dtype=int)
    for S in fwsp["shortlists"]:
        for i in S:
            arm_freq[i] += 1
    n_trials = len(fwsp["shortlists"])
    arm_pct = arm_freq / n_trials * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(1, K + 1), arm_pct, color=[f"C{i}" for i in range(K)],
                  edgecolor="black", alpha=0.8)
    for bar, pct in zip(bars, arm_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Arm")
    ax.set_ylabel("Selection frequency (%)")
    ax.set_title(f"Arm Inclusion in Final Shortlist (k={k})")
    ax.set_xticks(range(1, K + 1))
    ax.set_xticklabels([f"Arm {i+1}\n(θ={true_theta[i]:.2f})" for i in range(K)], fontsize=8)
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_arm_selection.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved: {save_prefix}_stopping_times.png, "
          f"{save_prefix}_allocation.png, {save_prefix}_arm_selection.png")


# ----------------------------------------------------------------------
# Main: simulation experiment
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # Problem instance
    true_theta = np.array([0.50, 0.40, 0.30, 0.20, 0.10], dtype=float)
    sigma2     = np.ones_like(true_theta)
    k          = 2
    delta      = 0.05
    N_max      = 50_000
    n_trials   = 20
    burn_in    = 5
    best_arm   = int(np.argmax(true_theta))

    print("=" * 60)
    print("FWSP Learning Experiment")
    print("=" * 60)
    print(f"  K = {len(true_theta)}, k = {k}, delta = {delta}")
    print(f"  theta = {true_theta}")
    print(f"  Best arm: {best_arm + 1} (theta* = {true_theta[best_arm]:.2f})")
    print(f"  Trials: {n_trials}, Max rounds: {N_max}, Burn-in: {burn_in}")
    print()

    # --- Single illustrative run ---
    print("--- Single Run (seed=42) ---")
    out = fwsp_shortlisting_learning(
        sigma2=sigma2, k=k, N_rounds=N_max,
        true_theta=true_theta, delta=delta,
        use_active_shortlist_grads=True,
        burn_in=burn_in, seed=42, verbose=False,
    )

    S_final = out["S_hat_Z"] if out["S_hat_Z"] is not None else out["S_hat_p"]
    print(f"  Stopping time tau = {out['tau']}")
    print(f"  Shortlist (evidence)  = {tuple(i+1 for i in S_final)}")
    print(f"  Shortlist (Gamma-val) = {tuple(i+1 for i in out['S_hat_p'])}")
    print(f"  Final allocation p    = {np.round(out['p'], 4)}")
    print(f"  Sample means theta_hat= {np.round(out['theta_hat'], 4)}")
    print(f"  Best arm {best_arm+1} in shortlist? {best_arm in S_final}")
    print()

    # Save single-run trajectory plots
    if out["t_traj"].size > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        for i in range(len(true_theta)):
            ax.plot(out["t_traj"], out["p_traj"][:, i],
                    label=f"Arm {i+1} (theta={true_theta[i]:.2f})")
        ax.set_xscale("log")
        ax.set_xlabel("Total pulls")
        ax.set_ylabel("Allocation proportion p_i")
        ax.set_title("FWSP Learning: Allocation Trajectory")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        ax = axes[1]
        for i in range(len(true_theta)):
            ax.plot(out["t_traj"], out["theta_traj"][:, i], label=f"Arm {i+1}")
            ax.axhline(true_theta[i], linestyle="--", linewidth=1, color=f"C{i}", alpha=0.4)
        ax.set_xscale("log")
        ax.set_xlabel("Total pulls")
        ax.set_ylabel("Sample mean theta_hat_i")
        ax.set_title("FWSP Learning: Sample-Mean Trajectory")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        fig.tight_layout()
        fig.savefig("fwsp_trajectories.png", dpi=150)
        plt.close(fig)
        print("  Trajectory plot saved: fwsp_trajectories.png")

    # --- Multi-trial experiment ---
    print()
    print(f"--- Multi-Trial Experiment ({n_trials} trials) ---")
    results = run_trials(
        true_theta=true_theta,
        sigma2=sigma2,
        k=k,
        delta=delta,
        N_max=N_max,
        n_trials=n_trials,
        burn_in=burn_in,
        use_active_shortlist_grads=True,
        base_seed=0,
        verbose=True,
        reward_dist="gaussian",
        run_naive=True,
        run_egreedy=True,
        epsilon=0.1,
        n_jobs=-1,
        check_every_fwsp=50,
        check_every_naive=50,
    )

    fwsp_res = results["fwsp"]
    naive_res = results["naive"]
    egreedy_res = results["egreedy"]

    print()
    print("  FWSP results:")
    print(f"    Success rate: {fwsp_res['successes'].mean():.4f}  (target >= {1 - delta:.2f})")
    print(f"    Stop fraction: {fwsp_res['stopped_flags'].mean():.4f}")
    print(f"    Mean pulls used: {fwsp_res['pulls_used'].mean():.1f}  +/- {fwsp_res['pulls_used'].std():.1f}")
    print(f"    Median pulls used: {np.median(fwsp_res['pulls_used']):.0f}")
    print(f"    Mean top-k overlap: {fwsp_res['overlap_rates'].mean():.4f}")
    print(f"    Mean ranking error: {fwsp_res['ranking_errors'].mean():.4f}")

    print()
    print("  Naive baseline results:")
    print(f"    Success rate: {naive_res['successes'].mean():.4f}")
    print(f"    Mean stopping time: {naive_res['taus'].mean():.1f}  +/- {naive_res['taus'].std():.1f}")
    print(f"    Median stopping time: {np.median(naive_res['taus']):.0f}")
    print(f"    Mean top-k overlap: {naive_res['overlap_rates'].mean():.4f}")
    print(f"    Mean ranking error: {naive_res['ranking_errors'].mean():.4f}")

    print()
    print("  Epsilon-greedy results:")
    print(f"    Success rate: {egreedy_res['successes'].mean():.4f}")
    print(f"    Mean pulls used: {egreedy_res['taus'].mean():.1f}  +/- {egreedy_res['taus'].std():.1f}")
    print(f"    Median pulls used: {np.median(egreedy_res['taus']):.0f}")
    print(f"    Mean top-k overlap: {egreedy_res['overlap_rates'].mean():.4f}")
    print(f"    Mean ranking error: {egreedy_res['ranking_errors'].mean():.4f}")

    plot_experiment_results(results, true_theta, k, save_prefix="fwsp")

    print()
    print("=" * 60)
    print("Done.")