"""4-trial smoke test for the joblib -> multiprocessing fix.
Run from the directory containing learning_ver.py. Should finish in ~30s."""
import sys, numpy as np
sys.path.insert(0, ".")
import learning_ver as lv

print("== smoke test ==")
print("python:", sys.version.split()[0])

out = lv.run_trials(
    true_theta=np.array([.5, .4, .3, .2, .1]),
    sigma2=np.ones(5),
    k=2, delta=.1, N_max=2000, n_trials=4,
    burn_in=5, base_seed=0, reward_dist="gaussian",
    run_naive=True, run_egreedy=True, n_jobs=4,
    check_every_fwsp=50, check_every_naive=50,
)
print("FWSP pulls_used:", out["fwsp"]["pulls_used"])
print("FWSP success:   ", out["fwsp"]["successes"])
print("naive  taus:    ", out["naive"]["taus"])
print("egreedy taus:   ", out["egreedy"]["taus"])
print("OK")
