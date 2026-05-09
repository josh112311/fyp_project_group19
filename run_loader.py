"""
Local convenience script.

This script is only used to generate obd_arms.npz from a hard-coded local CSV path.
It is not intended as a portable entry point for other users.
Use obd_loader.py with command-line arguments for the portable version.
"""

from obd_loader import load_obd_csv
import numpy as np

# Change this path to your local OBD CSV
CSV_PATH = r"C:\Users\Bill\Desktop\IEDA\Y4S2\FYP\Stage 2\14042026\open_bandit_dataset\random\all\all.csv"

item_ids, true_theta, sigma2, n_impr = load_obd_csv(
    CSV_PATH,
    position=None,
    min_impr=100,
)

np.savez(
    "obd_arms.npz",
    true_theta=true_theta,
    sigma2=sigma2,
    item_ids=item_ids,
    n_impr=n_impr,
)

print(f"Saved {len(item_ids)} arms to obd_arms.npz")
print("Top 10 CTRs:")
for i in range(min(10, len(item_ids))):
    print(f"rank {i+1}: item_id={item_ids[i]}, CTR={true_theta[i]:.4%}, impr={n_impr[i]}")