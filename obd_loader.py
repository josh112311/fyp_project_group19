"""
obd_loader.py

Load the ZOZOTOWN Open Bandit Dataset (OBD) and produce obd_arms.npz
for use by obd_experiment.py.

Usage
-----
1. Download the OBD dataset from https://research.zozo.com/data.html
2. Unzip so that you have a directory structure like:
       obd/random/all.csv      (or bts/all.csv, etc.)
3. Run:
       python obd_loader.py --csv obd/random/all.csv

   This will produce obd_arms.npz with arrays:
     - true_theta : (K,) empirical CTR per item, sorted descending
     - sigma2     : (K,) Bernoulli variance theta*(1-theta)
     - item_ids   : (K,) original item_id indices
     - n_impr     : (K,) number of impressions per item

Notes
-----
* We use the RANDOM policy data (not BTS) because random assignment
  gives unbiased CTR estimates — each item's empirical click rate is
  an unbiased estimate of its true CTR.
* We aggregate across all positions (1, 2, 3). If you want position-
  specific CTRs, use --position to filter.
* Items with fewer than --min-impr impressions are dropped to ensure
  stable CTR estimates.
* The "All" campaign has ~80 items; "Men" ~34; "Women" ~46.
"""

import argparse
import csv
import sys
import numpy as np
from collections import defaultdict



def load_obd_csv(csv_path: str, position: int = None, min_impr: int = 100):
    """
    Parse an OBD CSV file and compute per-item CTR.

    Parameters
    ----------
    csv_path : str
        Path to the CSV, e.g. obd/random/all.csv
    position : int or None
        If given, only count impressions at this position (1, 2, or 3).
    min_impr : int
        Minimum number of impressions required to include an item.

    Returns
    -------
    item_ids : np.ndarray of int
        Sorted by descending CTR.
    true_theta : np.ndarray of float
        Empirical CTR per item (descending order).
    sigma2 : np.ndarray of float
        Bernoulli variance = theta * (1 - theta).
    n_impr : np.ndarray of int
        Impression count per item.
    """
    # Accumulate clicks and impressions per item_id
    clicks = defaultdict(int)
    impressions = defaultdict(int)

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Check required columns exist
        first_row = next(reader)
        required = {"item_id", "click"}
        if not required.issubset(set(first_row.keys())):
            # Try common alternative column names
            print(f"Available columns: {list(first_row.keys())}")
            raise KeyError(f"CSV must contain columns: {required}")

        # Process first row
        def process_row(row):
            if position is not None and "position" in row:
                if int(row["position"]) != position:
                    return
            item = int(row["item_id"])
            click = int(row["click"])
            impressions[item] += 1
            clicks[item] += click

        process_row(first_row)
        for row in reader:
            process_row(row)

    # Filter by minimum impressions
    valid_items = [i for i in impressions if impressions[i] >= min_impr]
    if len(valid_items) == 0:
        raise ValueError(
            f"No items with >= {min_impr} impressions. "
            f"Total items: {len(impressions)}, "
            f"max impressions: {max(impressions.values()) if impressions else 0}"
        )

    # Compute CTR and sort descending
    items_ctr = []
    for item in valid_items:
        n = impressions[item]
        ctr = clicks[item] / n
        items_ctr.append((item, ctr, n))

    items_ctr.sort(key=lambda x: -x[1])  # descending by CTR

    item_ids = np.array([x[0] for x in items_ctr], dtype=int)
    true_theta = np.array([x[1] for x in items_ctr], dtype=float)
    n_impr_arr = np.array([x[2] for x in items_ctr], dtype=int)
    sigma2 = true_theta * (1.0 - true_theta)

    return item_ids, true_theta, sigma2, n_impr_arr


def main():
    parser = argparse.ArgumentParser(
        description="Load ZOZOTOWN OBD data and save as obd_arms.npz"
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to the OBD CSV file (e.g. obd/random/all.csv)"
    )
    parser.add_argument(
        "--position", type=int, default=None, choices=[1, 2, 3],
        help="Filter by display position (default: aggregate all positions)"
    )
    parser.add_argument(
        "--min-impr", type=int, default=100,
        help="Minimum impressions to include an item (default: 100)"
    )
    parser.add_argument(
        "--output", default="obd_arms.npz",
        help="Output .npz file path (default: obd_arms.npz)"
    )
    args = parser.parse_args()

    print(f"Loading OBD data from: {args.csv}")
    if args.position:
        print(f"  Filtering to position: {args.position}")
    print(f"  Minimum impressions: {args.min_impr}")

    item_ids, true_theta, sigma2, n_impr = load_obd_csv(
        args.csv, position=args.position, min_impr=args.min_impr
    )

    K = len(item_ids)
    print(f"\nFound {K} items meeting criteria.")
    print(f"\n{'Rank':>4}  {'item_id':>8}  {'CTR':>10}  {'Impressions':>12}  {'sigma2':>10}")
    for r in range(K):
        print(f"{r+1:>4}  {item_ids[r]:>8}  {true_theta[r]:>9.4%}  "
              f"{n_impr[r]:>12,}  {sigma2[r]:>10.6f}")

    print(f"\nCTR range: [{true_theta[-1]:.4%}, {true_theta[0]:.4%}]")
    if K >= 2:
        print(f"Gap (1st - 2nd): {true_theta[0] - true_theta[1]:.4%}")
    print(f"Total impressions: {n_impr.sum():,}")

    np.savez(
        args.output,
        true_theta=true_theta,
        sigma2=sigma2,
        item_ids=item_ids,
        n_impr=n_impr,
    )
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
