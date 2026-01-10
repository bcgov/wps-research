#!/usr/bin/env python3
"""
Convert global_stats.pkl -> global_stats.txt
Compatible with the C classifier (BLAS/LAPACK).

Usage:
    python3 global_stats_to_txt.py
"""

import pickle
import os
import sys
import numpy as np

PKL_FILE = "global_stats.pkl"
TXT_FILE = "global_stats.txt"


def main():
    if not os.path.isfile(PKL_FILE):
        print(f"[ERROR] {PKL_FILE} not found", file=sys.stderr)
        sys.exit(1)

    with open(PKL_FILE, "rb") as f:
        data = pickle.load(f)

    if "mean_covs" not in data:
        print("[ERROR] global_stats.pkl missing 'mean_covs'", file=sys.stderr)
        sys.exit(1)

    mean_covs = data["mean_covs"]

    with open(TXT_FILE, "w") as f:
        f.write("# global_stats.txt\n")
        f.write("# Generated from global_stats.pkl\n")
        f.write("# Format:\n")
        f.write("# CLASS <id>\n")
        f.write("# DIM <vector_length>\n")
        f.write("# MEAN\n")
        f.write("# <values...>\n")
        f.write("# COV\n")
        f.write("# <row-major values...>\n\n")

        for lbl in sorted(mean_covs.keys()):
            mean, cov = mean_covs[lbl]

            if mean is None or cov is None:
                continue

            mean = np.asarray(mean, dtype=np.float64)
            cov = np.asarray(cov, dtype=np.float64)

            if mean.ndim != 1 or cov.ndim != 2:
                print(f"[ERROR] Invalid shape for class {lbl}", file=sys.stderr)
                sys.exit(1)

            dim = mean.shape[0]

            if cov.shape != (dim, dim):
                print(f"[ERROR] Covariance shape mismatch for class {lbl}", file=sys.stderr)
                sys.exit(1)

            f.write(f"CLASS {lbl}\n")
            f.write(f"DIM {dim}\n")

            f.write("MEAN\n")
            for i in range(dim):
                f.write(f"{mean[i]:.17g}\n")

            f.write("COV\n")
            # Row-major order (C/BLAS compatible)
            for i in range(dim):
                for j in range(dim):
                    f.write(f"{cov[i, j]:.17g}\n")

            f.write("\n")

    print(f"[DONE] Wrote {TXT_FILE} (pickle preserved)")


if __name__ == "__main__":
    main()

