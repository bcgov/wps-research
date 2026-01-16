

#!/usr/bin/env python3
"""
Export training data from pickle to binary format for CUDA program
Usage: python3 export_training.py global_stats.pkl training.bin
"""
import sys
import pickle
import numpy as np
import struct

if len(sys.argv) < 3:
    print("Usage: python3 export_training.py global_stats.pkl training.bin")
    sys.exit(1)

pkl_file = sys.argv[1]
bin_file = sys.argv[2]

print(f"[STATUS] Loading {pkl_file}...")
with open(pkl_file, "rb") as f:
    data = pickle.load(f)

train_vectors = data["train_vectors"]
train_labels = data["train_labels"]

# Apply subsampling if needed (same as Python version)
TRAINING_SKIP = 100
train_vectors = train_vectors[::TRAINING_SKIP]
train_labels = train_labels[::TRAINING_SKIP]

print(f"[STATUS] Initial training samples: {len(train_vectors)}")

# Filter out any training samples containing NAN values
valid_mask = ~np.isnan(train_vectors).any(axis=1)
train_vectors = train_vectors[valid_mask]
train_labels = train_labels[valid_mask]

print(f"[STATUS] Training samples after NAN filtering: {len(train_vectors)}")

n_train = len(train_vectors)
patch_dim = train_vectors.shape[1]
n_classes = 2

print(f"[STATUS] Features per sample: {patch_dim}")
print(f"[STATUS] Classes: {n_classes}")

print(f"[STATUS] Writing to {bin_file}...")
with open(bin_file, "wb") as f:
    # Write header
    f.write(struct.pack('i', n_train))
    f.write(struct.pack('i', patch_dim))
    f.write(struct.pack('i', n_classes))

    # Write training vectors (float32)
    train_vectors.astype(np.float32).tofile(f)

    # Write labels (uint8)
    train_labels.astype(np.uint8).tofile(f)

print(f"[STATUS] Export complete!")
print(f"[STATUS] File size: {n_train * patch_dim * 4 + n_train + 12} bytes")


