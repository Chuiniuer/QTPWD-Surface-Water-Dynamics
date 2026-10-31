# -*- coding: utf-8 -*-
"""Calculate channel-wise z-score parameters from image patches."""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

root_dir = (
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step1_machine_learning_output\generalisation\tiles_train_test_val_splitted"
)

image_dirs = [
    os.path.join(root_dir, "train", "images"),
    os.path.join(root_dir, "val", "images"),
    os.path.join(root_dir, "test", "images"),
]

out_dir = (
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step1_machine_learning_output\generalisation\standardisation_configuration"
)

os.makedirs(out_dir, exist_ok=True)

feature_names = [
    "b2", "b3", "b4", "b8", "b11", "b12",
    "AWEI", "MBWI", "NDWI", "MNDWI",
    "slope", "dem", "wetness",
    "asm", "contrast", "dissimilarity", "entropy"
]

n_channels = len(feature_names)

all_files = []

for d in image_dirs:
    if not os.path.exists(d):
        print(f"[Warning] Images folder not found; skipping: {d}")
        continue

    files = [
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.lower().endswith(".npy")
    ]

    print(f"{d}")
    print(f"  Number of .npy files: {len(files)}")

    all_files.extend(files)

if len(all_files) == 0:
    raise ValueError("No .npy files found. Check root_dir and the images paths.")

print("\n" + "=" * 80)
print(f"Total samples found: {len(all_files)}")

sum_c = np.zeros(n_channels, dtype=np.float64)
sumsq_c = np.zeros(n_channels, dtype=np.float64)
count_c = np.zeros(n_channels, dtype=np.int64)


min_c = np.full(n_channels, np.inf, dtype=np.float64)
max_c = np.full(n_channels, -np.inf, dtype=np.float64)

bad_shape_files = []
empty_channel_records = []

for path in tqdm(all_files, desc="Calculating z-score parameters"):
    x = np.load(path)


    if x.ndim != 3:
        bad_shape_files.append((path, x.shape))
        continue

    if x.shape[0] != n_channels:
        bad_shape_files.append((path, x.shape))
        continue

    x = x.astype(np.float64)


    x[x == -32768] = np.nan
    x[x < -1.0e20] = np.nan

    # reshape: (17, H, W) -> (17, N)
    x = x.reshape(n_channels, -1)

    valid_mask = np.isfinite(x)

    for c in range(n_channels):
        vals = x[c, valid_mask[c]]

        if vals.size == 0:
            empty_channel_records.append((path, feature_names[c]))
            continue

        sum_c[c] += vals.sum()
        sumsq_c[c] += np.square(vals).sum()
        count_c[c] += vals.size

        vmin = vals.min()
        vmax = vals.max()

        if vmin < min_c[c]:
            min_c[c] = vmin
        if vmax > max_c[c]:
            max_c[c] = vmax

if len(bad_shape_files) > 0:
    print("\n" + "=" * 80)
    print("[Error] The following files have unexpected shapes and were skipped:")
    for p, shp in bad_shape_files[:50]:
        print(f"{p}, shape={shp}")

    if len(bad_shape_files) > 50:
        print(f"... {len(bad_shape_files) - 50} additional files not shown")

    raise ValueError(
        f"Found {len(bad_shape_files)} files with unexpected shapes. "
        f"Expected shape: ({n_channels}, H, W)."
    )

if np.any(count_c == 0):
    zero_channels = [
        feature_names[i]
        for i in range(n_channels)
        if count_c[i] == 0
    ]

    raise ValueError(
        f"Cannot calculate standardisation parameters; channels have no valid values: {zero_channels}"
    )

mean_c = sum_c / count_c
var_c = sumsq_c / count_c - np.square(mean_c)


var_c = np.maximum(var_c, 1e-12)

std_c = np.sqrt(var_c)

prefix = "tiles_train_test_val_splitted_all_tiles_zscore_configuration"

df = pd.DataFrame({
    "feature": feature_names,
    "mu": mean_c,
    "sigma": std_c,
    "count": count_c,
    "min": min_c,
    "max": max_c,
})

csv_path = os.path.join(out_dir, f"{prefix}.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

npz_path = os.path.join(out_dir, f"{prefix}.npz")

np.savez(
    npz_path,
    feature_names=np.array(feature_names),
    mean=mean_c,
    std=std_c,
    count=count_c,
    min=min_c,
    max=max_c,
)

txt_path = os.path.join(out_dir, f"{prefix}.txt")

with open(txt_path, "w", encoding="utf-8") as f:
    f.write("Z-score standardisation configuration\n")
    f.write("Computed from: train + val + test images\n")
    f.write(f"Input root: {root_dir}\n")
    f.write(f"Number of samples: {len(all_files)}\n\n")

    f.write("feature\tmu\tsigma\tcount\tmin\tmax\n")

    for name, mu, sigma, cnt, vmin, vmax in zip(
        feature_names, mean_c, std_c, count_c, min_c, max_c
    ):
        f.write(
            f"{name}\t"
            f"mu={mu}\t"
            f"sigma={sigma}\t"
            f"count={cnt}\t"
            f"min={vmin}\t"
            f"max={vmax}\n"
        )

empty_channel_csv = os.path.join(out_dir, f"{prefix}_empty_channel_records.csv")

if len(empty_channel_records) > 0:
    empty_df = pd.DataFrame(
        empty_channel_records,
        columns=["file", "empty_channel"]
    )
    empty_df.to_csv(empty_channel_csv, index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("Standardisation parameters calculated")

print("\nCSV:")
print(csv_path)

print("\nNPZ:")
print(npz_path)

print("\nTXT:")
print(txt_path)

if len(empty_channel_records) > 0:
    print("\nSome files have channels without valid values; records saved to:")
    print(empty_channel_csv)

print("\nResults:")
print(df)
