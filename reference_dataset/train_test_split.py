# -*- coding: utf-8 -*-
"""Split and standardise image patches."""

import re
import csv
import time
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm
from osgeo import gdal

INPUT_ROOT_DIR = Path(
    r"D:\phdl2\QTP surface water dynamics\machine learning output\tiles"
)

OUT_DIR = Path(
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step1_machine_learning_output\generalisation\tiles_train_test_val_splitted"
)

STANDARDISATION_PATH = Path(
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step1_machine_learning_output\generalisation\standardisation_configuration"
    r"\tiles_train_test_val_splitted_all_tiles_zscore_configuration.npz"
)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

DRY_RUN = False


CLEAR_OUTPUT_DIR = False

IMAGE_EXT = ".npy"
LABEL_EXT = ".tif"

N_CHANNELS = 17


SAVE_IMAGE_DTYPE = np.float16
SAVE_LABEL_DTYPE = np.uint8
SAVE_MASK_DTYPE = np.uint8


CLIP_ZSCORE = True
ZSCORE_CLIP_VALUE = 5.0

PRINT_EVERY = 500

gdal.UseExceptions()

def infer_class_from_name(base_name):

    name = base_name.lower()

    if "_background_" in name:
        return "background"
    elif "_midhigh_" in name:
        return "midhigh"
    elif "_high_" in name:
        return "high"
    elif "_low_" in name:
        return "low"
    elif "_mid_" in name:
        return "mid"
    else:
        return None


def is_windows_duplicate_copy(base_name):

    name = base_name.strip()
    pattern = re.compile(r".*\s?\(\d+\)$")
    return bool(pattern.match(name))

def make_output_dirs(root_dir):
    for split in ["train", "val", "test"]:
        (root_dir / split / "images_norm").mkdir(parents=True, exist_ok=True)
        (root_dir / split / "labels_npy").mkdir(parents=True, exist_ok=True)
        (root_dir / split / "valid_masks").mkdir(parents=True, exist_ok=True)

def read_tif_as_array(path):
    ds = gdal.Open(str(path))
    if ds is None:
        raise RuntimeError(f"Cannot open TIFF file: {path}")

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    ds = None

    return arr

def build_valid_mask_and_standardise(x, mean, std):

    if x.ndim != 3:
        raise ValueError(f"Invalid image dimensions; expected 3D, got shape={x.shape}")

    if x.shape[0] != N_CHANNELS:
        raise ValueError(
            f"Invalid channel count; expected {N_CHANNELS}, got shape={x.shape}"
        )

    x = x.astype(np.float32, copy=False)

    valid_pixel = np.ones(x.shape[1:], dtype=bool)

    for c in range(N_CHANNELS):
        band = x[c]

        invalid_band = (
            (band == -32768) |
            (band < -1.0e20) |
            (~np.isfinite(band))
        )

        if invalid_band.any():
            valid_pixel[invalid_band] = False
            band[invalid_band] = np.nan

    # z-score
    x = (x - mean) / std

    # clip
    if CLIP_ZSCORE:
        np.clip(
            x,
            -ZSCORE_CLIP_VALUE,
            ZSCORE_CLIP_VALUE,
            out=x
        )


    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
        copy=False
    ).astype(np.float32, copy=False)

    valid_mask = valid_pixel.astype(SAVE_MASK_DTYPE, copy=False)[None, :, :]

    return x, valid_mask

def process_and_save_sample(sample, split, out_dir, mean, std):
    """
    sample:
    {
        region_id,
        month,
        class,
        base_name,
        image_path,
        label_path
    }
    """
    region_id = sample["region_id"]
    month_name = sample["month"]
    base_name = sample["base_name"]

    src_img = sample["image_path"]
    src_lbl = sample["label_path"]


    new_base = f"region_{region_id}__{month_name}__{base_name}"

    out_img_path = out_dir / split / "images_norm" / f"{new_base}.npy"
    out_lbl_path = out_dir / split / "labels_npy" / f"{new_base}.npy"
    out_mask_path = out_dir / split / "valid_masks" / f"{new_base}.npy"

    if DRY_RUN:
        return {
            "out_image": str(out_img_path),
            "out_label": str(out_lbl_path),
            "out_mask": str(out_mask_path),
            "valid_ratio": "",
            "water_ratio_all": "",
            "water_ratio_valid": "",
            "image_dtype": "",
            "label_dtype": "",
            "mask_dtype": "",
        }

    # 1. image
    x = np.load(src_img)
    x_norm, valid_mask = build_valid_mask_and_standardise(x, mean, std)

    # 2. label
    y_raw = read_tif_as_array(src_lbl).astype(np.float32, copy=False)

    if y_raw.ndim != 2:
        raise ValueError(f"Invalid label dimensions: {src_lbl}, shape={y_raw.shape}")


    y = (y_raw == 1).astype(SAVE_LABEL_DTYPE, copy=False)
    y = y[None, :, :]


    if x_norm.shape[1:] != y.shape[1:]:
        raise ValueError(
            f"Image and label dimensions do not match:\n"
            f"image: {src_img}, shape={x_norm.shape}\n"
            f"label: {src_lbl}, shape={y.shape}"
        )

    if valid_mask.shape != y.shape:
        raise ValueError(
            f"Valid mask and label dimensions do not match:\n"
            f"mask: {valid_mask.shape}\n"
            f"label: {y.shape}"
        )


    x_save = x_norm.astype(SAVE_IMAGE_DTYPE, copy=False)

    np.save(out_img_path, x_save)
    np.save(out_lbl_path, y)
    np.save(out_mask_path, valid_mask)

    valid_ratio = float(valid_mask.mean())
    water_ratio_all = float(y.mean())

    if valid_mask.sum() > 0:
        water_ratio_valid = float((y * valid_mask).sum() / valid_mask.sum())
    else:
        water_ratio_valid = 0.0

    return {
        "out_image": str(out_img_path),
        "out_label": str(out_lbl_path),
        "out_mask": str(out_mask_path),
        "valid_ratio": valid_ratio,
        "water_ratio_all": water_ratio_all,
        "water_ratio_valid": water_ratio_valid,
        "image_dtype": str(x_save.dtype),
        "label_dtype": str(y.dtype),
        "mask_dtype": str(valid_mask.dtype),
    }

def main():
    start_time = time.time()

    if not INPUT_ROOT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_ROOT_DIR}")

    if not STANDARDISATION_PATH.exists():
        raise FileNotFoundError(f"Standardisation file does not exist: {STANDARDISATION_PATH}")

    if CLEAR_OUTPUT_DIR and OUT_DIR.exists():
        if DRY_RUN:
            print(f"[DRY_RUN] Would clear output directory: {OUT_DIR}")
        else:
            print(f"[Clearing output directory] {OUT_DIR}")
            shutil.rmtree(OUT_DIR)

    make_output_dirs(OUT_DIR)


    cfg = np.load(STANDARDISATION_PATH, allow_pickle=True)

    mean = cfg["mean"].astype(np.float32).reshape(N_CHANNELS, 1, 1)
    std = cfg["std"].astype(np.float32).reshape(N_CHANNELS, 1, 1)

    std[std == 0] = 1.0

    random.seed(RANDOM_SEED)

    print("=" * 90)
    print("Splitting samples and preparing training data")
    print("=" * 90)
    print("INPUT_ROOT_DIR:", INPUT_ROOT_DIR)
    print("OUT_DIR:", OUT_DIR)
    print("STANDARDISATION_PATH:", STANDARDISATION_PATH)
    print("DRY_RUN:", DRY_RUN)
    print("CLEAR_OUTPUT_DIR:", CLEAR_OUTPUT_DIR)
    print("SAVE_IMAGE_DTYPE:", SAVE_IMAGE_DTYPE)
    print("SAVE_LABEL_DTYPE:", SAVE_LABEL_DTYPE)
    print("SAVE_MASK_DTYPE:", SAVE_MASK_DTYPE)
    print("CLIP_ZSCORE:", CLIP_ZSCORE)
    print("ZSCORE_CLIP_VALUE:", ZSCORE_CLIP_VALUE)


    region_pattern = re.compile(r"^region_(\d+)_tiles_100$")

    region_dirs = {}

    for item in INPUT_ROOT_DIR.iterdir():
        if not item.is_dir():
            continue

        match = region_pattern.match(item.name)
        if match:
            region_id = int(match.group(1))
            region_dirs[region_id] = item

    region_ids = sorted(region_dirs.keys())

    print("\n" + "=" * 90)
    print(f"Number of regions: {len(region_ids)}")
    print("Region IDs:")
    print(region_ids)

    if len(region_ids) == 0:
        raise RuntimeError("No region_xx_tiles_100 folders were found.")


    groups = defaultdict(list)
    duplicate_records = []

    total_image_files = 0
    total_label_files = 0
    total_valid_samples = 0

    duplicate_image_count = 0
    duplicate_label_count = 0
    missing_image_count = 0
    unknown_class_count = 0

    print("\n" + "=" * 90)
    print("Collecting samples")

    for region_id in region_ids:
        region_root = region_dirs[region_id]

        print("\n" + "-" * 90)
        print(f"Processing region: region_{region_id}_tiles_100")
        print(f"Path: {region_root}")

        region_valid_count = 0
        region_dup_img_count = 0
        region_dup_lbl_count = 0

        for month in range(1, 13):
            month_name = f"month{month:02d}"
            month_dir = region_root / month_name

            img_dir = month_dir / "images"
            lbl_dir = month_dir / "labels_tif"

            if not month_dir.exists():
                print(f"[Skip] Month directory does not exist: {month_dir}")
                continue

            if not img_dir.exists():
                print(f"[Skip] Images directory does not exist: {img_dir}")
                continue

            if not lbl_dir.exists():
                print(f"[Skip] Label directory does not exist: {lbl_dir}")
                continue

            image_files = sorted(img_dir.glob(f"*{IMAGE_EXT}"))
            label_files = sorted(lbl_dir.glob(f"*{LABEL_EXT}"))

            total_image_files += len(image_files)
            total_label_files += len(label_files)


            for img_path in image_files:
                base_name = img_path.stem

                if is_windows_duplicate_copy(base_name):
                    duplicate_image_count += 1
                    region_dup_img_count += 1

                    paired_label_path = lbl_dir / f"{base_name}{LABEL_EXT}"

                    duplicate_records.append({
                        "region_id": region_id,
                        "month": month_name,
                        "file_type": "image",
                        "duplicate_path": str(img_path),
                        "paired_path": str(paired_label_path),
                        "paired_exists": paired_label_path.exists(),
                    })

                    print(f"[Skip] Possible duplicate image: {img_path}")

            month_valid_count = 0
            month_class_counter = Counter()


            for lbl_path in label_files:
                base_name = lbl_path.stem

                if is_windows_duplicate_copy(base_name):
                    duplicate_label_count += 1
                    region_dup_lbl_count += 1

                    paired_image_path = img_dir / f"{base_name}{IMAGE_EXT}"

                    duplicate_records.append({
                        "region_id": region_id,
                        "month": month_name,
                        "file_type": "label",
                        "duplicate_path": str(lbl_path),
                        "paired_path": str(paired_image_path),
                        "paired_exists": paired_image_path.exists(),
                    })

                    print(f"[Skip] Possible duplicate label: {lbl_path}")
                    continue

                cls = infer_class_from_name(base_name)

                if cls is None:
                    unknown_class_count += 1
                    print(f"[Warning] Unknown class; skipping: {lbl_path}")
                    continue

                img_path = img_dir / f"{base_name}{IMAGE_EXT}"

                if not img_path.exists():
                    missing_image_count += 1
                    print(f"[Warning] Matching image not found; skipping: {img_path}")
                    continue

                if is_windows_duplicate_copy(img_path.stem):
                    duplicate_image_count += 1
                    region_dup_img_count += 1

                    duplicate_records.append({
                        "region_id": region_id,
                        "month": month_name,
                        "file_type": "image",
                        "duplicate_path": str(img_path),
                        "paired_path": str(lbl_path),
                        "paired_exists": lbl_path.exists(),
                    })

                    print(f"[Skip] Possible duplicate image and paired sample: {img_path}")
                    continue

                sample_info = {
                    "region_id": region_id,
                    "region_folder": str(region_root),
                    "month": month_name,
                    "class": cls,
                    "base_name": base_name,
                    "image_path": img_path,
                    "label_path": lbl_path,
                }

                groups[cls].append(sample_info)

                month_valid_count += 1
                month_class_counter[cls] += 1

            region_valid_count += month_valid_count
            total_valid_samples += month_valid_count

            print(
                f"{month_name}: valid samples={month_valid_count}, "
                f"class counts={dict(month_class_counter)}, "
                f"cumulative duplicate labels={region_dup_lbl_count}, "
                f"cumulative duplicate images={region_dup_img_count}"
            )

        print(f"Region {region_id} valid samples: {region_valid_count}")
        print(f"Region {region_id} possible duplicate labels: {region_dup_lbl_count}")
        print(f"Region {region_id} possible duplicate images: {region_dup_img_count}")


    print("\n" + "=" * 90)
    print("Sample collection complete")
    print(f"Image files found: {total_image_files}")
    print(f"Label files found: {total_label_files}")
    print(f"Valid image-label pairs: {total_valid_samples}")
    print(f"Labels without matching images: {missing_image_count}")
    print(f"Labels with unknown class: {unknown_class_count}")
    print(f"Possible Windows duplicate labels: {duplicate_label_count}")
    print(f"Possible Windows duplicate images: {duplicate_image_count}")
    print(f"Total possible Windows duplicates: {len(duplicate_records)}")

    print("\nCounts by class:")
    for cls in sorted(groups.keys()):
        print(f"{cls}: {len(groups[cls])}")

    if total_valid_samples == 0:
        raise RuntimeError("No valid samples found; check the paths and file names.")


    train_samples = []
    val_samples = []
    test_samples = []

    print("\n" + "=" * 90)
    print("Splitting train / validation / test by class")

    for cls in sorted(groups.keys()):
        sample_list = groups[cls]
        random.shuffle(sample_list)

        n = len(sample_list)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val

        cls_train = sample_list[:n_train]
        cls_val = sample_list[n_train:n_train + n_val]
        cls_test = sample_list[n_train + n_val:]

        train_samples.extend(cls_train)
        val_samples.extend(cls_val)
        test_samples.extend(cls_test)

        print(
            f"{cls}: total={n}, "
            f"train={len(cls_train)}, "
            f"val={len(cls_val)}, "
            f"test={len(cls_test)}"
        )

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    print("\nFinal split:")
    print(f"train: {len(train_samples)}")
    print(f"val  : {len(val_samples)}")
    print(f"test : {len(test_samples)}")
    print(f"total: {len(train_samples) + len(val_samples) + len(test_samples)}")


    manifest_rows = []

    total_processed = 0
    total_failed = 0

    print("\n" + "=" * 90)
    print("Preparing images_norm / labels_npy / valid_masks")

    split_to_samples = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    for split, samples in split_to_samples.items():
        print("\n" + "-" * 90)
        print(f"Processing split: {split}, samples={len(samples)}")

        split_processed = 0
        split_failed = 0

        pbar = tqdm(samples, desc=f"Process {split}")

        for i, sample in enumerate(pbar, start=1):
            try:
                info = process_and_save_sample(
                    sample=sample,
                    split=split,
                    out_dir=OUT_DIR,
                    mean=mean,
                    std=std,
                )

                split_processed += 1
                total_processed += 1

                manifest_rows.append({
                    "split": split,
                    "region_id": sample["region_id"],
                    "month": sample["month"],
                    "class": sample["class"],
                    "base_name": sample["base_name"],
                    "src_image": str(sample["image_path"]),
                    "src_label": str(sample["label_path"]),
                    "out_image": info["out_image"],
                    "out_label": info["out_label"],
                    "out_mask": info["out_mask"],
                    "valid_ratio": info["valid_ratio"],
                    "water_ratio_all": info["water_ratio_all"],
                    "water_ratio_valid": info["water_ratio_valid"],
                    "image_dtype": info["image_dtype"],
                    "label_dtype": info["label_dtype"],
                    "mask_dtype": info["mask_dtype"],
                    "status": "processed" if not DRY_RUN else "dry_run",
                    "error": "",
                })

                if i % PRINT_EVERY == 0:
                    pbar.set_postfix({
                        "processed": split_processed,
                        "valid": info["valid_ratio"],
                        "water_valid": info["water_ratio_valid"],
                    })

            except Exception as e:
                split_failed += 1
                total_failed += 1

                manifest_rows.append({
                    "split": split,
                    "region_id": sample["region_id"],
                    "month": sample["month"],
                    "class": sample["class"],
                    "base_name": sample["base_name"],
                    "src_image": str(sample["image_path"]),
                    "src_label": str(sample["label_path"]),
                    "out_image": "",
                    "out_label": "",
                    "out_mask": "",
                    "valid_ratio": "",
                    "water_ratio_all": "",
                    "water_ratio_valid": "",
                    "image_dtype": "",
                    "label_dtype": "",
                    "mask_dtype": "",
                    "status": "failed",
                    "error": repr(e),
                })

                print(f"\n[Error] Processing failed: {sample['image_path']}")
                print(repr(e))

        print(f"[{split}] processed: {split_processed}")
        print(f"[{split}] failed   : {split_failed}")


    manifest_path = OUT_DIR / "split_preprocess_manifest.csv"
    duplicate_csv_path = OUT_DIR / "duplicate_copy_files_check.csv"

    manifest_fields = [
        "split",
        "region_id",
        "month",
        "class",
        "base_name",
        "src_image",
        "src_label",
        "out_image",
        "out_label",
        "out_mask",
        "valid_ratio",
        "water_ratio_all",
        "water_ratio_valid",
        "image_dtype",
        "label_dtype",
        "mask_dtype",
        "status",
        "error",
    ]

    duplicate_fields = [
        "region_id",
        "month",
        "file_type",
        "duplicate_path",
        "paired_path",
        "paired_exists",
    ]

    if not DRY_RUN:
        with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=manifest_fields)
            writer.writeheader()
            writer.writerows(manifest_rows)

        with open(duplicate_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=duplicate_fields)
            writer.writeheader()
            writer.writerows(duplicate_records)

    elapsed = time.time() - start_time

    print("\n" + "=" * 90)
    print("Complete")
    print("=" * 90)
    print("OUT_DIR:", OUT_DIR)
    print("manifest:", manifest_path)
    print("duplicate csv:", duplicate_csv_path)
    print(f"total_valid_samples: {total_valid_samples}")
    print(f"total_processed    : {total_processed}")
    print(f"total_failed       : {total_failed}")
    print(f"elapsed seconds    : {elapsed:.1f}")
    print(f"elapsed minutes    : {elapsed / 60:.1f}")

    if DRY_RUN:
        print("\nDRY_RUN=True; no output files written.")
    else:
        print("\nOutput:")
        print("  train/images_norm, train/labels_npy, train/valid_masks")
        print("  val/images_norm, val/labels_npy, val/valid_masks")
        print("  test/images_norm, test/labels_npy, test/valid_masks")

if __name__ == "__main__":
    main()