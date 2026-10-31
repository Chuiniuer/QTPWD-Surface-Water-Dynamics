# -*- coding: utf-8 -*-
"""Extract 512-pixel image patches and candidate labels."""

import os
import random
import numpy as np
from osgeo import gdal, ogr
from tqdm import tqdm

region_id = 79
year = 2021

mask_root = rf"D:\phdl2\QTP surface water dynamics\machine learning output\merged_original\{region_id}"

region_cfg = {
        "region_id": 79,
        "bands_folder": r"J:\sentinel2_toa_79",
        "indices_folder": r"O:\QTP_output\79_water_indices_merged",
        "env_folder": r"O:\QTP_output\terrain_features_79_resampled",
        "texture_folder": r"O:\QTP_output\79_textures_merged",
    }

root_out_folder = r"K:\Monthly surface water dynamics TIB\supervised classification\step2_semantic_segmentation\inference\2021_inference_validation"
out_folder = os.path.join(root_out_folder, f"region_{region_id}_tiles_100")
os.makedirs(out_folder, exist_ok=True)

# shapefile
region_shp_path = r"D:\phdl2\QTP surface water dynamics\regions\QTP.shp"


use_shapefile_mode = "auto"


require_patch_fully_inside_shp = True


min_inside_ratio = 0.95

chip_size = 512
random_seed = 43
months = range(1, 13)


n_background = 5
n_low_water = 10
n_mid_high_water = 10


low_thresh = 0.15


candidate_num = 20000


max_feature_invalid_ratio = 0.05
max_label_invalid_ratio = 0.05

random.seed(random_seed)
np.random.seed(random_seed)

def write_tif(out_path, array, gt, proj, dtype=gdal.GDT_Byte):
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        out_path,
        array.shape[1],
        array.shape[0],
        1,
        dtype,
        options=["COMPRESS=LZW"]
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    band = ds.GetRasterBand(1)
    band.WriteArray(array)
    band.FlushCache()
    band = None
    ds = None


def calc_new_gt(gt, xoff, yoff):
    return (
        gt[0] + gt[1] * xoff + gt[2] * yoff,
        gt[1],
        gt[2],
        gt[3] + gt[4] * xoff + gt[5] * yoff,
        gt[4],
        gt[5]
    )


def get_feature_paths(region_id, month, cfg):
    bands_folder = cfg["bands_folder"]
    indices_folder = cfg["indices_folder"]
    env_folder = cfg["env_folder"]
    texture_folder = cfg["texture_folder"]

    return [
        os.path.join(bands_folder,   f"{region_id}_sentinel2_{year}{month:02d}_B2.tif"),
        os.path.join(bands_folder,   f"{region_id}_sentinel2_{year}{month:02d}_B3.tif"),
        os.path.join(bands_folder,   f"{region_id}_sentinel2_{year}{month:02d}_B4.tif"),
        os.path.join(bands_folder,   f"{region_id}_sentinel2_{year}{month:02d}_B8.tif"),
        os.path.join(bands_folder,   f"{region_id}_sentinel2_{year}{month:02d}_B11.tif"),
        os.path.join(bands_folder,   f"{region_id}_sentinel2_{year}{month:02d}_B12.tif"),

        os.path.join(indices_folder, f"{region_id}_sentinel2_{year}{month:02d}_AWEI.tif"),
        os.path.join(indices_folder, f"{region_id}_sentinel2_{year}{month:02d}_MBWI.tif"),
        os.path.join(indices_folder, f"{region_id}_sentinel2_{year}{month:02d}_NDWI.tif"),
        os.path.join(indices_folder, f"{region_id}_sentinel2_{year}{month:02d}_MNDWI.tif"),

        os.path.join(env_folder,     f"{region_id}_slope.tif"),
        os.path.join(env_folder,     f"{region_id}_elevation.tif"),
        os.path.join(env_folder,     f"{region_id}_wetness.tif"),

        os.path.join(texture_folder, f"ASM_{region_id}_sentinel2_{year}_{month:02d}_pc1.tif"),
        os.path.join(texture_folder, f"contrast_{region_id}_sentinel2_{year}_{month:02d}_pc1.tif"),
        os.path.join(texture_folder, f"dissimilarity_{region_id}_sentinel2_{year}_{month:02d}_pc1.tif"),
        os.path.join(texture_folder, f"entropy_{region_id}_sentinel2_{year}_{month:02d}_pc1.tif"),
    ]


def open_feature_datasets(region_id, month, cfg):
    paths = get_feature_paths(region_id, month, cfg)
    datasets = []
    for p in paths:
        ds = gdal.Open(p)
        if ds is None:
            raise FileNotFoundError(f"Could not open feature file: {p}")
        datasets.append(ds)
    return datasets


def close_feature_datasets(datasets):
    for i in range(len(datasets)):
        datasets[i] = None


def invalid_mask_from_array(arr):
    return np.isnan(arr) | np.isinf(arr) | (arr == -32768)


def build_chip_from_open_datasets_allow_invalid(feature_datasets, xoff, yoff, chip_size):

    bands = []
    invalid_pixels_total = 0
    total_values = 0

    for ds in feature_datasets:
        arr = ds.ReadAsArray(xoff=xoff, yoff=yoff, xsize=chip_size, ysize=chip_size)
        if arr is None or arr.shape != (chip_size, chip_size):
            return None, 1.0, False

        arr = arr.astype(np.float32)
        invalid_mask = invalid_mask_from_array(arr)

        invalid_pixels_total += int(invalid_mask.sum())
        total_values += arr.size

        if invalid_mask.any():
            valid_vals = arr[~invalid_mask]
            fill_value = float(np.median(valid_vals)) if valid_vals.size > 0 else 0.0
            arr[invalid_mask] = fill_value

        bands.append(arr)

    x = np.stack(bands, axis=0)
    feature_invalid_ratio = invalid_pixels_total / total_values
    return x, feature_invalid_ratio, True


def calc_label_invalid_ratio(label):
    return float((label == 2).sum()) / label.size


def calc_water_ratio_ignore_invalid(label):
    valid_mask = (label == 0) | (label == 1)
    valid_count = int(valid_mask.sum())

    if valid_count == 0:
        return None

    water_ratio = float((label[valid_mask] == 1).sum()) / valid_count
    return water_ratio


def classify_patch(water_ratio, low_thresh=0.15):
    if water_ratio == 0:
        return "background"
    elif water_ratio <= low_thresh:
        return "low"
    else:
        return "midhigh"


def save_patch(region_id, month, patch_type, serial_id, xoff, yoff, x, label, gt, proj,
               img_dir, lbl_tif_dir):
    base_name = f"serial{serial_id:04d}_r{region_id}_m{month:02d}_{patch_type}_{xoff}_{yoff}"

    np.save(os.path.join(img_dir, base_name + ".npy"), x)

    new_gt = calc_new_gt(gt, xoff, yoff)
    write_tif(
        os.path.join(lbl_tif_dir, base_name + ".tif"),
        label,
        new_gt,
        proj,
        dtype=gdal.GDT_Byte
    )


def generate_unique_random_candidates(xsize, ysize, chip_size, candidate_num, seed=None):
    rng = np.random.default_rng(seed)

    max_x = xsize - chip_size
    max_y = ysize - chip_size

    candidate_set = set()

    while len(candidate_set) < candidate_num:
        need = candidate_num - len(candidate_set)
        batch_n = max(need * 2, 3000)

        xoffs = rng.integers(0, max_x + 1, size=batch_n)
        yoffs = rng.integers(0, max_y + 1, size=batch_n)

        for xoff, yoff in zip(xoffs, yoffs):
            candidate_set.add((int(xoff), int(yoff)))
            if len(candidate_set) >= candidate_num:
                break

    candidate_list = list(candidate_set)
    rng.shuffle(candidate_list)
    return candidate_list


def rasterize_shapefile_to_mask(shp_path, xsize, ysize, gt, proj_wkt):

    mem_driver = gdal.GetDriverByName("MEM")
    mask_ds = mem_driver.Create("", xsize, ysize, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(gt)
    mask_ds.SetProjection(proj_wkt)

    band = mask_ds.GetRasterBand(1)
    band.Fill(0)

    shp_ds = ogr.Open(shp_path)
    if shp_ds is None:
        raise FileNotFoundError(f"Could not open shapefile: {shp_path}")

    layer = shp_ds.GetLayer()
    gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1])

    region_mask = band.ReadAsArray().astype(np.uint8)

    band = None
    mask_ds = None
    shp_ds = None

    return region_mask


def decide_use_region_mask(use_mode, region_mask):

    if use_mode == "always":
        return True
    elif use_mode == "never":
        return False
    elif use_mode == "auto":


        return np.any(region_mask == 0)
    else:
        raise ValueError('use_shapefile_mode must be "auto", "always" or "never"')


def patch_inside_region(region_mask, xoff, yoff, chip_size,
                        require_full=True, min_ratio=0.95):
    region_patch = region_mask[yoff:yoff + chip_size, xoff:xoff + chip_size]
    if region_patch.shape != (chip_size, chip_size):
        return False

    if require_full:
        return np.all(region_patch == 1)
    else:
        inside_ratio = float(region_patch.mean())
        return inside_ratio >= min_ratio


def fill_from_candidates(candidate_list, target_num, occupied, chip_size, feature_datasets,
                         mask, region_mask, use_region_mask,
                         region_id, month, gt, proj,
                         img_dir, lbl_tif_dir,
                         serial_start, patch_type):
    saved_count = 0
    invalid_feature_count = 0
    invalid_label_count = 0
    tried_count = 0
    serial_id = serial_start

    for xoff, yoff in candidate_list:
        if saved_count >= target_num:
            break

        tried_count += 1


        if occupied[yoff:yoff + chip_size, xoff:xoff + chip_size].any():
            continue


        if use_region_mask:
            ok = patch_inside_region(
                region_mask,
                xoff, yoff, chip_size,
                require_full=require_patch_fully_inside_shp,
                min_ratio=min_inside_ratio
            )
            if not ok:
                continue


        x, feature_invalid_ratio, valid = build_chip_from_open_datasets_allow_invalid(
            feature_datasets, xoff, yoff, chip_size
        )
        if (not valid) or (x is None) or (feature_invalid_ratio > max_feature_invalid_ratio):
            invalid_feature_count += 1
            continue


        label = mask[yoff:yoff + chip_size, xoff:xoff + chip_size]
        if label.shape != (chip_size, chip_size):
            continue

        label_invalid_ratio = calc_label_invalid_ratio(label)
        if label_invalid_ratio > max_label_invalid_ratio:
            invalid_label_count += 1
            continue

        occupied[yoff:yoff + chip_size, xoff:xoff + chip_size] = True

        save_patch(
            region_id, month, patch_type, serial_id, xoff, yoff, x, label, gt, proj,
            img_dir, lbl_tif_dir
        )

        saved_count += 1
        serial_id += 1

    return saved_count, invalid_feature_count, invalid_label_count, tried_count, serial_id

print("=" * 80)
print(f"Processing region_{region_id}, January–December {year}")
print(f"use_shapefile_mode = {use_shapefile_mode}")
print("=" * 80)

for month in months:
    print("\n" + "=" * 80)
    print(f"Processing month {month:02d}")
    print("=" * 80)

    mask_path = os.path.join(mask_root, f"mask_{region_id}_{year}_{month:02d}.tif")
    if not os.path.exists(mask_path):
        print(f"[SKIP] Mask not found: {mask_path}")
        continue

    month_dir = os.path.join(out_folder, f"month{month:02d}")
    img_dir = os.path.join(month_dir, "images")
    lbl_tif_dir = os.path.join(month_dir, "labels_tif")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_tif_dir, exist_ok=True)

    ds_mask = gdal.Open(mask_path)
    if ds_mask is None:
        print(f"[SKIP] Could not open mask: {mask_path}")
        continue

    xsize = ds_mask.RasterXSize
    ysize = ds_mask.RasterYSize
    gt = ds_mask.GetGeoTransform()
    proj = ds_mask.GetProjection()

    if xsize < chip_size or ysize < chip_size:
        print(f"[SKIP] Month {month:02d}: image is smaller than chip_size")
        ds_mask = None
        continue

    mask = ds_mask.ReadAsArray().astype(np.uint8)
    ds_mask = None


    region_mask = rasterize_shapefile_to_mask(region_shp_path, xsize, ysize, gt, proj)
    use_region_mask = decide_use_region_mask(use_shapefile_mode, region_mask)

    print(f"month {month:02d}: use_region_mask = {use_region_mask}")

    try:
        feature_datasets = open_feature_datasets(region_id, month, region_cfg)
    except Exception as e:
        print(f"[SKIP] Month {month:02d}: could not open features: {e}")
        continue

    print("Generating unique random candidates...")
    candidates = generate_unique_random_candidates(
        xsize=xsize,
        ysize=ysize,
        chip_size=chip_size,
        candidate_num=candidate_num,
        seed=random_seed + region_id * 100 + month
    )

    candidates_bg = []
    candidates_low = []
    candidates_midhigh = []

    print("Calculating water fractions and classifying candidate patches...")
    for xoff, yoff in tqdm(candidates, total=len(candidates), desc=f"r{region_id}-m{month:02d}-classify"):
        label = mask[yoff:yoff + chip_size, xoff:xoff + chip_size]
        if label.shape != (chip_size, chip_size):
            continue

        label_invalid_ratio = calc_label_invalid_ratio(label)
        if label_invalid_ratio > max_label_invalid_ratio:
            continue

        water_ratio = calc_water_ratio_ignore_invalid(label)
        if water_ratio is None:
            continue

        patch_type = classify_patch(water_ratio, low_thresh=low_thresh)

        if patch_type == "background":
            candidates_bg.append((xoff, yoff))
        elif patch_type == "low":
            candidates_low.append((xoff, yoff))
        else:
            candidates_midhigh.append((xoff, yoff))

    print(f"Background candidates: {len(candidates_bg)}")
    print(f"Low-water candidates: {len(candidates_low)}")
    print(f"Medium/high-water candidates: {len(candidates_midhigh)}")

    random.shuffle(candidates_bg)
    random.shuffle(candidates_low)
    random.shuffle(candidates_midhigh)

    occupied = np.zeros((ysize, xsize), dtype=bool)
    serial_id = 1

    print("Selecting and saving patches from the candidate pool...")

    count_bg, inv_feat_bg, inv_lbl_bg, tried_bg, serial_id = fill_from_candidates(
        candidates_bg, n_background, occupied, chip_size, feature_datasets,
        mask, region_mask, use_region_mask,
        region_id, month, gt, proj,
        img_dir, lbl_tif_dir,
        serial_start=serial_id, patch_type="background"
    )

    count_low, inv_feat_low, inv_lbl_low, tried_low, serial_id = fill_from_candidates(
        candidates_low, n_low_water, occupied, chip_size, feature_datasets,
        mask, region_mask, use_region_mask,
        region_id, month, gt, proj,
        img_dir, lbl_tif_dir,
        serial_start=serial_id, patch_type="low"
    )

    count_midhigh, inv_feat_midhigh, inv_lbl_midhigh, tried_midhigh, serial_id = fill_from_candidates(
        candidates_midhigh, n_mid_high_water, occupied, chip_size, feature_datasets,
        mask, region_mask, use_region_mask,
        region_id, month, gt, proj,
        img_dir, lbl_tif_dir,
        serial_start=serial_id, patch_type="midhigh"
    )

    close_feature_datasets(feature_datasets)

    saved_total = count_bg + count_low + count_midhigh
    inv_feat_total = inv_feat_bg + inv_feat_low + inv_feat_midhigh
    inv_lbl_total = inv_lbl_bg + inv_lbl_low + inv_lbl_midhigh
    tried_total = tried_bg + tried_low + tried_midhigh

    print("\n" + "=" * 80)
    print(f"Completed region_{region_id}, month {month:02d}")
    print(f"Output directory: {month_dir}")
    print(f"background saved : {count_bg}/{n_background} (tried={tried_bg}, feature_invalid={inv_feat_bg}, label_invalid={inv_lbl_bg})")
    print(f"low saved        : {count_low}/{n_low_water} (tried={tried_low}, feature_invalid={inv_feat_low}, label_invalid={inv_lbl_low})")
    print(f"midhigh saved    : {count_midhigh}/{n_mid_high_water} (tried={tried_midhigh}, feature_invalid={inv_feat_midhigh}, label_invalid={inv_lbl_midhigh})")
    print(f"feature invalid dropped : {inv_feat_total}")
    print(f"label invalid dropped   : {inv_lbl_total}")
    print(f"total tried             : {tried_total}")
    print(f"total saved             : {saved_total}")
    print("=" * 80)

print("\nAll months completed.")
