"""Map initial monthly water classes using the point-trained random forest.

Outputs are tiled GeoTIFFs: 0 for non-water, 1 for water and 2 for invalid.
"""

from osgeo import gdal
import numpy as np
import os
import json
import joblib
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

gdal.UseExceptions()


model_path = (
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step1_machine_learning_output\feature matrix\final_rf_unweighted_all_samples"
    r"\rf_unweighted_all_samples.pkl"
)


base_feature_dir = (
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step1_machine_learning_output\feature matrix"
)
median_json_path = os.path.join(base_feature_dir, "feature_impute_medians.json")
all_samples_csv = os.path.join(base_feature_dir, "all_samples.csv")


output_root = (
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step1_machine_learning_output\generalisation\mapping_result_all_regions\original_tiles"
)
os.makedirs(output_root, exist_ok=True)


region_configs = [
    {
        "region_id": 62,
        "bands_folder": r"M:\sentinel2_toa_62",
        "indices_folder": r"N:\QTP_output\62_water_indices_merged",
        "env_folder": r"N:\QTP_output\terrain_features_62_resampled",
        "texture_folder": r"N:\QTP_output\62_textures_merged",
    },
    {
        "region_id": 63,
        "bands_folder": r"M:\sentinel2_toa_63",
        "indices_folder": r"O:\QTP_output\63_water_indices_merged",
        "env_folder": r"O:\QTP_output\terrain_features_63_resampled",
        "texture_folder": r"O:\QTP_output\63_textures_merged",
    },
    {
        "region_id": 79,
        "bands_folder": r"J:\sentinel2_toa_79",
        "indices_folder": r"O:\QTP_output\79_water_indices_merged",
        "env_folder": r"O:\QTP_output\terrain_features_79_resampled",
        "texture_folder": r"O:\QTP_output\79_textures_merged",
    },
]


features = [
    "b2", "b3", "b4", "b8", "b11", "b12",
    "AWEI", "MBWI", "NDWI", "MNDWI",
    "slope", "dem", "wetness",
    "asm", "contrast", "dissimilar", "entropy"
]


year = 2022
months = range(1, 13)


blockx = 10
blocky = 10


max_workers_process = 2


rf_n_jobs = 5


max_missing_features = 5

class_nonwater = 0
class_water = 1
class_invalid = 2


verbose_block = False


def log(msg, log_file=None, also_print=True):
    if also_print:
        print(msg)
    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")


def write_image(out_path, array, xsize, ysize, gt, proj):
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        out_path,
        xsize,
        ysize,
        1,
        gdal.GDT_Byte,
        options=["COMPRESS=LZW"]
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)

    band = out_ds.GetRasterBand(1)
    band.WriteArray(array)
    band.FlushCache()

    band = None
    out_ds = None


def clean_array(arr):
    arr = arr.astype(np.float32)
    arr[arr == -32768] = np.nan
    arr[arr == -9999] = np.nan
    arr[arr == -999] = np.nan
    arr[np.isinf(arr)] = np.nan
    return arr


def fill_nan_edge_by_neighbor(arr):
    arr = arr.copy()
    if arr.ndim != 2 or arr.shape[1] < 2:
        return arr

    if np.all(np.isnan(arr[:, 0])):
        arr[:, 0] = arr[:, 1]
    if np.all(np.isnan(arr[:, -1])):
        arr[:, -1] = arr[:, -2]
    return arr


def open_ds_or_raise(path):
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Unable to open file: {path}")
    return ds


def read_block_from_ds(ds, xoff, yoff, xsize, ysize):
    arr = ds.ReadAsArray(
        xoff=xoff,
        yoff=yoff,
        xsize=xsize,
        ysize=ysize
    )
    if arr is None:
        raise RuntimeError("ReadAsArray returned None")
    return arr


def build_medians_from_csv(csv_path, feature_names):
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=feature_names)
    df = df.replace([-32768, np.inf, -np.inf], np.nan)

    medians = {}
    for col in feature_names:
        med = df[col].median()
        if np.isnan(med):
            raise ValueError(f"Feature {col} has a NaN median")
        medians[col] = float(med)
    return medians


def load_or_build_feature_medians():
    if os.path.exists(median_json_path):
        with open(median_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if not os.path.exists(all_samples_csv):
        raise FileNotFoundError(f"Cannot find all_samples.csv: {all_samples_csv}")

    medians = build_medians_from_csv(all_samples_csv, features)

    with open(median_json_path, "w", encoding="utf-8") as f:
        json.dump(medians, f, ensure_ascii=False, indent=2)

    return medians


def get_month_paths(cfg, month):
    rid = cfg["region_id"]
    bands = cfg["bands_folder"]
    inds = cfg["indices_folder"]
    env = cfg["env_folder"]
    tex = cfg["texture_folder"]

    return {
        "b2": os.path.join(bands, f"{rid}_sentinel2_{year}{month:02d}_B2.tif"),
        "b3": os.path.join(bands, f"{rid}_sentinel2_{year}{month:02d}_B3.tif"),
        "b4": os.path.join(bands, f"{rid}_sentinel2_{year}{month:02d}_B4.tif"),
        "b8": os.path.join(bands, f"{rid}_sentinel2_{year}{month:02d}_B8.tif"),
        "b11": os.path.join(bands, f"{rid}_sentinel2_{year}{month:02d}_B11.tif"),
        "b12": os.path.join(bands, f"{rid}_sentinel2_{year}{month:02d}_B12.tif"),

        "AWEI": os.path.join(inds, f"{rid}_sentinel2_{year}{month:02d}_AWEI.tif"),
        "MBWI": os.path.join(inds, f"{rid}_sentinel2_{year}{month:02d}_MBWI.tif"),
        "NDWI": os.path.join(inds, f"{rid}_sentinel2_{year}{month:02d}_NDWI.tif"),
        "MNDWI": os.path.join(inds, f"{rid}_sentinel2_{year}{month:02d}_MNDWI.tif"),

        "slope": os.path.join(env, f"{rid}_slope.tif"),
        "dem": os.path.join(env, f"{rid}_elevation.tif"),
        "wetness": os.path.join(env, f"{rid}_wetness.tif"),

        "asm": os.path.join(tex, f"ASM_{rid}_sentinel2_{year}_{month:02d}_pc1.tif"),
        "contrast": os.path.join(tex, f"contrast_{rid}_sentinel2_{year}_{month:02d}_pc1.tif"),
        "dissimilar": os.path.join(tex, f"dissimilarity_{rid}_sentinel2_{year}_{month:02d}_pc1.tif"),
        "entropy": os.path.join(tex, f"entropy_{rid}_sentinel2_{year}_{month:02d}_pc1.tif"),
    }


def should_skip_output(out_path):
    if not os.path.exists(out_path):
        return False
    try:
        if os.path.getsize(out_path) < 1024:
            return False
        ds = gdal.Open(out_path)
        if ds is None:
            return False
        arr = ds.ReadAsArray(0, 0, 1, 1)
        ds = None
        if arr is None:
            return False
    except Exception:
        return False
    return True


def predict_block_full(block_arrays, model, feature_medians):
    h, w = next(iter(block_arrays.values())).shape

    cube = np.stack([block_arrays[f] for f in features], axis=2).astype(np.float32)  
    flat = cube.reshape(-1, len(features))  

    nan_mask = np.isnan(flat)
    nan_count = nan_mask.sum(axis=1)

    out = np.full(flat.shape[0], class_invalid, dtype=np.uint8)

    valid_mask = nan_count <= max_missing_features
    if np.any(valid_mask):
        valid_idx = np.where(valid_mask)[0]
        valid_data = flat[valid_mask].copy()

        median_vec = np.array([feature_medians[f] for f in features], dtype=np.float32)
        col_nan = np.isnan(valid_data)
        if np.any(col_nan):
            valid_data[col_nan] = np.take(median_vec, np.where(col_nan)[1])

        pred = model.predict(valid_data)
        out[valid_idx] = pred.astype(np.uint8)

    out = out.reshape(h, w)

    total_pixels = flat.shape[0]
    full_pixels = int((nan_count == 0).sum())
    filled_pixels = int(((nan_count > 0) & (nan_count <= max_missing_features)).sum())
    invalid_pixels = int((nan_count > max_missing_features).sum())

    return out, total_pixels, full_pixels, filled_pixels, invalid_pixels


def process_region_month(task):
    cfg = task["cfg"]
    month = task["month"]
    feature_medians = task["feature_medians"]

    rid = cfg["region_id"]
    region_out = os.path.join(output_root, str(rid))
    os.makedirs(region_out, exist_ok=True)

    log_file = os.path.join(region_out, f"mapping_{rid}_month{month:02d}_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Start mapping region {rid}, month {month:02d}\n")

    try:
        model = joblib.load(model_path)
        if hasattr(model, "n_jobs"):
            model.n_jobs = rf_n_jobs

        ref_path = os.path.join(cfg["bands_folder"], f"{rid}_sentinel2_{year}01_B2.tif")
        ref_ds = open_ds_or_raise(ref_path)
        xsize = ref_ds.RasterXSize
        ysize = ref_ds.RasterYSize
        gt = ref_ds.GetGeoTransform()
        proj = ref_ds.GetProjection()
        ref_ds = None

        bxsize = int(np.ceil(xsize / blockx))
        bysize = int(np.ceil(ysize / blocky))

        month_paths = get_month_paths(cfg, month)
        missing_files = [k for k, v in month_paths.items() if not os.path.exists(v)]
        if missing_files:
            log(f"[SKIP] region={rid}, month={month:02d} missing files: {missing_files}", log_file)
            return (rid, month, "missing_files")

        ds_dict = {k: open_ds_or_raise(v) for k, v in month_paths.items()}

        existing_count = 0
        for l in range(blocky):
            for c in range(blockx):
                out_path = os.path.join(region_out, f"water_{year}_{month:02d}_{c}_{l}.tif")
                if should_skip_output(out_path):
                    existing_count += 1

        log(f"region={rid}, month={month:02d}, existing blocks: {existing_count}/{blockx * blocky}", log_file)

        month_start = time.time()

        for l in range(blocky):
            for c in range(blockx):
                out_path = os.path.join(region_out, f"water_{year}_{month:02d}_{c}_{l}.tif")

                if should_skip_output(out_path):
                    if verbose_block:
                        log(f"[SKIP] existing valid output: {out_path}", log_file, also_print=False)
                    continue

                xoff = bxsize * c
                yoff = bysize * l

                out_xsize = min(bxsize, xsize - xoff)
                out_ysize = min(bysize, ysize - yoff)

                if out_xsize <= 0 or out_ysize <= 0:
                    continue

                try:
                    block_arrays = {}

                    for feat in features:
                        arr = read_block_from_ds(ds_dict[feat], xoff, yoff, out_xsize, out_ysize)
                        arr = clean_array(arr)
                        if feat in ["slope", "wetness"]:
                            arr = fill_nan_edge_by_neighbor(arr)
                        block_arrays[feat] = arr

                    out_block, total_pixels, full_pixels, filled_pixels, invalid_pixels = predict_block_full(
                        block_arrays, model, feature_medians
                    )

                    new_gt = (
                        gt[0] + gt[1] * xoff,
                        gt[1],
                        0.0,
                        gt[3] + gt[5] * yoff,
                        0.0,
                        gt[5]
                    )

                    write_image(out_path, out_block, out_block.shape[1], out_block.shape[0], new_gt, proj)

                    if verbose_block:
                        log(
                            f"month={month:02d}, block=({c},{l}), "
                            f"pixels={total_pixels}, complete={full_pixels}, "
                            f"imputed={filled_pixels}, invalid={invalid_pixels}",
                            log_file,
                            also_print=False
                        )

                    del block_arrays, out_block

                except Exception as e:
                    log(f"[FAILED] region={rid}, month={month:02d}, block=({c},{l}) -> {repr(e)}", log_file)

        for k in ds_dict:
            ds_dict[k] = None

        month_time = time.time() - month_start
        log(f"[DONE] region={rid}, month={month:02d}, elapsed {month_time:.2f} s", log_file)

        return (rid, month, "done", month_time)

    except Exception as e:
        log(f"[ERROR] region={rid}, month={month:02d}\n{repr(e)}\n{traceback.format_exc()}", log_file)
        return (rid, month, "error", repr(e))


if __name__ == "__main__":
    start_time = time.time()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    feature_medians = load_or_build_feature_medians()
    print("Loaded or generated feature medians.")

    tasks = []
    for cfg in region_configs:
        for month in months:
            tasks.append({
                "cfg": cfg,
                "month": month,
                "feature_medians": feature_medians
            })

    print(f"Number of tasks: {len(tasks)}")
    print(f"Worker processes: {max_workers_process}")
    print(f"RF n_jobs: {rf_n_jobs}")
    print(f"Blocks: {blockx} x {blocky}")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers_process) as executor:
        futures = [executor.submit(process_region_month, task) for task in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print("Task completed:", res)

    total_time = time.time() - start_time
    print("\nAll regions completed.")
    print(f"Total elapsed time: {total_time:.2f} s")