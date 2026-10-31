# -*- coding: utf-8 -*-
""

# ============================================================
# 0. GDAL environment
# ============================================================
import os




# GDAL_CACHEMAX_MB = 4096
# GDAL_THREAD_NUM = "4"
#
# os.environ["GDAL_CACHEMAX"] = str(GDAL_CACHEMAX_MB)   # MB
# os.environ["GDAL_NUM_THREADS"] = GDAL_THREAD_NUM
# os.environ["CPL_NUM_THREADS"] = GDAL_THREAD_NUM

import sys
import time
import csv
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from osgeo import gdal
from tqdm import tqdm

import torch


# ============================================================
# 1. Basic settings
# ============================================================





REGION_CONFIGS = [
    {
        "region_id": 98,
        "band_dir": Path(r"F:\sentinel2_toa_98"),
        "qtp_root": Path(r"E:\QTP_output_2023"),
    },
    {
        "region_id": 100,
        "band_dir": Path(r"G:\sentinel-2_data_2023_2025\sentinel2_toa_100"),
        "qtp_root": Path(r"E:\QTP_output_2023"),
    },


]


YEARS = list(range(2023, 2026))
MONTHS = list(range(1, 13))


# YEARS = [2018]
# MONTHS = [1]

TILE_SIZE = 512
OVERLAP = 32
STRIDE = TILE_SIZE - OVERLAP

INFER_BATCH_SIZE = 4
THRESHOLD = 0.5

USE_AMP_INFERENCE = False



CLEAR_CUDA_CACHE_EVERY_BATCH = False




USE_STRIPE_READING = True

NODATA_VALUE = -32768
OUTPUT_INVALID_VALUE = 2

FINAL_CHUNK_H = 512


RESUME_INFERENCE = True
SKIP_EXISTING_FINAL = True
PROGRESS_SAVE_EVERY = 200


DELETE_TEMP_FILES = True



SKIP_WHOLE_MONTH_ON_READ_ERROR = True


DELETE_INCOMPLETE_TEMP_ON_MONTH_ERROR = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. Paths
# ============================================================
MODEL_PATH = Path(
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step2_semantic_segmentation\model_comparison\model_outputs"
    r"\FULL_DATA_FGSA_UNET_200EPOCHS_SAFE_FP32"
    r"\best_model.pth"
)

STATS_PATH = Path(
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step1_machine_learning_output\generalisation"
    r"\standardisation_configuration"
    r"\tiles_train_test_val_splitted_all_tiles_zscore_configuration.npz"
)


OUT_BASE = Path(
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step2_semantic_segmentation\inference"
    r"\FULL_DATA_FGSA_UNET_200EPOCHS_SAFE_FP32"
)


REGION_ID = None
BAND_DIR = None
INDEX_DIR = None
TERRAIN_DIR = None
TEXTURE_DIR = None
OUT_ROOT = None
LOG_ROOT = None
GLOBAL_ERROR_LOG = None
MONTH_STATUS_LOG = None


def configure_region(region_config):
    ""
    global REGION_ID
    global BAND_DIR, INDEX_DIR, TERRAIN_DIR, TEXTURE_DIR
    global OUT_ROOT, LOG_ROOT, GLOBAL_ERROR_LOG, MONTH_STATUS_LOG

    REGION_ID = int(region_config["region_id"])
    BAND_DIR = Path(region_config["band_dir"])
    qtp_root = Path(region_config["qtp_root"])

    INDEX_DIR = qtp_root / f"{REGION_ID}_water_indices_merged"
    TERRAIN_DIR = qtp_root / f"terrain_features_{REGION_ID}_resampled"
    TEXTURE_DIR = qtp_root / f"{REGION_ID}_textures_merged"

    OUT_ROOT = OUT_BASE / f"region_{REGION_ID}"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    LOG_ROOT = OUT_ROOT / "logs"
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    GLOBAL_ERROR_LOG = LOG_ROOT / f"global_error_log_r{REGION_ID}.txt"
    MONTH_STATUS_LOG = LOG_ROOT / f"month_status_r{REGION_ID}.csv"


def check_current_region_dirs():
    ""
    required_dirs = {
        "BAND_DIR": BAND_DIR,
        "INDEX_DIR": INDEX_DIR,
        "TERRAIN_DIR": TERRAIN_DIR,
        "TEXTURE_DIR": TEXTURE_DIR,
    }

    missing = []
    for name, path in required_dirs.items():
        if path is None or not Path(path).exists():
            missing.append((name, path))

    return missing



MODEL_CODE_DIR = Path(__file__).resolve().parent.parent / "FGSA_UNet"


# ============================================================
# 3. Import model
# ============================================================
if MODEL_CODE_DIR is not None:
    sys.path.insert(0, str(MODEL_CODE_DIR))

try:
    from model import FGSAUNet
except Exception as e:
    raise ImportError(f"Could not import FGSAUNet from model.py: {e}") from e


# ============================================================
# 4. Channel order
# ============================================================
CHANNEL_ORDER = [
    "B2", "B3", "B4", "B8", "B11", "B12",
    "AWEI", "MBWI", "NDWI", "MNDWI",
    "slope", "elevation", "wetness",
    "ASM", "contrast", "dissimilarity", "entropy",
]


# ============================================================
# 5. GDAL setup and helpers
# ============================================================
gdal.UseExceptions()
# gdal.SetCacheMax(GDAL_CACHEMAX_MB * 1024 * 1024)
# gdal.SetConfigOption("GDAL_NUM_THREADS", GDAL_THREAD_NUM)
# gdal.SetConfigOption("CPL_NUM_THREADS", GDAL_THREAD_NUM)


def now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_text_log(log_path, message, also_print=True):
    ""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"[{now_string()}] {message}"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

    if also_print:
        print(text)


def append_csv_row(csv_path, fieldnames, row):
    ""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def log_exception(log_path, title, exc):
    ""
    msg = (
        f"{title}\n"
        f"Exception: {repr(exc)}\n"
        f"Traceback:\n{traceback.format_exc()}"
    )
    append_text_log(log_path, msg, also_print=True)


def open_raster(path):
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open raster: {path}")
    return ds


def create_temp_raster(path, xsize, ysize, dtype, gt, proj):
    ""
    options = [
        "BIGTIFF=YES",
        "TILED=YES",
        "BLOCKXSIZE=512",
        "BLOCKYSIZE=512",
        "COMPRESS=NONE",
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), xsize, ysize, 1, dtype, options=options)

    if ds is None:
        raise RuntimeError(f"Cannot create temp raster: {path}")

    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    return ds


def create_output_raster(path, xsize, ysize, dtype, gt, proj):
    ""
    options = [
        "COMPRESS=LZW",
        "BIGTIFF=YES",
        "TILED=YES",
        "BLOCKXSIZE=512",
        "BLOCKYSIZE=512",
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), xsize, ysize, 1, dtype, options=options)

    if ds is None:
        raise RuntimeError(f"Cannot create output raster: {path}")

    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    return ds


def read_block_from_ds(ds, xoff, yoff, xsize, ysize):
    arr = ds.ReadAsArray(
        xoff=xoff,
        yoff=yoff,
        xsize=xsize,
        ysize=ysize,
    )

    if arr is None:
        raise RuntimeError(
            f"ReadAsArray failed: xoff={xoff}, yoff={yoff}, "
            f"xsize={xsize}, ysize={ysize}"
        )

    return arr


def get_tile_starts(full_size, tile_size, stride):
    ""
    if full_size <= 0:
        raise ValueError(f"Invalid raster size: {full_size}")

    if full_size <= tile_size:
        return [0]

    starts = list(range(0, full_size - tile_size + 1, stride))

    if starts[-1] != full_size - tile_size:
        starts.append(full_size - tile_size)

    return starts


def pad_feature_array(array, target_h, target_w):
    ""
    if array.ndim != 3:
        raise ValueError(f"Expected [C, H, W], got shape={array.shape}")

    channels, h, w = array.shape

    if h > target_h or w > target_w:
        raise ValueError(
            f"Cannot pad array shape={array.shape} to "
            f"target=({channels}, {target_h}, {target_w})"
        )

    if h == target_h and w == target_w:
        return array

    padded = np.full(
        (channels, target_h, target_w),
        NODATA_VALUE,
        dtype=np.float32,
    )
    padded[:, :h, :w] = array
    return padded


# ============================================================
# 5.1 Resume / temp helpers
# ============================================================
def temp_files_exist(prob_sum_path, count_path, invalid_path):
    return (
        Path(prob_sum_path).exists()
        and Path(count_path).exists()
        and Path(invalid_path).exists()
    )


def temp_raster_sizes_match(paths, expected_xsize, expected_ysize):
    ""
    for path in paths:
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        if ds is None:
            return False

        matched = (
            ds.RasterXSize == expected_xsize
            and ds.RasterYSize == expected_ysize
        )
        ds = None

        if not matched:
            print(
                f"[Warning] Temp raster size mismatch: {path} | "
                f"expected={expected_xsize}x{expected_ysize}"
            )
            return False

    return True


def open_temp_rasters_for_update(prob_sum_path, count_path, invalid_path):
    ""
    ds_sum = gdal.Open(str(prob_sum_path), gdal.GA_Update)
    ds_count = gdal.Open(str(count_path), gdal.GA_Update)
    ds_invalid = gdal.Open(str(invalid_path), gdal.GA_Update)

    if ds_sum is None or ds_count is None or ds_invalid is None:
        raise RuntimeError(
            "Cannot open existing temporary files for update; remove them and rerun this month."
        )

    return ds_sum, ds_count, ds_invalid


def create_new_temp_rasters(prob_sum_path, count_path, invalid_path, xsize, ysize, gt, proj):
    ""
    ds_sum = create_temp_raster(prob_sum_path, xsize, ysize, gdal.GDT_Float32, gt, proj)
    ds_count = create_temp_raster(count_path, xsize, ysize, gdal.GDT_UInt16, gt, proj)
    ds_invalid = create_temp_raster(invalid_path, xsize, ysize, gdal.GDT_Byte, gt, proj)

    band_sum = ds_sum.GetRasterBand(1)
    band_count = ds_count.GetRasterBand(1)
    band_invalid = ds_invalid.GetRasterBand(1)

    band_sum.Fill(0)
    band_count.Fill(0)
    band_invalid.Fill(0)

    band_sum.FlushCache()
    band_count.FlushCache()
    band_invalid.FlushCache()

    return ds_sum, ds_count, ds_invalid


def load_or_init_progress(progress_path, total_tiles, allow_resume=True):
    ""
    progress_path = Path(progress_path)

    if allow_resume and progress_path.exists():
        processed = np.load(progress_path)

        if processed.shape == (total_tiles,):
            processed = processed.astype(bool)
            print(f"Loaded progress: {progress_path}")
            print(f"Already processed tiles: {processed.sum()}/{total_tiles}")
            return processed

        print(
            f"[Warning] Progress shape mismatch: {processed.shape}, "
            f"expected ({total_tiles},). Reset progress."
        )

    processed = np.zeros(total_tiles, dtype=bool)
    np.save(progress_path, processed)
    return processed


def save_progress(progress_path, processed):
    ""
    progress_path = Path(progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = progress_path.with_suffix(".tmp.npy")
    np.save(tmp_path, processed)

    if progress_path.exists():
        progress_path.unlink()

    tmp_path.rename(progress_path)


def delete_file_and_sidecars(path):
    ""
    path = Path(path)

    candidates = [
        path,
        Path(str(path) + ".aux.xml"),
        Path(str(path) + ".ovr"),
    ]

    for q in candidates:
        try:
            if q.exists():
                q.unlink()
                print(f"Deleted temp file: {q}")
        except Exception as e:
            print(f"[Warning] failed to delete temp file: {q} | {e}")


def delete_temp_files_after_success(paths):
    ""
    for p in paths:
        delete_file_and_sidecars(p)


def remove_old_incomplete_files(paths):
    ""
    for p in paths:
        delete_file_and_sidecars(p)


# ============================================================
# 6. File finding
# ============================================================
def valid_band_file(path, band):
    name = path.name.upper()

    if band.upper() == "B8":
        if "B8A" in name:
            return False

    return True


def find_by_candidates_or_glob(candidates, folder, patterns, description, validator=None):
    for p in candidates:
        p = Path(p)
        if p.exists():
            if validator is None or validator(p):
                return p

    for pat in patterns:
        matches = sorted(Path(folder).glob(pat))

        for m in matches:
            if validator is None or validator(m):
                return m

    msg = f"Cannot find {description}.\n"
    msg += "Tried candidates:\n"
    msg += "\n".join(str(p) for p in candidates)
    msg += "\nTried glob patterns:\n"
    msg += "\n".join(str(Path(folder) / pat) for pat in patterns)
    raise FileNotFoundError(msg)


def get_band_path(year, month, band):
    candidates = [
        BAND_DIR / f"{REGION_ID}_sentinel2_{year}{month:02d}_{band}.tif",
        BAND_DIR / f"{REGION_ID}_sentinel2_{year}_{month:02d}_{band}.tif",
        BAND_DIR / f"{REGION_ID}_sentinel2_{year}{month:02d}_{band}.TIF",
        BAND_DIR / f"{REGION_ID}_sentinel2_{year}_{month:02d}_{band}.TIF",
    ]

    patterns = [
        f"*{REGION_ID}*{year}{month:02d}*_{band}.tif",
        f"*{REGION_ID}*{year}_{month:02d}*_{band}.tif",
        f"*{REGION_ID}*{year}{month:02d}*{band}*.tif",
        f"*{REGION_ID}*{year}_{month:02d}*{band}*.tif",
        f"*{year}{month:02d}*{band}*.tif",
        f"*{year}_{month:02d}*{band}*.tif",
    ]

    return find_by_candidates_or_glob(
        candidates=candidates,
        folder=BAND_DIR,
        patterns=patterns,
        description=f"band {band}",
        validator=lambda p: valid_band_file(p, band),
    )


def get_index_path(year, month, index_name):
    candidates = [
        INDEX_DIR / f"{REGION_ID}_sentinel2_{year}{month:02d}_{index_name}.tif",
        INDEX_DIR / f"{REGION_ID}_sentinel2_{year}_{month:02d}_{index_name}.tif",
        INDEX_DIR / f"{REGION_ID}_{year}{month:02d}_{index_name}.tif",
        INDEX_DIR / f"{REGION_ID}_{year}_{month:02d}_{index_name}.tif",
    ]

    patterns = [
        f"*{REGION_ID}*{year}{month:02d}*{index_name}*.tif",
        f"*{REGION_ID}*{year}_{month:02d}*{index_name}*.tif",
        f"*{year}{month:02d}*{index_name}*.tif",
        f"*{year}_{month:02d}*{index_name}*.tif",
    ]

    return find_by_candidates_or_glob(
        candidates=candidates,
        folder=INDEX_DIR,
        patterns=patterns,
        description=f"water index {index_name}",
    )


def get_terrain_path(feature_name):
    name = feature_name.lower()

    if name == "slope":
        candidates = [
            TERRAIN_DIR / f"{REGION_ID}_slope.tif",
            TERRAIN_DIR / f"slope_{REGION_ID}.tif",
            TERRAIN_DIR / f"{REGION_ID}_slope_resampled.tif",
        ]
        patterns = [
            f"*{REGION_ID}*slope*.tif",
            f"*slope*{REGION_ID}*.tif",
            f"*slope*.tif",
        ]

    elif name in ["elevation", "dem"]:
        candidates = [
            TERRAIN_DIR / f"{REGION_ID}_elevation.tif",
            TERRAIN_DIR / f"{REGION_ID}_dem.tif",
            TERRAIN_DIR / f"elevation_{REGION_ID}.tif",
            TERRAIN_DIR / f"dem_{REGION_ID}.tif",
            TERRAIN_DIR / f"{REGION_ID}_elevation_resampled.tif",
            TERRAIN_DIR / f"{REGION_ID}_dem_resampled.tif",
        ]
        patterns = [
            f"*{REGION_ID}*elevation*.tif",
            f"*{REGION_ID}*dem*.tif",
            f"*elevation*{REGION_ID}*.tif",
            f"*dem*{REGION_ID}*.tif",
            f"*elevation*.tif",
            f"*dem*.tif",
        ]

    elif name == "wetness":
        candidates = [
            TERRAIN_DIR / f"{REGION_ID}_wetness.tif",
            TERRAIN_DIR / f"wetness_{REGION_ID}.tif",
            TERRAIN_DIR / f"{REGION_ID}_wetness_resampled.tif",
        ]
        patterns = [
            f"*{REGION_ID}*wetness*.tif",
            f"*wetness*{REGION_ID}*.tif",
            f"*wetness*.tif",
        ]

    else:
        raise ValueError(f"Unknown terrain feature: {feature_name}")

    return find_by_candidates_or_glob(
        candidates=candidates,
        folder=TERRAIN_DIR,
        patterns=patterns,
        description=f"terrain feature {feature_name}",
    )


def get_texture_path(year, month, texture_name):
    candidates = [
        TEXTURE_DIR / f"{texture_name}_{REGION_ID}_sentinel2_{year}_{month:02d}_pc1.tif",
        TEXTURE_DIR / f"{texture_name}_{REGION_ID}_sentinel2_{year}{month:02d}_pc1.tif",
        TEXTURE_DIR / f"{texture_name}_{REGION_ID}_{year}_{month:02d}_pc1.tif",
        TEXTURE_DIR / f"{texture_name}_{REGION_ID}_{year}{month:02d}_pc1.tif",
    ]

    texture_lower = texture_name.lower()

    candidates += [
        TEXTURE_DIR / f"{texture_lower}_{REGION_ID}_sentinel2_{year}_{month:02d}_pc1.tif",
        TEXTURE_DIR / f"{texture_lower}_{REGION_ID}_sentinel2_{year}{month:02d}_pc1.tif",
        TEXTURE_DIR / f"{texture_lower}_{REGION_ID}_{year}_{month:02d}_pc1.tif",
        TEXTURE_DIR / f"{texture_lower}_{REGION_ID}_{year}{month:02d}_pc1.tif",
    ]

    patterns = [
        f"*{texture_name}*{REGION_ID}*{year}_{month:02d}*.tif",
        f"*{texture_name}*{REGION_ID}*{year}{month:02d}*.tif",
        f"*{texture_lower}*{REGION_ID}*{year}_{month:02d}*.tif",
        f"*{texture_lower}*{REGION_ID}*{year}{month:02d}*.tif",
        f"*{texture_name}*{year}_{month:02d}*.tif",
        f"*{texture_name}*{year}{month:02d}*.tif",
        f"*{texture_lower}*{year}_{month:02d}*.tif",
        f"*{texture_lower}*{year}{month:02d}*.tif",
    ]

    return find_by_candidates_or_glob(
        candidates=candidates,
        folder=TEXTURE_DIR,
        patterns=patterns,
        description=f"texture feature {texture_name}",
    )


def get_feature_paths(year, month):
    paths = []

    for band in ["B2", "B3", "B4", "B8", "B11", "B12"]:
        paths.append(get_band_path(year, month, band))

    for idx in ["AWEI", "MBWI", "NDWI", "MNDWI"]:
        paths.append(get_index_path(year, month, idx))

    paths.append(get_terrain_path("slope"))
    paths.append(get_terrain_path("elevation"))
    paths.append(get_terrain_path("wetness"))

    for tex in ["ASM", "contrast", "dissimilarity", "entropy"]:
        paths.append(get_texture_path(year, month, tex))

    if len(paths) != 17:
        raise RuntimeError(f"Expected 17 feature paths, got {len(paths)}")

    return paths


# ============================================================
# 7. Standardization
# ============================================================
def load_mean_std(stats_path):
    stats_path = Path(stats_path)

    if not stats_path.exists():
        raise FileNotFoundError(f"Standardisation file not found: {stats_path}")

    stats = np.load(stats_path)

    print("=" * 100)
    print("Loaded standardization npz:")
    print(stats_path)
    print("Available keys:", stats.files)
    print("=" * 100)

    mean_keys = ["mean", "means", "channel_mean", "channel_means"]
    std_keys = ["std", "stds", "channel_std", "channel_stds"]

    mean = None
    std = None

    for k in mean_keys:
        if k in stats.files:
            mean = stats[k]
            break

    for k in std_keys:
        if k in stats.files:
            std = stats[k]
            break

    if mean is None or std is None:
        raise KeyError(
            f"Cannot find mean/std in {stats_path}. "
            f"Available keys: {stats.files}"
        )

    mean = mean.astype(np.float32).reshape(-1)
    std = std.astype(np.float32).reshape(-1)

    if mean.size != 17 or std.size != 17:
        raise ValueError(
            f"mean/std length should be 17. Got mean={mean.size}, std={std.size}"
        )

    std[std == 0] = 1.0

    print("Mean/std loaded:")
    for i, name in enumerate(CHANNEL_ORDER):
        print(f"{i:02d} | {name:15s} | mean={mean[i]: .6f} | std={std[i]: .6f}")

    return mean, std


def preprocess_block(x_block, mean, std):
    ""
    x = x_block.astype(np.float32, copy=True)

    
    # B2, B3, B4, B8, B11, B12
    spectral_invalid_mask = np.all(x[0:6, :, :] == NODATA_VALUE, axis=0)

    
    x[x == NODATA_VALUE] = np.nan

    
    x = (x - mean[:, None, None]) / std[:, None, None]

    
    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)

    return x, spectral_invalid_mask


# ============================================================
# 8. Model loading
# ============================================================
def strip_module_prefix(state_dict):
    new_state = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v

    return new_state


def safe_torch_load(path, map_location):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def load_fgsa_model(model_path, device):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")

    model = FGSAUNet(
        in_channels=17,
        num_classes=1,
        features=(32, 64, 128, 256, 512),
        norm_type="group",
        use_spatial_gate=True,
    ).to(device)

    checkpoint = safe_torch_load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", None)
        best = checkpoint.get("best", None)
    else:
        state_dict = checkpoint
        epoch = None
        best = None

    state_dict = strip_module_prefix(state_dict)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("=" * 100)
    print("Loaded FGSA-UNet model")
    print("Model path:", model_path)
    print("Checkpoint epoch:", epoch)
    if best is not None:
        print("Best info:", best)
    print("=" * 100)

    return model


@torch.no_grad()
def predict_batch(model, x_batch_np, device):
    x = torch.from_numpy(x_batch_np).to(device, non_blocking=True)

    try:
        if USE_AMP_INFERENCE:
            with torch.amp.autocast("cuda"):
                logits = model(x)
        else:
            logits = model(x)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        prob = torch.sigmoid(logits).squeeze(1)
        out = prob.detach().cpu().numpy().astype(np.float32)

        del x, logits, prob

        return out

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n[CUDA OOM] INFER_BATCH_SIZE may be too large.")
            print("Try setting INFER_BATCH_SIZE to 2 or 1.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        raise e


# ============================================================
# 9. Feature reading and alignment
# ============================================================
def print_feature_paths(feature_paths):
    print("=" * 100)
    print("Final 17-channel input order")
    print("=" * 100)

    for i, p in enumerate(feature_paths):
        print(f"{i:02d} | {CHANNEL_ORDER[i]:15s} | {p}")

    print("=" * 100)


def open_feature_datasets(feature_paths):
    ""
    datasets = []

    for i, p in enumerate(feature_paths):
        datasets.append(
            {
                "channel_index": i,
                "channel_name": CHANNEL_ORDER[i],
                "path": Path(p),
                "ds": open_raster(p),
            }
        )

    return datasets


def close_feature_datasets(datasets):
    for item in datasets:
        if isinstance(item, dict):
            item["ds"] = None


def check_feature_alignment(feature_paths):
    ref_ds = open_raster(feature_paths[0])

    ref_x = ref_ds.RasterXSize
    ref_y = ref_ds.RasterYSize
    ref_gt = ref_ds.GetGeoTransform()
    ref_proj = ref_ds.GetProjection()

    ref_ds = None

    for p in feature_paths[1:]:
        ds = open_raster(p)

        if ds.RasterXSize != ref_x or ds.RasterYSize != ref_y:
            raise ValueError(
                "Raster size mismatch:\n"
                f"Reference: {feature_paths[0]} | {ref_x} x {ref_y}\n"
                f"Current  : {p} | {ds.RasterXSize} x {ds.RasterYSize}"
            )

        if ds.GetGeoTransform() != ref_gt:
            print("[Warning] GeoTransform differs:", p)

        if ds.GetProjection() != ref_proj:
            print("[Warning] Projection differs:", p)

        ds = None

    return ref_x, ref_y, ref_gt, ref_proj


def build_feature_block_from_datasets(
    feature_datasets,
    xoff,
    yoff,
    xsize,
    ysize,
    tile_index=None,
    year=None,
    month=None,
):
    ""
    bands = []

    for item in feature_datasets:
        ch_i = item["channel_index"]
        ch_name = item["channel_name"]
        path = item["path"]
        ds = item["ds"]

        try:
            arr = read_block_from_ds(
                ds,
                xoff=xoff,
                yoff=yoff,
                xsize=xsize,
                ysize=ysize,
            ).astype(np.float32)

        except Exception as e:
            error_info = {
                "time": now_string(),
                "region_id": REGION_ID,
                "year": year,
                "month": month,
                "tile_index": tile_index,
                "xoff": xoff,
                "yoff": yoff,
                "xsize": xsize,
                "ysize": ysize,
                "channel_index": ch_i,
                "channel_name": ch_name,
                "path": str(path),
                "error_type": type(e).__name__,
                "error_message": str(e).replace("\n", " | "),
            }

            return None, error_info

        bands.append(arr)

    x_block = np.stack(bands, axis=0)

    if x_block.shape[0] != 17:
        raise ValueError(f"Feature block channel error: {x_block.shape}")

    return x_block, None


def build_feature_stripe_from_datasets(
    feature_datasets,
    yoff,
    xsize,
    stripe_h,
    year=None,
    month=None,
):
    ""
    bands = []

    for item in feature_datasets:
        ch_i = item["channel_index"]
        ch_name = item["channel_name"]
        path = item["path"]
        ds = item["ds"]

        try:
            arr = read_block_from_ds(
                ds,
                xoff=0,
                yoff=yoff,
                xsize=xsize,
                ysize=stripe_h,
            ).astype(np.float32)

        except Exception as e:
            error_info = {
                "time": now_string(),
                "region_id": REGION_ID,
                "year": year,
                "month": month,
                "tile_index": None,
                "xoff": 0,
                "yoff": yoff,
                "xsize": xsize,
                "ysize": stripe_h,
                "channel_index": ch_i,
                "channel_name": ch_name,
                "path": str(path),
                "error_type": type(e).__name__,
                "error_message": str(e).replace("\n", " | "),
            }
            return None, error_info

        bands.append(arr)

    x_stripe = np.stack(bands, axis=0)

    if x_stripe.shape[0] != 17:
        raise ValueError(f"Feature stripe channel error: {x_stripe.shape}")

    if x_stripe.shape[1] != stripe_h or x_stripe.shape[2] != xsize:
        raise ValueError(
            f"Feature stripe shape error: {x_stripe.shape}, "
            f"expected=(17, {stripe_h}, {xsize})"
        )

    return x_stripe, None

# ============================================================
# 10. Inference for one month
# ============================================================
def infer_one_month(model, mean, std, year, month):
    print("\n" + "=" * 100)
    print(f"Start inference: Region {REGION_ID}, {year}-{month:02d}")
    print("=" * 100)

    out_folder = OUT_ROOT / f"{year}" / f"month{month:02d}"
    temp_folder = out_folder / "temp"

    out_folder.mkdir(parents=True, exist_ok=True)
    temp_folder.mkdir(parents=True, exist_ok=True)

    prob_sum_path = temp_folder / f"prob_sum_r{REGION_ID}_{year}_{month:02d}.tif"
    count_path = temp_folder / f"count_r{REGION_ID}_{year}_{month:02d}.tif"
    invalid_path = temp_folder / f"invalid_r{REGION_ID}_{year}_{month:02d}.tif"
    progress_path = temp_folder / f"progress_r{REGION_ID}_{year}_{month:02d}.npy"

    out_mask_path = out_folder / f"mask_fgsa_unet_r{REGION_ID}_{year}_{month:02d}.tif"

    month_log_path = out_folder / f"inference_log_r{REGION_ID}_{year}_{month:02d}.txt"
    bad_tile_csv_path = out_folder / f"bad_read_tiles_r{REGION_ID}_{year}_{month:02d}.csv"

    append_text_log(month_log_path, f"Start inference: Region {REGION_ID}, {year}-{month:02d}")

    # --------------------------------------------------------
    
    # --------------------------------------------------------
    if SKIP_EXISTING_FINAL and out_mask_path.exists():
        print(f"[Skip] Final mask already exists:")
        print(out_mask_path)

        if DELETE_TEMP_FILES:
            delete_temp_files_after_success(
                [
                    prob_sum_path,
                    count_path,
                    invalid_path,
                    progress_path,
                ]
            )
        append_text_log(month_log_path, f"Skip existing final mask: {out_mask_path}")
        return {
            "status": "skip_existing_final",
            "message": str(out_mask_path),
        }

    feature_paths = get_feature_paths(year, month)
    print_feature_paths(feature_paths)

    xsize, ysize, gt, proj = check_feature_alignment(feature_paths)

    
    
    work_xsize = max(xsize, TILE_SIZE)
    work_ysize = max(ysize, TILE_SIZE)

    x_starts = get_tile_starts(work_xsize, TILE_SIZE, STRIDE)
    y_starts = get_tile_starts(work_ysize, TILE_SIZE, STRIDE)

    positions = [(xoff, yoff) for yoff in y_starts for xoff in x_starts]
    total_tiles = len(positions)

    print("Raster size:", xsize, ysize)
    if work_xsize != xsize or work_ysize != ysize:
        print(
            "Small-raster padding enabled:",
            f"original={xsize}x{ysize}, work={work_xsize}x{work_ysize}"
        )
    print("TILE_SIZE:", TILE_SIZE)
    print("OVERLAP:", OVERLAP)
    print("STRIDE:", STRIDE)
    print("INFER_BATCH_SIZE:", INFER_BATCH_SIZE)
    print("x tiles:", len(x_starts))
    print("y tiles:", len(y_starts))
    print("total tiles:", total_tiles)

    # --------------------------------------------------------
    
    # --------------------------------------------------------
    can_resume = (
        RESUME_INFERENCE
        and temp_files_exist(prob_sum_path, count_path, invalid_path)
        and progress_path.exists()
        and temp_raster_sizes_match(
            [prob_sum_path, count_path, invalid_path],
            expected_xsize=work_xsize,
            expected_ysize=work_ysize,
        )
    )

    if can_resume:
        print("[Resume] Found existing temp files and progress file.")
        ds_sum, ds_count, ds_invalid = open_temp_rasters_for_update(
            prob_sum_path,
            count_path,
            invalid_path,
        )
        processed = load_or_init_progress(
            progress_path=progress_path,
            total_tiles=total_tiles,
            allow_resume=True,
        )

    else:
        if RESUME_INFERENCE:
            print("[New run] No complete temp/progress found. Start this month from beginning.")

        
        remove_old_incomplete_files(
            [
                prob_sum_path,
                count_path,
                invalid_path,
                progress_path,
            ]
        )

        ds_sum, ds_count, ds_invalid = create_new_temp_rasters(
            prob_sum_path=prob_sum_path,
            count_path=count_path,
            invalid_path=invalid_path,
            xsize=work_xsize,
            ysize=work_ysize,
            gt=gt,
            proj=proj,
        )

        processed = load_or_init_progress(
            progress_path=progress_path,
            total_tiles=total_tiles,
            allow_resume=False,
        )

    band_sum = ds_sum.GetRasterBand(1)
    band_count = ds_count.GetRasterBand(1)
    band_invalid = ds_invalid.GetRasterBand(1)

    already_done = int(processed.sum())
    if already_done == total_tiles:
        print(f"[Resume] All tiles already accumulated: {already_done}/{total_tiles}")
        need_tile_inference = False
    else:
        need_tile_inference = True

    feature_datasets = None
    pbar = None

    batch_x = []
    batch_masks = []
    batch_positions = []  # each item: (tile_index, xoff, yoff)

    tile_print_marker = -1

    t0_month = time.time()
    t_read = 0.0
    t_preprocess = 0.0
    t_gpu = 0.0
    t_write = 0.0

    processed_since_save = 0
    read_error_tiles = 0

    bad_tile_fields = [
        "time",
        "region_id",
        "year",
        "month",
        "tile_index",
        "xoff",
        "yoff",
        "xsize",
        "ysize",
        "channel_index",
        "channel_name",
        "path",
        "error_type",
        "error_message",
        "action",
    ]

    def flush_one_batch():
        nonlocal batch_x, batch_masks, batch_positions
        nonlocal t_gpu, t_write, processed_since_save

        if len(batch_x) == 0:
            return 0

        x_batch = np.stack(batch_x, axis=0)

        # -----------------------------
        # GPU inference
        # -----------------------------
        t0 = time.time()

        prob_batch = predict_batch(
            model=model,
            x_batch_np=x_batch,
            device=DEVICE,
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t_gpu += time.time() - t0

        # -----------------------------
        # Disk accumulation
        # -----------------------------
        t0 = time.time()

        written_count = 0

        for i in range(len(batch_positions)):
            tile_index_i, xoff_i, yoff_i = batch_positions[i]

            prob = prob_batch[i].astype(np.float32)
            invalid_mask = batch_masks[i]
            valid_mask = (~invalid_mask).astype(np.uint16)

            old_sum = band_sum.ReadAsArray(
                xoff=xoff_i,
                yoff=yoff_i,
                win_xsize=TILE_SIZE,
                win_ysize=TILE_SIZE,
            ).astype(np.float32)

            old_count = band_count.ReadAsArray(
                xoff=xoff_i,
                yoff=yoff_i,
                win_xsize=TILE_SIZE,
                win_ysize=TILE_SIZE,
            ).astype(np.uint16)

            old_invalid = band_invalid.ReadAsArray(
                xoff=xoff_i,
                yoff=yoff_i,
                win_xsize=TILE_SIZE,
                win_ysize=TILE_SIZE,
            ).astype(np.uint8)

            new_sum = old_sum + prob * valid_mask
            new_count = old_count + valid_mask
            new_invalid = np.maximum(
                old_invalid,
                invalid_mask.astype(np.uint8),
            )

            band_sum.WriteArray(new_sum, xoff=xoff_i, yoff=yoff_i)
            band_count.WriteArray(new_count, xoff=xoff_i, yoff=yoff_i)
            band_invalid.WriteArray(new_invalid, xoff=xoff_i, yoff=yoff_i)

            
            processed[tile_index_i] = True
            processed_since_save += 1
            written_count += 1

            del old_sum, old_count, old_invalid
            del new_sum, new_count, new_invalid
            del prob, invalid_mask, valid_mask

        t_write += time.time() - t0

        
        if processed_since_save >= PROGRESS_SAVE_EVERY:
            band_sum.FlushCache()
            band_count.FlushCache()
            band_invalid.FlushCache()
            save_progress(progress_path, processed)
            processed_since_save = 0

        batch_x = []
        batch_masks = []
        batch_positions = []

        del x_batch, prob_batch

        
        
        if CLEAR_CUDA_CACHE_EVERY_BATCH and DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        return written_count

    try:
        if need_tile_inference:
            feature_datasets = open_feature_datasets(feature_paths)

            pbar = tqdm(
                total=total_tiles,
                initial=int(processed.sum()),
                desc=f"Inference {year}-{month:02d}",
            )

            if USE_STRIPE_READING:
                # ====================================================
                
                
                # ====================================================
                n_x_tiles = len(x_starts)

                for y_idx, yoff in enumerate(y_starts):
                    row_tile_indices = [
                        y_idx * n_x_tiles + x_idx
                        for x_idx in range(n_x_tiles)
                    ]

                    
                    if all(processed[idx] for idx in row_tile_indices):
                        continue

                    # -----------------------------
                    # Read one 17-channel stripe
                    # -----------------------------
                    t0 = time.time()

                    
                    
                    read_stripe_h = min(TILE_SIZE, ysize - yoff)

                    x_stripe, read_error_info = build_feature_stripe_from_datasets(
                        feature_datasets=feature_datasets,
                        yoff=yoff,
                        xsize=xsize,
                        stripe_h=read_stripe_h,
                        year=year,
                        month=month,
                    )

                    t_read += time.time() - t0

                    if read_error_info is not None:
                        read_error_tiles += 1
                        read_error_info["action"] = "skip_whole_month_on_first_stripe_read_error"

                        append_csv_row(
                            bad_tile_csv_path,
                            fieldnames=bad_tile_fields,
                            row=read_error_info,
                        )

                        msg = (
                            "[MONTH READ ERROR - SKIP WHOLE MONTH] "
                            f"year={year}, month={month:02d}, "
                            f"yoff={yoff}, "
                            f"channel={read_error_info['channel_index']:02d} "
                            f"{read_error_info['channel_name']}, "
                            f"path={read_error_info['path']}, "
                            f"error={read_error_info['error_message']}"
                        )

                        append_text_log(month_log_path, msg)
                        append_text_log(GLOBAL_ERROR_LOG, msg)

                        append_csv_row(
                            MONTH_STATUS_LOG,
                            fieldnames=["time", "region_id", "year", "month", "status", "message"],
                            row={
                                "time": now_string(),
                                "region_id": REGION_ID,
                                "year": year,
                                "month": month,
                                "status": "skipped_read_error",
                                "message": (
                                    f"yoff={yoff}; "
                                    f"channel={read_error_info['channel_index']:02d} "
                                    f"{read_error_info['channel_name']}; "
                                    f"path={read_error_info['path']}; "
                                    f"error={read_error_info['error_message']}"
                                ),
                            },
                        )

                        try:
                            if pbar is not None:
                                pbar.close()
                        except Exception:
                            pass

                        if feature_datasets is not None:
                            close_feature_datasets(feature_datasets)
                            feature_datasets = None

                        batch_x = []
                        batch_masks = []
                        batch_positions = []

                        try:
                            band_sum.FlushCache()
                            band_count.FlushCache()
                            band_invalid.FlushCache()
                        except Exception:
                            pass

                        band_sum = None
                        band_count = None
                        band_invalid = None
                        ds_sum = None
                        ds_count = None
                        ds_invalid = None

                        if DELETE_INCOMPLETE_TEMP_ON_MONTH_ERROR:
                            remove_old_incomplete_files(
                                [
                                    prob_sum_path,
                                    count_path,
                                    invalid_path,
                                    progress_path,
                                ]
                            )

                        if DEVICE.type == "cuda":
                            torch.cuda.empty_cache()

                        print(f"[Skip month] {year}-{month:02d} skipped due to feature stripe read error.")
                        return {
                            "status": "skipped_read_error",
                            "message": msg,
                        }

                    # -----------------------------
                    # Cut tiles from stripe in memory
                    # -----------------------------
                    for x_idx, xoff in enumerate(x_starts):
                        tile_index = y_idx * n_x_tiles + x_idx

                        if processed[tile_index]:
                            continue

                        
                        
                        read_w = min(TILE_SIZE, xsize - xoff)
                        x_block = x_stripe[:, :read_stripe_h, xoff:xoff + read_w]
                        x_block = pad_feature_array(
                            x_block,
                            target_h=TILE_SIZE,
                            target_w=TILE_SIZE,
                        )

                        if x_block.shape != (17, TILE_SIZE, TILE_SIZE):
                            raise RuntimeError(
                                f"x_block shape error: {x_block.shape}, "
                                f"tile_index={tile_index}, xoff={xoff}, yoff={yoff}"
                            )

                        # -----------------------------
                        # Standardization
                        # -----------------------------
                        t0 = time.time()

                        x_norm, invalid_mask = preprocess_block(
                            x_block=x_block,
                            mean=mean,
                            std=std,
                        )

                        t_preprocess += time.time() - t0

                        batch_x.append(x_norm)
                        batch_masks.append(invalid_mask)
                        batch_positions.append((tile_index, xoff, yoff))

                        if len(batch_x) == INFER_BATCH_SIZE:
                            written = flush_one_batch()
                            pbar.update(written)

                            done_now = int(processed.sum())
                            if done_now // 200 != tile_print_marker and done_now > 0:
                                tile_print_marker = done_now // 200
                                elapsed = time.time() - t0_month
                                print(
                                    f"\n[Profile {year}-{month:02d}] "
                                    f"tiles={done_now}/{total_tiles} | "
                                    f"elapsed={elapsed:.1f}s | "
                                    f"read={t_read:.1f}s | "
                                    f"preprocess={t_preprocess:.1f}s | "
                                    f"gpu={t_gpu:.1f}s | "
                                    f"write={t_write:.1f}s"
                                )

                    del x_stripe

            else:
                # ====================================================
                
                
                # ====================================================
                for tile_index, (xoff, yoff) in enumerate(positions):
                    
                    if processed[tile_index]:
                        continue

                    # -----------------------------
                    # Read 17-channel block
                    # -----------------------------
                    t0 = time.time()

                    read_w = min(TILE_SIZE, xsize - xoff)
                    read_h = min(TILE_SIZE, ysize - yoff)

                    x_block, read_error_info = build_feature_block_from_datasets(
                        feature_datasets=feature_datasets,
                        xoff=xoff,
                        yoff=yoff,
                        xsize=read_w,
                        ysize=read_h,
                        tile_index=tile_index,
                        year=year,
                        month=month,
                    )

                    t_read += time.time() - t0

                    if read_error_info is not None:
                        read_error_tiles += 1
                        read_error_info["action"] = "skip_whole_month_on_first_read_error"

                        append_csv_row(
                            bad_tile_csv_path,
                            fieldnames=bad_tile_fields,
                            row=read_error_info,
                        )

                        msg = (
                            "[MONTH READ ERROR - SKIP WHOLE MONTH] "
                            f"year={year}, month={month:02d}, "
                            f"tile_index={tile_index}, "
                            f"xoff={xoff}, yoff={yoff}, "
                            f"channel={read_error_info['channel_index']:02d} "
                            f"{read_error_info['channel_name']}, "
                            f"path={read_error_info['path']}, "
                            f"error={read_error_info['error_message']}"
                        )

                        append_text_log(month_log_path, msg)
                        append_text_log(GLOBAL_ERROR_LOG, msg)

                        append_csv_row(
                            MONTH_STATUS_LOG,
                            fieldnames=["time", "region_id", "year", "month", "status", "message"],
                            row={
                                "time": now_string(),
                                "region_id": REGION_ID,
                                "year": year,
                                "month": month,
                                "status": "skipped_read_error",
                                "message": (
                                    f"tile_index={tile_index}; "
                                    f"xoff={xoff}; yoff={yoff}; "
                                    f"channel={read_error_info['channel_index']:02d} "
                                    f"{read_error_info['channel_name']}; "
                                    f"path={read_error_info['path']}; "
                                    f"error={read_error_info['error_message']}"
                                ),
                            },
                        )

                        try:
                            if pbar is not None:
                                pbar.close()
                        except Exception:
                            pass

                        if feature_datasets is not None:
                            close_feature_datasets(feature_datasets)
                            feature_datasets = None

                        batch_x = []
                        batch_masks = []
                        batch_positions = []

                        try:
                            band_sum.FlushCache()
                            band_count.FlushCache()
                            band_invalid.FlushCache()
                        except Exception:
                            pass

                        band_sum = None
                        band_count = None
                        band_invalid = None
                        ds_sum = None
                        ds_count = None
                        ds_invalid = None

                        if DELETE_INCOMPLETE_TEMP_ON_MONTH_ERROR:
                            remove_old_incomplete_files(
                                [
                                    prob_sum_path,
                                    count_path,
                                    invalid_path,
                                    progress_path,
                                ]
                            )

                        if DEVICE.type == "cuda":
                            torch.cuda.empty_cache()

                        print(f"[Skip month] {year}-{month:02d} skipped due to feature read error.")
                        return {
                            "status": "skipped_read_error",
                            "message": msg,
                        }

                    x_block = pad_feature_array(
                        x_block,
                        target_h=TILE_SIZE,
                        target_w=TILE_SIZE,
                    )

                    # -----------------------------
                    # Standardization
                    # -----------------------------
                    t0 = time.time()

                    x_norm, invalid_mask = preprocess_block(
                        x_block=x_block,
                        mean=mean,
                        std=std,
                    )

                    t_preprocess += time.time() - t0

                    batch_x.append(x_norm)
                    batch_masks.append(invalid_mask)
                    batch_positions.append((tile_index, xoff, yoff))

                    del x_block

                    if len(batch_x) == INFER_BATCH_SIZE:
                        written = flush_one_batch()
                        pbar.update(written)

                        done_now = int(processed.sum())
                        if done_now // 200 != tile_print_marker and done_now > 0:
                            tile_print_marker = done_now // 200
                            elapsed = time.time() - t0_month
                            print(
                                f"\n[Profile {year}-{month:02d}] "
                                f"tiles={done_now}/{total_tiles} | "
                                f"elapsed={elapsed:.1f}s | "
                                f"read={t_read:.1f}s | "
                                f"preprocess={t_preprocess:.1f}s | "
                                f"gpu={t_gpu:.1f}s | "
                                f"write={t_write:.1f}s"
                            )

            
            written = flush_one_batch()
            pbar.update(written)

            pbar.close()

            
            band_sum.FlushCache()
            band_count.FlushCache()
            band_invalid.FlushCache()
            save_progress(progress_path, processed)

        if int(processed.sum()) != total_tiles:
            raise RuntimeError(
                f"Tile accumulation incomplete: {processed.sum()}/{total_tiles}. "
                f"Progress saved at: {progress_path}"
            )

    finally:
        if feature_datasets is not None:
            close_feature_datasets(feature_datasets)

    band_sum.FlushCache()
    band_count.FlushCache()
    band_invalid.FlushCache()

    ds_sum = None
    ds_count = None
    ds_invalid = None

    # ========================================================
    # Final probability averaging and thresholding
    # ========================================================
    print("\nStart writing final mask...")

    ds_sum = open_raster(prob_sum_path)
    ds_count = open_raster(count_path)
    ds_invalid = open_raster(invalid_path)

    band_sum = ds_sum.GetRasterBand(1)
    band_count = ds_count.GetRasterBand(1)
    band_invalid = ds_invalid.GetRasterBand(1)

    ds_out = create_output_raster(out_mask_path, xsize, ysize, gdal.GDT_Byte, gt, proj)
    band_out = ds_out.GetRasterBand(1)
    band_out.SetNoDataValue(OUTPUT_INVALID_VALUE)

    for y0 in tqdm(range(0, ysize, FINAL_CHUNK_H), desc=f"Writing final mask {year}-{month:02d}"):
        h = min(FINAL_CHUNK_H, ysize - y0)

        sum_arr = band_sum.ReadAsArray(
            xoff=0,
            yoff=y0,
            win_xsize=xsize,
            win_ysize=h,
        ).astype(np.float32)

        count_arr = band_count.ReadAsArray(
            xoff=0,
            yoff=y0,
            win_xsize=xsize,
            win_ysize=h,
        ).astype(np.uint16)

        invalid_arr = band_invalid.ReadAsArray(
            xoff=0,
            yoff=y0,
            win_xsize=xsize,
            win_ysize=h,
        ).astype(np.uint8)

        avg_prob = np.zeros_like(sum_arr, dtype=np.float32)

        valid_prob = count_arr > 0
        avg_prob[valid_prob] = sum_arr[valid_prob] / count_arr[valid_prob]

        pred = (avg_prob >= THRESHOLD).astype(np.uint8)

        
        nodata_mask = (invalid_arr == 1) | (count_arr == 0)
        pred[nodata_mask] = OUTPUT_INVALID_VALUE

        band_out.WriteArray(pred, xoff=0, yoff=y0)

        del sum_arr, count_arr, invalid_arr, avg_prob, pred, nodata_mask

    band_out.FlushCache()

    ds_sum = None
    ds_count = None
    ds_invalid = None
    ds_out = None

    elapsed_month = time.time() - t0_month

    print("\nSaved final mask:")
    print(out_mask_path)
    print(
        f"[Month profile] total={elapsed_month:.1f}s | "
        f"read={t_read:.1f}s | "
        f"preprocess={t_preprocess:.1f}s | "
        f"gpu={t_gpu:.1f}s | "
        f"write={t_write:.1f}s | "
        f"bad_read_tiles={read_error_tiles}"
    )

    append_text_log(
        month_log_path,
        f"Finished inference: Region {REGION_ID}, {year}-{month:02d}, "
        f"final={out_mask_path}, bad_read_tiles={read_error_tiles}, "
        f"elapsed={elapsed_month:.1f}s",
    )

    # ========================================================
    # Delete temp files after final mask is successfully saved
    # ========================================================
    if DELETE_TEMP_FILES:
        delete_temp_files_after_success(
            [
                prob_sum_path,
                count_path,
                invalid_path,
                progress_path,
            ]
        )

    return {
        "status": "finished",
        "message": str(out_mask_path),
    }


# ============================================================
# 11. Main
# ============================================================
def main():
    print("=" * 100)
    print("Stable/Fast FGSA-UNet PyTorch GPU inference - multi-region sequential run")
    print("=" * 100)
    print("DEVICE:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        raise RuntimeError("CUDA is unavailable; check the PyTorch GPU environment.")

    print("MODEL_PATH:", MODEL_PATH)
    print("STATS_PATH:", STATS_PATH)
    print("OUT_BASE:", OUT_BASE)
    print("REGION_CONFIGS:", REGION_CONFIGS)
    print("YEARS:", YEARS)
    print("MONTHS:", MONTHS)
    print("RESUME_INFERENCE:", RESUME_INFERENCE)
    print("SKIP_EXISTING_FINAL:", SKIP_EXISTING_FINAL)
    print("DELETE_TEMP_FILES:", DELETE_TEMP_FILES)
    # print("GDAL_CACHEMAX_MB:", GDAL_CACHEMAX_MB)
    # print("GDAL_THREAD_NUM:", GDAL_THREAD_NUM)
    print("PROGRESS_SAVE_EVERY:", PROGRESS_SAVE_EVERY)
    print("CLEAR_CUDA_CACHE_EVERY_BATCH:", CLEAR_CUDA_CACHE_EVERY_BATCH)
    print("USE_STRIPE_READING:", USE_STRIPE_READING)
    print("SKIP_WHOLE_MONTH_ON_READ_ERROR:", SKIP_WHOLE_MONTH_ON_READ_ERROR)
    print("DELETE_INCOMPLETE_TEMP_ON_MONTH_ERROR:", DELETE_INCOMPLETE_TEMP_ON_MONTH_ERROR)
    print("=" * 100)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {MODEL_PATH}")

    if not STATS_PATH.exists():
        raise FileNotFoundError(f"STATS_PATH does not exist: {STATS_PATH}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    
    mean, std = load_mean_std(STATS_PATH)

    model = load_fgsa_model(
        model_path=MODEL_PATH,
        device=DEVICE,
    )

    for region_config in REGION_CONFIGS:
        configure_region(region_config)

        print("\n" + "#" * 100)
        print(f"Start Region {REGION_ID}")
        print("#" * 100)
        print("BAND_DIR:", BAND_DIR)
        print("INDEX_DIR:", INDEX_DIR)
        print("TERRAIN_DIR:", TERRAIN_DIR)
        print("TEXTURE_DIR:", TEXTURE_DIR)
        print("OUT_ROOT:", OUT_ROOT)
        print("LOG_ROOT:", LOG_ROOT)
        print("#" * 100)

        missing_dirs = check_current_region_dirs()
        if missing_dirs:
            msg_lines = [f"[REGION SKIPPED] Region {REGION_ID} has missing input directories:"]
            for name, path in missing_dirs:
                msg_lines.append(f"  {name}: {path}")
            msg = "\n".join(msg_lines)

            print(msg)
            append_text_log(GLOBAL_ERROR_LOG, msg, also_print=False)
            append_csv_row(
                MONTH_STATUS_LOG,
                fieldnames=["time", "region_id", "year", "month", "status", "message"],
                row={
                    "time": now_string(),
                    "region_id": REGION_ID,
                    "year": "",
                    "month": "",
                    "status": "region_skipped_missing_dirs",
                    "message": msg.replace("\n", " | "),
                },
            )
            continue

        for year in YEARS:
            for month in MONTHS:
                try:
                    result = infer_one_month(
                        model=model,
                        mean=mean,
                        std=std,
                        year=year,
                        month=month,
                    )

                    
                    
                    if result is None:
                        result = {"status": "returned_none", "message": "infer_one_month returned None"}

                    append_csv_row(
                        MONTH_STATUS_LOG,
                        fieldnames=["time", "region_id", "year", "month", "status", "message"],
                        row={
                            "time": now_string(),
                            "region_id": REGION_ID,
                            "year": year,
                            "month": month,
                            "status": result.get("status", "unknown_return"),
                            "message": result.get("message", ""),
                        },
                    )

                except Exception as e:
                    
                    
                    log_exception(
                        GLOBAL_ERROR_LOG,
                        f"[MONTH FAILED] Region {REGION_ID}, {year}-{month:02d}",
                        e,
                    )

                    append_csv_row(
                        MONTH_STATUS_LOG,
                        fieldnames=["time", "region_id", "year", "month", "status", "message"],
                        row={
                            "time": now_string(),
                            "region_id": REGION_ID,
                            "year": year,
                            "month": month,
                            "status": "failed_and_continue",
                            "message": repr(e),
                        },
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    continue

        print(f"\nRegion {REGION_ID} finished. Check logs at:")
        print(LOG_ROOT)

    print("\nAll regions finished.")
    print("Output base:")
    print(OUT_BASE)


if __name__ == "__main__":
    main()
