"""Generate water indices, PC1 and GLCM textures from Sentinel-2 bands."""

import csv
import gc
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import fast_glcm
import numpy as np
import pandas as pd
from osgeo import gdal
from tqdm import tqdm


REGION_ID = 101
START_YEAR = 2023
END_YEAR = 2025

INPUT_FOLDER = Path(
    rf"E:\sentinel-2_data_2023_2025\sentinel2_toa_{REGION_ID}"
)
OUTPUT_ROOT = Path(r"D:\phdl2\QTP surface water dynamics\code availability\test_output\feature_preparation\production")

WATER_FOLDER = OUTPUT_ROOT / f"{REGION_ID}_water_indices_merged"
PC1_FOLDER = OUTPUT_ROOT / f"{REGION_ID}_PC1_merged"
VARIANCE_FOLDER = OUTPUT_ROOT / f"{REGION_ID}_PC1_variance"
TEXTURE_FOLDER = OUTPUT_ROOT / f"{REGION_ID}_textures_merged"
LOG_FOLDER = OUTPUT_ROOT / f"{REGION_ID}_logs"
WORK_FOLDER = OUTPUT_ROOT / f"{REGION_ID}_full_month_work"

MASTER_LOG = LOG_FOLDER / f"{REGION_ID}_full_month_pipeline.log"
SUMMARY_CSV = LOG_FOLDER / f"{REGION_ID}_full_month_pipeline_summary.csv"

# Each image is divided into 20 × 20 independent processing blocks.
BLOCK_X = 20
BLOCK_Y = 20
NODATA_INT16 = -32768


MAX_WORKERS = 1


VERIFY_EXISTING_FINALS = True


VERIFY_AFTER_FINISH = True


CHECKPOINT_EACH_ROW = True


GTIFF_OPTIONS = [
    "BIGTIFF=YES",
    "TILED=YES",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512",
    "COMPRESS=LZW",
]

INDEX_FEATURES = ("NDWI", "MNDWI", "AWEI", "MBWI")
TEXTURE_FEATURES = ("contrast", "dissimilarity", "entropy", "ASM")
BAND_NAMES = ("B2", "B3", "B4", "B8", "B11", "B12")


gdal.UseExceptions()
gdal.SetConfigOption("GDAL_NUM_THREADS", "1")
gdal.SetConfigOption("CPL_NUM_THREADS", "1")


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_text(path, text):
    mkdir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as file:
        file.write(text + "\n")


def master_log(message):
    text = f"[{now()}] {message}"
    print(text, flush=True)
    append_text(MASTER_LOG, text)


def month_log_path(year, month):
    return LOG_FOLDER / f"{REGION_ID}_{year}_{month:02d}_full_month.log"


def month_log(year, month, message):
    text = f"[{now()}] {message}"
    print(text, flush=True)
    append_text(month_log_path(year, month), text)


def write_summary(year, month, status, elapsed, completed_blocks, message=""):
    mkdir(LOG_FOLDER)
    fields = [
        "time",
        "region",
        "year",
        "month",
        "status",
        "completed_blocks",
        "expected_blocks",
        "elapsed_seconds",
        "message",
    ]

    row = {
        "time": now(),
        "region": REGION_ID,
        "year": year,
        "month": f"{month:02d}",
        "status": status,
        "completed_blocks": completed_blocks,
        "expected_blocks": BLOCK_X * BLOCK_Y,
        "elapsed_seconds": f"{elapsed:.3f}",
        "message": message,
    }

    is_new = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def gdal_open_or_raise(path, access=gdal.GA_ReadOnly):
    dataset = gdal.Open(str(path), access)
    if dataset is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    return dataset


def band_path(year, month, band):
    return INPUT_FOLDER / f"{REGION_ID}_sentinel2_{year}{month:02d}_{band}.tif"


def final_paths(year, month):
    return {
        "NDWI": WATER_FOLDER / f"{REGION_ID}_sentinel2_{year}{month:02d}_NDWI.tif",
        "MNDWI": WATER_FOLDER / f"{REGION_ID}_sentinel2_{year}{month:02d}_MNDWI.tif",
        "AWEI": WATER_FOLDER / f"{REGION_ID}_sentinel2_{year}{month:02d}_AWEI.tif",
        "MBWI": WATER_FOLDER / f"{REGION_ID}_sentinel2_{year}{month:02d}_MBWI.tif",
        "PC1": PC1_FOLDER / f"{REGION_ID}_sentinel2_{year}_{month:02d}_pc1.tif",
        "contrast": TEXTURE_FOLDER
        / f"contrast_{REGION_ID}_sentinel2_{year}_{month:02d}_pc1.tif",
        "dissimilarity": TEXTURE_FOLDER
        / f"dissimilarity_{REGION_ID}_sentinel2_{year}_{month:02d}_pc1.tif",
        "entropy": TEXTURE_FOLDER
        / f"entropy_{REGION_ID}_sentinel2_{year}_{month:02d}_pc1.tif",
        "ASM": TEXTURE_FOLDER
        / f"ASM_{REGION_ID}_sentinel2_{year}_{month:02d}_pc1.tif",
    }


def work_dir(year, month):
    return WORK_FOLDER / f"{year}_{month:02d}"


def part_paths(year, month):
    work = work_dir(year, month)
    finals = final_paths(year, month)
    return {
        key: work / f"{path.name}.part.tif"
        for key, path in finals.items()
    }


def status_path(year, month):
    return work_dir(year, month) / "block_status.npy"


def variance_work_path(year, month):
    return work_dir(year, month) / "pc1_variance_work.csv"


def variance_final_path(year, month):
    return VARIANCE_FOLDER / f"{REGION_ID}_{year}_{month:02d}_pc1_variance.csv"


def atomic_save_status(path, status):
    path = Path(path)
    mkdir(path.parent)
    temp = Path(str(path) + ".tmp")
    with open(temp, "wb") as file:
        np.save(file, status)
    os.replace(temp, path)


def load_status(path):
    path = Path(path)
    if not path.exists():
        return np.zeros((BLOCK_Y, BLOCK_X), dtype=np.uint8)

    with open(path, "rb") as file:
        status = np.load(file)

    if status.shape != (BLOCK_Y, BLOCK_X):
        raise ValueError(
            f"Invalid status shape: {status.shape}, "
            f"expected {(BLOCK_Y, BLOCK_X)}"
        )

    return status.astype(np.uint8)


def flush_outputs(output_bands, output_datasets):
    for band in output_bands.values():
        if band is not None:
            band.FlushCache()

    for dataset in output_datasets.values():
        if dataset is not None:
            dataset.FlushCache()


def close_outputs(output_bands, output_datasets):
    for key in list(output_bands):
        output_bands[key] = None
    for key in list(output_datasets):
        output_datasets[key] = None
    gc.collect()


def fast_verify_raster(path, expected_xsize=None, expected_ysize=None):
    dataset = None
    band = None

    try:
        dataset = gdal_open_or_raise(path)

        if dataset.RasterCount < 1:
            return False, "RasterCount < 1"

        xsize = dataset.RasterXSize
        ysize = dataset.RasterYSize

        if expected_xsize is not None and xsize != expected_xsize:
            return False, f"xsize mismatch: {xsize} != {expected_xsize}"

        if expected_ysize is not None and ysize != expected_ysize:
            return False, f"ysize mismatch: {ysize} != {expected_ysize}"

        band = dataset.GetRasterBand(1)
        sample_x = min(256, xsize)
        sample_y = min(256, ysize)

        first = band.ReadAsArray(0, 0, sample_x, sample_y)
        if first is None:
            return False, "cannot read first sample"
        del first

        xoff = max(0, xsize - sample_x)
        yoff = max(0, ysize - sample_y)
        last = band.ReadAsArray(xoff, yoff, sample_x, sample_y)
        if last is None:
            return False, "cannot read last sample"
        del last

        return True, ""

    except Exception as error:
        return False, f"{type(error).__name__}: {error}"

    finally:
        band = None
        dataset = None
        gc.collect()


def finals_are_complete_and_valid(year, month):
    paths = final_paths(year, month)

    if not all(path.exists() for path in paths.values()):
        return False

    if not variance_final_path(year, month).exists():
        return False

    if not VERIFY_EXISTING_FINALS:
        return True

    for path in paths.values():
        ok, message = fast_verify_raster(path)
        if not ok:
            month_log(
                year,
                month,
                f"[FINAL INVALID - REBUILD] {path} | {message}",
            )
            return False

    return True


def reset_incomplete_work(year, month):
    work = work_dir(year, month)
    if not work.exists():
        return

    for path in work.glob("*"):
        try:
            path.unlink()
        except Exception:
            pass


def input_paths_and_alignment(year, month):
    paths = [band_path(year, month, band) for band in BAND_NAMES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input bands: {missing}")

    datasets = [gdal_open_or_raise(path) for path in paths]
    reference = datasets[0]

    xsize = reference.RasterXSize
    ysize = reference.RasterYSize
    gt = reference.GetGeoTransform()
    proj = reference.GetProjection()

    for dataset, path in zip(datasets[1:], paths[1:]):
        if dataset.RasterXSize != xsize or dataset.RasterYSize != ysize:
            raise ValueError(f"Size mismatch: {path}")
        if dataset.GetGeoTransform() != gt:
            raise ValueError(f"GeoTransform mismatch: {path}")
        if dataset.GetProjection() != proj:
            raise ValueError(f"Projection mismatch: {path}")

    return paths, datasets, xsize, ysize, gt, proj


def create_full_raster(path, xsize, ysize, gt, proj, gdal_type, nodata):
    path = Path(path)
    mkdir(path.parent)

    if path.exists():
        path.unlink()

    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise RuntimeError("GDAL GTiff driver not found")

    dataset = driver.Create(
        str(path),
        xsize=xsize,
        ysize=ysize,
        bands=1,
        eType=gdal_type,
        options=GTIFF_OPTIONS,
    )
    if dataset is None:
        raise RuntimeError(f"Create failed: {path}")

    dataset.SetGeoTransform(gt)
    dataset.SetProjection(proj)

    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(nodata)

    return dataset, band


def open_or_create_parts(year, month, xsize, ysize, gt, proj):
    parts = part_paths(year, month)
    status_file = status_path(year, month)

    all_parts_exist = all(path.exists() for path in parts.values())
    status_exists = status_file.exists()


    if all_parts_exist != status_exists:
        month_log(
            year,
            month,
            "[RESET PARTIAL STATE] part files and status are inconsistent",
        )
        reset_incomplete_work(year, month)
        all_parts_exist = False
        status_exists = False

    output_datasets = {}
    output_bands = {}

    if all_parts_exist and status_exists:
        for key, path in parts.items():
            dataset = gdal_open_or_raise(path, gdal.GA_Update)

            if dataset.RasterXSize != xsize or dataset.RasterYSize != ysize:
                close_outputs(output_bands, output_datasets)
                reset_incomplete_work(year, month)
                return open_or_create_parts(year, month, xsize, ysize, gt, proj)

            output_datasets[key] = dataset
            output_bands[key] = dataset.GetRasterBand(1)

        status = load_status(status_file)
        return parts, output_datasets, output_bands, status

    mkdir(work_dir(year, month))

    int16_keys = ("NDWI", "MNDWI", "AWEI", "MBWI", "PC1")
    float32_keys = ("contrast", "dissimilarity", "entropy", "ASM")

    for key in int16_keys:
        dataset, band = create_full_raster(
            parts[key],
            xsize,
            ysize,
            gt,
            proj,
            gdal.GDT_Int16,
            NODATA_INT16,
        )
        output_datasets[key] = dataset
        output_bands[key] = band

    for key in float32_keys:
        dataset, band = create_full_raster(
            parts[key],
            xsize,
            ysize,
            gt,
            proj,
            gdal.GDT_Float32,
            np.nan,
        )
        output_datasets[key] = dataset
        output_bands[key] = band

    status = np.zeros((BLOCK_Y, BLOCK_X), dtype=np.uint8)
    atomic_save_status(status_file, status)

    return parts, output_datasets, output_bands, status


class S2WaterIndices:
    def __init__(self, b2, b3, b4, b8, b11, b12):
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.b8 = b8
        self.b11 = b11
        self.b12 = b12

    def MNDWI(self):
        return (self.b3 - self.b11) / (self.b3 + self.b11)

    def NDWI(self):
        return (self.b3 - self.b8) / (self.b3 + self.b8)

    def MBWI(self):
        return 3 * self.b3 - self.b4 - self.b8 - self.b11 - self.b12

    def AWEI_sh(self):
        return (
            self.b2
            + 2.5 * self.b3
            - 1.5 * (self.b8 + self.b11)
            - 0.25 * self.b12
        )


def array_process(array):
    array = array.astype(np.float32)
    array[array == NODATA_INT16] = np.nan
    array = array / 10000.0
    return array


def to_int16_with_nodata(array):
    array = array.copy()
    array[np.isnan(array)] = NODATA_INT16
    return array.astype(np.int16)


def normalize_to_uint8(img, nodata_mask):
    img = img.astype(np.float32)
    valid = ~nodata_mask

    if np.sum(valid) == 0:
        return np.zeros_like(img, dtype=np.uint8)

    vmin = np.nanmin(img[valid])
    vmax = np.nanmax(img[valid])

    if vmax == vmin:
        return np.zeros_like(img, dtype=np.uint8)

    output = np.zeros_like(img, dtype=np.float32)
    output[valid] = 255.0 * (img[valid] - vmin) / (vmax - vmin)
    return np.clip(output, 0, 255).astype(np.uint8)


def scale_features(array, nodata_mask):
    output = array.copy()
    output[nodata_mask] = np.nan
    return output


def first_pc_numpy(X):
    mean_vec = X.mean(axis=0, keepdims=True)
    Xc = X - mean_vec

    covariance = np.dot(Xc.T, Xc) / max(Xc.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    index = np.argmax(eigenvalues)
    pc1_vector = eigenvectors[:, index]
    pc1_flat = np.dot(Xc, pc1_vector)

    total_variance = np.sum(eigenvalues)
    variance_ratio = (
        0.0
        if total_variance <= 0
        else float(eigenvalues[index] / total_variance)
    )

    return pc1_flat, variance_ratio


def empty_texture(shape):
    return np.full(shape, np.nan, dtype=np.float32)


def calculate_block(raw_arrays):
    """Calculate four water indices, blockwise PC1 and GLCM textures."""
    b2, b3, b4, b8, b11, b12 = [
        array_process(array) for array in raw_arrays
    ]

    s2 = S2WaterIndices(b2, b3, b4, b8, b11, b12)

    mndwi = to_int16_with_nodata(s2.MNDWI() * 10000)
    ndwi = to_int16_with_nodata(s2.NDWI() * 10000)
    mbwi = to_int16_with_nodata(s2.MBWI() * 10000)
    awei = to_int16_with_nodata(s2.AWEI_sh() * 10000)


    composited = np.stack([ndwi, mndwi, awei, mbwi], axis=0)
    nodata_mask = composited[0] == NODATA_INT16

    ysize = composited.shape[1]
    xsize = composited.shape[2]

    X = composited.transpose(1, 2, 0).reshape(xsize * ysize, 4)
    X = X.astype(np.float32)
    # Match the original PCA input treatment for missing values.
    X[X == NODATA_INT16] = 0

    results = {
        "NDWI": ndwi,
        "MNDWI": mndwi,
        "AWEI": awei,
        "MBWI": mbwi,
    }

    if np.all(X == 0):
        results["PC1"] = np.full(
            (ysize, xsize),
            NODATA_INT16,
            dtype=np.int16,
        )
        for feature in TEXTURE_FEATURES:
            results[feature] = empty_texture((ysize, xsize))
        return results, None, "ALL_ZERO"

    pc1_flat, variance_ratio = first_pc_numpy(X)
    pc1 = pc1_flat.reshape(ysize, xsize)
    pc1[nodata_mask] = NODATA_INT16
    pc1_int16 = pc1.astype(np.int16)
    results["PC1"] = pc1_int16

    pc1_float = pc1_int16.astype(np.float32)
    pc1_nodata_mask = pc1_float == NODATA_INT16
    pc1_float[pc1_nodata_mask] = 0

    img_uint8 = normalize_to_uint8(pc1_float, pc1_nodata_mask)
    valid_values = img_uint8[~pc1_nodata_mask]

    if valid_values.size == 0:
        for feature in TEXTURE_FEATURES:
            results[feature] = empty_texture((ysize, xsize))
        return results, variance_ratio, "NO_VALID_PIXELS"

    if np.min(valid_values) == np.max(valid_values):
        for feature in TEXTURE_FEATURES:
            results[feature] = empty_texture((ysize, xsize))
        return results, variance_ratio, "CONSTANT_BLOCK"


    # GLCM counts use eight grey levels and a 7 × 7 window.
    nbit = 8
    slide_window = 7
    step = 2
    angle = 0

    glcm = fast_glcm.fast_glcm(
        img_uint8,
        vmin=0,
        vmax=255,
        levels=nbit,
        kernel_size=slide_window,
        distance=step,
        angle=angle,
    )

    contrast = fast_glcm.fast_glcm_contrast_from_glcm(
        glcm,
        img_uint8,
        levels=nbit,
    )
    results["contrast"] = scale_features(
        contrast,
        pc1_nodata_mask,
    ).astype(np.float32)

    dissimilarity = fast_glcm.fast_glcm_dissimilarity_from_glcm(
        glcm,
        img_uint8,
        levels=nbit,
    )
    results["dissimilarity"] = scale_features(
        dissimilarity,
        pc1_nodata_mask,
    ).astype(np.float32)

    entropy = fast_glcm.fast_glcm_entropy_from_glcm(
        glcm,
        img_uint8,
        ks=slide_window,
    )
    results["entropy"] = scale_features(
        entropy,
        pc1_nodata_mask,
    ).astype(np.float32)

    asm, _ = fast_glcm.fast_glcm_ASM_from_glcm(
        glcm,
        img_uint8,
        levels=nbit,
    )
    results["ASM"] = scale_features(
        asm,
        pc1_nodata_mask,
    ).astype(np.float32)

    return results, variance_ratio, "OK"


def read_variance_records(year, month):
    path = variance_work_path(year, month)
    if not path.exists():
        return {}

    dataframe = pd.read_csv(path)
    records = {}

    for row in dataframe.itertuples(index=False):
        records[(int(row.block_c), int(row.block_l))] = float(
            row.variance
        )

    return records


def save_variance_records(year, month, records, final=False):
    rows = [
        {
            "block_c": c,
            "block_l": l,
            "variance": variance,
        }
        for (c, l), variance in sorted(
            records.items(),
            key=lambda item: (item[0][1], item[0][0]),
        )
    ]

    dataframe = pd.DataFrame(
        rows,
        columns=["block_c", "block_l", "variance"],
    )

    target = (
        variance_final_path(year, month)
        if final
        else variance_work_path(year, month)
    )

    mkdir(target.parent)
    dataframe.to_csv(target, index=False, encoding="utf-8-sig")


def install_finished_parts(year, month, parts, xsize, ysize):
    finals = final_paths(year, month)

    if VERIFY_AFTER_FINISH:
        for key, path in parts.items():
            ok, message = fast_verify_raster(
                path,
                expected_xsize=xsize,
                expected_ysize=ysize,
            )
            if not ok:
                raise RuntimeError(
                    f"Part verify failed: key={key}, path={path}, "
                    f"message={message}"
                )


    for key in finals:
        mkdir(finals[key].parent)
        os.replace(parts[key], finals[key])


def process_month(year, month):
    start_time = time.time()
    input_datasets = []
    output_datasets = {}
    output_bands = {}

    try:
        mkdir(WATER_FOLDER)
        mkdir(PC1_FOLDER)
        mkdir(VARIANCE_FOLDER)
        mkdir(TEXTURE_FOLDER)
        mkdir(LOG_FOLDER)
        mkdir(WORK_FOLDER)

        month_log(year, month, "=" * 80)
        month_log(year, month, f"MONTH START {year}-{month:02d}")

        if finals_are_complete_and_valid(year, month):
            month_log(year, month, "[SKIP] all final outputs exist and are valid")
            return {
                "year": year,
                "month": month,
                "status": "SKIP_VERIFIED",
                "elapsed": time.time() - start_time,
                "completed_blocks": BLOCK_X * BLOCK_Y,
                "message": "",
            }

        (
            input_paths,
            input_datasets,
            xsize,
            ysize,
            gt,
            proj,
        ) = input_paths_and_alignment(year, month)

        (
            parts,
            output_datasets,
            output_bands,
            status,
        ) = open_or_create_parts(year, month, xsize, ysize, gt, proj)

        variance_records = read_variance_records(year, month)
        initial_completed = int(status.sum())

        month_log(
            year,
            month,
            f"[RESUME] completed={initial_completed}/{BLOCK_X * BLOCK_Y}",
        )

        progress = tqdm(
            total=BLOCK_X * BLOCK_Y,
            initial=initial_completed,
            desc=f"Region {REGION_ID} {year}-{month:02d}",
            position=0,
            leave=True,
        )

        for block_l in range(BLOCK_Y):
            row_changed = False

            yoff = (block_l * ysize) // BLOCK_Y
            yend = ((block_l + 1) * ysize) // BLOCK_Y
            out_ysize = yend - yoff

            for block_c in range(BLOCK_X):
                if status[block_l, block_c] == 1:
                    continue

                xoff = (block_c * xsize) // BLOCK_X
                xend = ((block_c + 1) * xsize) // BLOCK_X
                out_xsize = xend - xoff

                if out_xsize <= 0 or out_ysize <= 0:
                    status[block_l, block_c] = 1
                    row_changed = True
                    progress.update(1)
                    continue

                try:

                    variance_records.pop((block_c, block_l), None)

                    raw_arrays = []
                    for dataset, path in zip(input_datasets, input_paths):
                        array = dataset.ReadAsArray(
                            xoff=xoff,
                            yoff=yoff,
                            xsize=out_xsize,
                            ysize=out_ysize,
                        )
                        if array is None:
                            raise RuntimeError(
                                f"ReadAsArray returned None: {path}, "
                                f"c={block_c}, l={block_l}"
                            )
                        raw_arrays.append(array)

                    results, variance_ratio, block_state = calculate_block(
                        raw_arrays
                    )

                    for key, array in results.items():
                        output_bands[key].WriteArray(
                            array,
                            xoff=xoff,
                            yoff=yoff,
                        )

                    if variance_ratio is not None:
                        variance_records[(block_c, block_l)] = variance_ratio

                    status[block_l, block_c] = 1
                    row_changed = True
                    progress.update(1)

                    if block_state != "OK":
                        month_log(
                            year,
                            month,
                            f"[{block_state}] c={block_c} l={block_l}",
                        )

                    del raw_arrays, results

                except Exception as block_error:
                    month_log(
                        year,
                        month,
                        f"[BLOCK_ERROR] c={block_c} l={block_l} "
                        f"error={repr(block_error)}\n"
                        f"{traceback.format_exc()}",
                    )
                    continue

            if row_changed and CHECKPOINT_EACH_ROW:
                flush_outputs(output_bands, output_datasets)
                atomic_save_status(status_path(year, month), status)
                save_variance_records(
                    year,
                    month,
                    variance_records,
                    final=False,
                )

        progress.close()

        flush_outputs(output_bands, output_datasets)
        atomic_save_status(status_path(year, month), status)
        save_variance_records(
            year,
            month,
            variance_records,
            final=False,
        )

        completed_blocks = int(status.sum())


        close_outputs(output_bands, output_datasets)
        input_datasets = [None] * len(input_datasets)
        gc.collect()
        time.sleep(1.0)

        if completed_blocks != BLOCK_X * BLOCK_Y:
            message = (
                f"month remains partial: completed={completed_blocks}/"
                f"{BLOCK_X * BLOCK_Y}; rerun to retry failed blocks"
            )
            month_log(year, month, f"[PARTIAL] {message}")

            return {
                "year": year,
                "month": month,
                "status": "PARTIAL",
                "elapsed": time.time() - start_time,
                "completed_blocks": completed_blocks,
                "message": message,
            }

        install_finished_parts(year, month, parts, xsize, ysize)
        save_variance_records(
            year,
            month,
            variance_records,
            final=True,
        )


        for path in (
            status_path(year, month),
            variance_work_path(year, month),
        ):
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

        try:
            work_dir(year, month).rmdir()
        except Exception:
            pass

        month_log(year, month, "[PASS] full monthly outputs installed")

        return {
            "year": year,
            "month": month,
            "status": "PASS",
            "elapsed": time.time() - start_time,
            "completed_blocks": completed_blocks,
            "message": "",
        }

    except Exception as error:
        close_outputs(output_bands, output_datasets)
        input_datasets = [None] * len(input_datasets)
        gc.collect()

        message = f"{type(error).__name__}: {error}"
        month_log(
            year,
            month,
            f"[MONTH_ERROR] {message}\n{traceback.format_exc()}",
        )

        completed = 0
        try:
            completed = int(load_status(status_path(year, month)).sum())
        except Exception:
            pass

        return {
            "year": year,
            "month": month,
            "status": "ERROR",
            "elapsed": time.time() - start_time,
            "completed_blocks": completed,
            "message": message,
        }


def main():
    for folder in (
        WATER_FOLDER,
        PC1_FOLDER,
        VARIANCE_FOLDER,
        TEXTURE_FOLDER,
        LOG_FOLDER,
        WORK_FOLDER,
    ):
        mkdir(folder)

    master_log("=" * 90)
    master_log("START direct full-month water-index and texture pipeline")
    master_log(f"Region: {REGION_ID}")
    master_log(f"Input: {INPUT_FOLDER}")
    master_log(f"Years: {START_YEAR}-{END_YEAR}")
    master_log(f"MAX_WORKERS: {MAX_WORKERS}")
    master_log("=" * 90)

    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            f"Input folder does not exist or disk is not mounted: "
            f"{INPUT_FOLDER}"
        )

    tasks = [
        (year, month)
        for year in range(START_YEAR, END_YEAR + 1)
        for month in range(1, 13)
    ]

    results = []

    if MAX_WORKERS <= 1:
        for year, month in tasks:
            result = process_month(year, month)
            results.append(result)
            write_summary(
                result["year"],
                result["month"],
                result["status"],
                result["elapsed"],
                result["completed_blocks"],
                result["message"],
            )
    else:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(process_month, year, month): (year, month)
                for year, month in tasks
            }

            for future in as_completed(futures):
                year, month = futures[future]

                try:
                    result = future.result()
                except Exception as error:
                    result = {
                        "year": year,
                        "month": month,
                        "status": "PROCESS_ERROR",
                        "elapsed": 0.0,
                        "completed_blocks": 0,
                        "message": f"{type(error).__name__}: {error}",
                    }

                results.append(result)
                write_summary(
                    result["year"],
                    result["month"],
                    result["status"],
                    result["elapsed"],
                    result["completed_blocks"],
                    result["message"],
                )

    status_counts = {}
    for result in results:
        status_counts[result["status"]] = (
            status_counts.get(result["status"], 0) + 1
        )

    master_log("=" * 90)
    master_log(f"FINISHED. Status counts: {status_counts}")
    master_log(f"Summary: {SUMMARY_CSV}")
    master_log("=" * 90)


if __name__ == "__main__":
    main()
