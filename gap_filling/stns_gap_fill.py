# -*- coding: utf-8 -*-

"""Fill monthly water-mask gaps using spatiotemporal similarity."""

# IMPORTANT: constrain nested native threading BEFORE importing numpy/OpenCV

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
import gc
import math
import shutil
import time
import traceback
import multiprocessing as mp

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from osgeo import gdal
from tqdm import tqdm


INPUT_ROOT = Path(
    r"K:\Monthly surface water dynamics TIB"
    r"\supervised classification"
    r"\step2_semantic_segmentation"
    r"\inference"
    r"\QTP_water_mask_clipped"
)

OUTPUT_ROOT = Path(
    r"K:\Monthly surface water dynamics TIB"
    r"\supervised classification"
    r"\step3_gap_filling"
    r"\QTP_STGF_gap_filled_final_4proc"
)

QTP_MASK_PATH = INPUT_ROOT / "_QTP_fixed_mask.tif"

GAP_STAT_ROOT = Path(
    r"K:\Monthly surface water dynamics TIB"
    r"\supervised classification"
    r"\step3_gap_filling"
    r"\QTP_gap_statistics"
)

GAP_COUNT_PATH = GAP_STAT_ROOT / "gap_counts_block_month.npz"

STACK_VRT_PATH = OUTPUT_ROOT / "_QTP_120month_stack.vrt"

LOG_PATH = OUTPUT_ROOT / "STGF_4proc_log.txt"
ERROR_LOG = OUTPUT_ROOT / "STGF_4proc_errors.txt"


START_YEAR = 2016
END_YEAR = 2025

YEARS = list(range(START_YEAR, END_YEAR + 1))

MONTH_KEYS = [
    (year, month)
    for year in YEARS
    for month in range(1, 13)
]

N_MONTHS = len(MONTH_KEYS)

KEY_TO_INDEX = {
    key: index
    for index, key in enumerate(MONTH_KEYS)
}

YEAR_TO_INDEX = {
    year: index
    for index, year in enumerate(YEARS)
}


K = 4
SPATIAL_YEAR_RANGE = 3
TEMPORAL_YEAR_SPAN = 3
TEMPORAL_MONTH_SPAN = 2

R = 30

WINDOW = 2 * R + 1
WINDOW_AREA = np.float32(WINDOW * WINDOW)
TEMPORAL_LENGTH = 2 * TEMPORAL_MONTH_SPAN + 1

EPSILON = np.float32(0.001)

LAND = np.uint8(0)
WATER = np.uint8(1)
GAP = np.uint8(2)


N_WORKERS = 4

PROCESS_BLOCK_SIZE = 1024

# Based on the measured QTP block-month gap distribution.
SPARSE_LIMIT = 5000

# Output initialization.
OVERWRITE_OUTPUTS = False

# VRT is tiny; rebuilding it guarantees current source paths.
REBUILD_STACK_VRT = True

# Native threads PER PROCESS.
# Keep these small because there are already 4 worker processes.
GDAL_THREADS_PER_WORKER = "1"
OPENCV_THREADS_PER_WORKER = 1

# Persist worker progress every N completed active blocks.
# Smaller = safer after abrupt interruption, but more flush overhead.
CHECKPOINT_EVERY = 5

# Flush progress bar print / checkpoint details.
WORKER_PROGRESS_POSITION_OFFSET = 1

# GDAL cache PER PROCESS.
# Four workers * 2 GB = up to ~8 GB cache total.
GDAL_CACHE_GB_PER_WORKER = 2


def now_string():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def append_log(message):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{now_string()}] {message}\n")


def append_error(message):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"[{now_string()}]\n")
        f.write(message)
        f.write("\n")


def output_path(key):
    year, month = key
    return (
        OUTPUT_ROOT
        / str(year)
        / f"month{month:02d}"
        / f"QTP_water_mask_STGF_{year}_{month:02d}.tif"
    )


def worker_progress_path(worker_id):
    return OUTPUT_ROOT / f"_worker_{worker_id}_block_progress.npy"


def find_input(year, month):

    candidates = [
        INPUT_ROOT / f"QTP_water_mask_{year}_{month:02d}_clip.tif",
        INPUT_ROOT / f"QTP_water_mask_{year}_{month:02d}.tif",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def build_catalog():

    catalog = {}

    for key in MONTH_KEYS:
        path = find_input(*key)
        if path is not None:
            catalog[key] = path

    return catalog


def raster_info(path):

    ds = gdal.Open(str(path), gdal.GA_ReadOnly)

    if ds is None:
        raise RuntimeError(f"Cannot open:\n{path}")

    result = (
        ds.RasterXSize,
        ds.RasterYSize,
        tuple(ds.GetGeoTransform()),
        ds.GetProjection(),
    )

    ds = None
    return result


def check_inputs(catalog):

    if len(catalog) != N_MONTHS:

        missing = [
            key for key in MONTH_KEYS
            if key not in catalog
        ]

        raise FileNotFoundError(
            f"Expected {N_MONTHS} monthly inputs. Missing: {missing}"
        )

    reference = raster_info(catalog[MONTH_KEYS[0]])

    print("\nChecking 120 monthly grids...")

    for key in tqdm(MONTH_KEYS, desc="Grid check"):

        current = raster_info(catalog[key])

        if current[:2] != reference[:2]:
            raise ValueError(f"Raster size mismatch: {key}")

        if not np.allclose(
            current[2],
            reference[2],
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError(f"GeoTransform mismatch: {key}")

        if current[3] != reference[3]:
            raise ValueError(f"Projection mismatch: {key}")

    mask_info = raster_info(QTP_MASK_PATH)

    if (
        mask_info[:2] != reference[:2]
        or not np.allclose(
            mask_info[2],
            reference[2],
            rtol=0,
            atol=1e-9,
        )
        or mask_info[3] != reference[3]
    ):
        raise ValueError(
            "QTP fixed mask does not match monthly raster grid."
        )

    return reference


def build_blocks(width, height):

    blocks = []

    n_block_x = math.ceil(width / PROCESS_BLOCK_SIZE)
    n_block_y = math.ceil(height / PROCESS_BLOCK_SIZE)

    for by in range(n_block_y):

        yoff = by * PROCESS_BLOCK_SIZE
        h = min(PROCESS_BLOCK_SIZE, height - yoff)

        for bx in range(n_block_x):

            xoff = bx * PROCESS_BLOCK_SIZE
            w = min(PROCESS_BLOCK_SIZE, width - xoff)

            blocks.append(
                (xoff, yoff, w, h)
            )

    return blocks, n_block_x, n_block_y


def add_halo(block, raster_width, raster_height):

    xoff, yoff, width, height = block

    x0 = max(0, xoff - R)
    y0 = max(0, yoff - R)

    x1 = min(
        raster_width,
        xoff + width + R,
    )

    y1 = min(
        raster_height,
        yoff + height + R,
    )

    return (
        x0,
        y0,
        x1 - x0,
        y1 - y0,
    )


def load_gap_counts(width, height, n_blocks):

    if not GAP_COUNT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find:\n{GAP_COUNT_PATH}"
        )

    data = np.load(GAP_COUNT_PATH)

    gap_counts = data["gap_counts"]

    expected_shape = (
        n_blocks,
        N_MONTHS,
    )

    if gap_counts.shape != expected_shape:
        raise ValueError(
            f"gap_counts shape mismatch: "
            f"{gap_counts.shape} != {expected_shape}"
        )

    if "width" in data.files:
        if int(data["width"]) != width:
            raise ValueError("Gap-statistics width mismatch.")

    if "height" in data.files:
        if int(data["height"]) != height:
            raise ValueError("Gap-statistics height mismatch.")

    return gap_counts


def build_stack_vrt(catalog):

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if (
        STACK_VRT_PATH.exists()
        and REBUILD_STACK_VRT
    ):
        STACK_VRT_PATH.unlink()

    if STACK_VRT_PATH.exists():
        return

    paths = [
        str(catalog[key])
        for key in MONTH_KEYS
    ]

    ds = gdal.BuildVRT(
        str(STACK_VRT_PATH),
        paths,
        options=gdal.BuildVRTOptions(
            separate=True,
        ),
    )

    if ds is None:
        raise RuntimeError(
            "Failed to build 120-band VRT."
        )

    ds.FlushCache()
    ds = None


def assign_months_to_workers(gap_counts):

    """
    Greedy workload balancing.

    Estimated workload per month = total number of QTP-internal gap pixels
    measured across all spatial blocks.

    Every month is assigned to exactly ONE worker, so output writing is safe.
    """

    month_load = gap_counts.sum(
        axis=0,
        dtype=np.uint64,
    )

    order = np.argsort(
        month_load
    )[::-1]

    groups = [
        []
        for _ in range(N_WORKERS)
    ]

    loads = [
        np.uint64(0)
        for _ in range(N_WORKERS)
    ]

    for month_index in order:

        worker_id = int(
            np.argmin(loads)
        )

        groups[
            worker_id
        ].append(
            int(month_index)
        )

        loads[
            worker_id
        ] += month_load[
            month_index
        ]

    for worker_id in range(N_WORKERS):

        groups[
            worker_id
        ].sort()

    return groups, loads, month_load


def prepare_outputs(catalog):

    """
    Fast file-level copy.

    Existing output files are left untouched when OVERWRITE_OUTPUTS=False.
    This is required for safe resume.
    """

    print("\nPreparing output TIFFs...")

    for key in tqdm(
        MONTH_KEYS,
        desc="Initialize outputs",
    ):

        src = catalog[key]
        dst = output_path(key)

        dst.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            dst.exists()
            and not OVERWRITE_OUTPUTS
        ):
            continue

        if dst.exists():
            dst.unlink()

        shutil.copy2(
            src,
            dst,
        )


def box_count(mask):

    return cv2.boxFilter(
        mask.astype(
            np.uint8,
            copy=False,
        ),
        cv2.CV_32S,
        (
            WINDOW,
            WINDOW,
        ),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )


def windows_at(array, rows, cols):

    padded = np.pad(
        array,
        R,
        constant_values=GAP,
    )

    view = sliding_window_view(
        padded,
        (
            WINDOW,
            WINDOW,
        ),
    )

    return view[
        rows,
        cols,
    ]


def build_prior_prefix(core_stack):

    height, width = core_stack.shape[1:]

    yearly_valid = np.zeros(
        (
            len(YEARS),
            height,
            width,
        ),
        dtype=np.uint8,
    )

    yearly_water = np.zeros_like(
        yearly_valid
    )

    for year_index in range(len(YEARS)):

        start = year_index * 12
        end = start + 12

        year_data = core_stack[
            start:end
        ]

        yearly_valid[
            year_index
        ] = np.sum(
            year_data != GAP,
            axis=0,
            dtype=np.uint8,
        )

        yearly_water[
            year_index
        ] = np.sum(
            year_data == WATER,
            axis=0,
            dtype=np.uint8,
        )

    valid_prefix = np.zeros(
        (
            len(YEARS) + 1,
            height,
            width,
        ),
        dtype=np.uint16,
    )

    water_prefix = np.zeros_like(
        valid_prefix
    )

    valid_prefix[1:] = np.cumsum(
        yearly_valid,
        axis=0,
        dtype=np.uint16,
    )

    water_prefix[1:] = np.cumsum(
        yearly_water,
        axis=0,
        dtype=np.uint16,
    )

    return valid_prefix, water_prefix


def prior_at(
    target_year,
    rows,
    cols,
    valid_prefix,
    water_prefix,
):

    year_index = YEAR_TO_INDEX[
        target_year
    ]

    start = max(
        0,
        year_index - K,
    )

    end = min(
        len(YEARS),
        year_index + K + 1,
    )

    valid = (
        valid_prefix[
            end,
            rows,
            cols,
        ]
        -
        valid_prefix[
            start,
            rows,
            cols,
        ]
    )

    water = (
        water_prefix[
            end,
            rows,
            cols,
        ]
        -
        water_prefix[
            start,
            rows,
            cols,
        ]
    )

    probability = np.zeros(
        rows.size,
        dtype=np.float32,
    )

    usable = (
        valid > 0
    )

    probability[
        usable
    ] = (
        water[
            usable
        ].astype(np.float32)
        /
        valid[
            usable
        ].astype(np.float32)
    )

    return probability, usable


SPATIAL_REF_INDICES = {}

for year in YEARS:

    SPATIAL_REF_INDICES[
        year
    ] = np.array(
        [
            index
            for index, key
            in enumerate(MONTH_KEYS)
            if (
                year - SPATIAL_YEAR_RANGE
                <= key[0]
                <= year + SPATIAL_YEAR_RANGE
            )
        ],
        dtype=np.int16,
    )


def spatial_scores(
    stack,
    current,
    year,
    rows,
    cols,
):

    refs = SPATIAL_REF_INDICES[
        year
    ]

    count = int(
        refs.size
    )

    water_sum = np.zeros(
        rows.size,
        dtype=np.float32,
    )

    land_sum = np.zeros_like(
        water_sum
    )

    if (
        rows.size == 0
        or count == 0
    ):
        return (
            water_sum + EPSILON,
            land_sum + EPSILON,
        )

    # Sparse route

    if rows.size <= SPARSE_LIMIT:

        target_windows_all = windows_at(
            current,
            rows,
            cols,
        )

        target_has_valid = np.any(
            target_windows_all != GAP,
            axis=(1, 2),
        )

        active = np.flatnonzero(
            target_has_valid
        )

        if active.size == 0:

            del target_windows_all

            return (
                water_sum + EPSILON,
                land_sum + EPSILON,
            )

        active_rows = rows[
            active
        ]

        active_cols = cols[
            active
        ]

        target_windows = target_windows_all[
            active
        ]

        target_windows_valid = (
            target_windows != GAP
        )

        del target_windows_all

        for index in refs:

            reference = stack[
                index
            ]

            center = reference[
                active_rows,
                active_cols,
            ]

            center_valid = (
                center != GAP
            )

            if not np.any(
                center_valid
            ):
                continue

            local_valid = np.flatnonzero(
                center_valid
            )

            ref_rows = active_rows[
                local_valid
            ]

            ref_cols = active_cols[
                local_valid
            ]

            reference_windows = windows_at(
                reference,
                ref_rows,
                ref_cols,
            )

            target_sub = target_windows[
                local_valid
            ]

            target_valid_sub = target_windows_valid[
                local_valid
            ]

            matches = (
                (
                    target_sub
                    == reference_windows
                )
                & target_valid_sub
                & (
                    reference_windows
                    != GAP
                )
            )

            similarity = (
                matches.sum(
                    axis=(1, 2),
                    dtype=np.int32,
                ).astype(np.float32)
                / WINDOW_AREA
            )

            global_positions = active[
                local_valid
            ]

            center_values = center[
                local_valid
            ]

            water_sum[
                global_positions
            ] += (
                similarity
                * (
                    center_values == WATER
                )
            )

            land_sum[
                global_positions
            ] += (
                similarity
                * (
                    center_values == LAND
                )
            )

            del reference_windows
            del target_sub
            del target_valid_sub
            del matches
            del similarity

        del target_windows
        del target_windows_valid

    # Dense route

    else:

        target_valid = (
            box_count(
                current != GAP
            )[
                rows,
                cols,
            ]
            > 0
        )

        active = np.flatnonzero(
            target_valid
        )

        if active.size == 0:

            return (
                water_sum + EPSILON,
                land_sum + EPSILON,
            )

        active_rows = rows[
            active
        ]

        active_cols = cols[
            active
        ]

        current_valid = (
            current != GAP
        )

        for index in refs:

            reference = stack[
                index
            ]

            center = reference[
                active_rows,
                active_cols,
            ]

            if not np.any(
                center != GAP
            ):
                continue

            match = (
                current_valid
                & (
                    reference != GAP
                )
                & (
                    current == reference
                )
            )

            similarity = (
                box_count(
                    match
                )[
                    active_rows,
                    active_cols,
                ].astype(np.float32)
                / WINDOW_AREA
            )

            water_sum[
                active
            ] += (
                similarity
                * (
                    center == WATER
                )
            )

            land_sum[
                active
            ] += (
                similarity
                * (
                    center == LAND
                )
            )

            del match
            del similarity

    return (
        water_sum
        / np.float32(count)
        + EPSILON,

        land_sum
        / np.float32(count)
        + EPSILON,
    )


def shifted_index(
    year,
    month,
    offset,
):

    number = (
        year * 12
        + month - 1
        + offset
    )

    key = (
        number // 12,
        number % 12 + 1,
    )

    return KEY_TO_INDEX.get(
        key
    )


def temporal_scores(
    stack,
    target,
    rows,
    cols,
):

    year, month = target

    # Preserve original edge handling while using the formal +/-3 year span.
    reference_years = [
        ref_year
        for ref_year in YEARS
        if (
            max(
                START_YEAR + 1,
                year - TEMPORAL_YEAR_SPAN,
            )
            <= ref_year
            <= min(
                END_YEAR - 1,
                year + TEMPORAL_YEAR_SPAN,
            )
        )
    ]

    water_sum = np.zeros(
        rows.size,
        dtype=np.float32,
    )

    land_sum = np.zeros_like(
        water_sum
    )

    if not reference_years:

        return (
            water_sum + EPSILON,
            land_sum + EPSILON,
        )

    target_sequence = np.full(
        (
            TEMPORAL_LENGTH,
            rows.size,
        ),
        GAP,
        dtype=np.uint8,
    )

    offsets = range(
        -TEMPORAL_MONTH_SPAN,
        TEMPORAL_MONTH_SPAN + 1,
    )

    for offset_index, offset in enumerate(
        offsets
    ):

        index = shifted_index(
            year,
            month,
            offset,
        )

        if index is not None:

            target_sequence[
                offset_index
            ] = stack[
                index,
                rows,
                cols,
            ]

    for ref_year in reference_years:

        center_index = KEY_TO_INDEX[
            (
                ref_year,
                month,
            )
        ]

        center = stack[
            center_index,
            rows,
            cols,
        ]

        center_valid = (
            center != GAP
        )

        if not np.any(
            center_valid
        ):
            continue

        valid_idx = np.flatnonzero(
            center_valid
        )

        reference_sequence = np.full(
            (
                TEMPORAL_LENGTH,
                valid_idx.size,
            ),
            GAP,
            dtype=np.uint8,
        )

        for offset_index, offset in enumerate(
            offsets
        ):

            index = shifted_index(
                ref_year,
                month,
                offset,
            )

            if index is not None:

                reference_sequence[
                    offset_index
                ] = stack[
                    index,
                    rows[
                        valid_idx
                    ],
                    cols[
                        valid_idx
                    ],
                ]

        target_sub = target_sequence[
            :,
            valid_idx,
        ]

        matches = (
            (
                target_sub
                == reference_sequence
            )
            & (
                target_sub != GAP
            )
            & (
                reference_sequence != GAP
            )
        )

        similarity = matches.mean(
            axis=0,
            dtype=np.float32,
        )

        water_sum[
            valid_idx
        ] += (
            similarity
            * (
                center[
                    valid_idx
                ]
                == WATER
            )
        )

        land_sum[
            valid_idx
        ] += (
            similarity
            * (
                center[
                    valid_idx
                ]
                == LAND
            )
        )

    count = np.float32(
        len(reference_years)
    )

    return (
        water_sum / count + EPSILON,
        land_sum / count + EPSILON,
    )


def fill_target(
    stack,
    target,
    core,
    qtp_core,
    valid_prefix,
    water_prefix,
):

    cy, cx, height, width = core

    target_index = KEY_TO_INDEX[
        target
    ]

    current = stack[
        target_index
    ]

    current_core = current[
        cy:
        cy + height,
        cx:
        cx + width,
    ]

    candidate = (
        (
            current_core == GAP
        )
        & (
            qtp_core != 0
        )
    )

    rows, cols = np.where(
        candidate
    )

    n_gap = int(
        rows.size
    )

    if n_gap == 0:

        return (
            None,
            0,
            0,
            0,
        )

    probability, usable = prior_at(
        target[0],
        rows,
        cols,
        valid_prefix,
        water_prefix,
    )

    if not np.any(
        usable
    ):

        return (
            None,
            n_gap,
            0,
            n_gap,
        )

    rows = rows[
        usable
    ]

    cols = cols[
        usable
    ]

    probability = probability[
        usable
    ]

    ext_rows = (
        rows + cy
    )

    ext_cols = (
        cols + cx
    )

    water_spatial, land_spatial = spatial_scores(
        stack,
        current,
        target[0],
        ext_rows,
        ext_cols,
    )

    water_temporal, land_temporal = temporal_scores(
        stack,
        target,
        ext_rows,
        ext_cols,
    )

    p_water = (
        water_spatial
        * water_temporal
        * probability
    )

    p_land = (
        land_spatial
        * land_temporal
        * (
            np.float32(1.0)
            - probability
        )
    )

    values = np.where(
        p_water > p_land,
        WATER,
        LAND,
    ).astype(
        np.uint8
    )

    output = current_core.copy()

    output[
        rows,
        cols,
    ] = values

    # QTP exterior is always preserved as gap/no-data value 2.
    output[
        qtp_core == 0
    ] = GAP

    filled = int(
        rows.size
    )

    remaining = (
        n_gap - filled
    )

    return (
        output,
        n_gap,
        filled,
        remaining,
    )


class WorkerOutputManager:

    def __init__(self, allowed_month_indices):

        self.allowed = set(
            int(x)
            for x in allowed_month_indices
        )

        self.datasets = {}
        self.bands = {}

    def band(self, month_index):

        month_index = int(
            month_index
        )

        if month_index not in self.allowed:
            raise RuntimeError(
                f"Worker attempted to write unassigned month index "
                f"{month_index}"
            )

        band = self.bands.get(
            month_index
        )

        if band is not None:
            return band

        key = MONTH_KEYS[
            month_index
        ]

        path = output_path(
            key
        )

        ds = gdal.Open(
            str(path),
            gdal.GA_Update,
        )

        if ds is None:
            raise RuntimeError(
                f"Cannot update:\n{path}"
            )

        band = ds.GetRasterBand(1)

        self.datasets[
            month_index
        ] = ds

        self.bands[
            month_index
        ] = band

        return band

    def flush(self):

        for band in self.bands.values():
            if band is not None:
                band.FlushCache()

        for ds in self.datasets.values():
            if ds is not None:
                ds.FlushCache()

    def close(self):

        self.flush()

        self.bands.clear()
        self.datasets.clear()

        gc.collect()


class WorkerMaskReader:

    def __init__(self):

        self.ds = gdal.Open(
            str(QTP_MASK_PATH),
            gdal.GA_ReadOnly,
        )

        if self.ds is None:
            raise RuntimeError(
                f"Cannot open QTP mask:\n{QTP_MASK_PATH}"
            )

        self.band = self.ds.GetRasterBand(1)

    def read(self, window):

        xoff, yoff, width, height = window

        arr = self.band.ReadAsArray(
            xoff,
            yoff,
            width,
            height,
        )

        if arr is None:
            raise RuntimeError(
                "QTP mask read failed."
            )

        return arr.astype(
            np.uint8,
            copy=False,
        )

    def close(self):

        self.band = None
        self.ds = None


def load_worker_progress(
    worker_id,
    n_blocks,
):

    path = worker_progress_path(
        worker_id
    )

    if path.exists():

        try:

            progress = np.load(
                path
            )

            if progress.shape == (
                n_blocks,
            ):
                return progress.astype(
                    bool
                )

        except Exception:
            # Corrupted progress file:
            # safest behavior is to recompute relevant blocks.
            pass

    return np.zeros(
        n_blocks,
        dtype=bool,
    )


def save_worker_progress(
    worker_id,
    progress,
):

    path = worker_progress_path(
        worker_id
    )

    temp = path.with_suffix(
        ".tmp.npy"
    )

    np.save(
        temp,
        progress,
    )

    os.replace(
        temp,
        path,
    )


def worker_main(
    worker_id,
    assigned_month_indices,
    width,
    height,
    n_blocks,
):

    """
    One safe worker.

    It writes ONLY months in assigned_month_indices.

    Resume safety:
    - a block is marked complete in memory after successful writes;
    - persisted progress is written only after outputs.flush();
    - if interrupted before checkpoint, those recent blocks are recomputed.
    """

    try:

        # Prevent nested oversubscription.

        gdal.UseExceptions()

        gdal.SetConfigOption(
            "GDAL_NUM_THREADS",
            GDAL_THREADS_PER_WORKER,
        )

        gdal.SetConfigOption(
            "CPL_NUM_THREADS",
            GDAL_THREADS_PER_WORKER,
        )

        gdal.SetCacheMax(
            GDAL_CACHE_GB_PER_WORKER
            * 1024
            * 1024
            * 1024
        )

        cv2.setUseOptimized(True)
        cv2.setNumThreads(
            OPENCV_THREADS_PER_WORKER
        )

        # Reconstruct blocks locally.

        blocks, _, _ = build_blocks(
            width,
            height,
        )

        if len(blocks) != n_blocks:
            raise RuntimeError(
                "Worker block count mismatch."
            )

        # Load gap statistics locally (~small relative to imagery).

        data = np.load(
            GAP_COUNT_PATH
        )

        gap_counts = data[
            "gap_counts"
        ]

        if gap_counts.shape != (
            n_blocks,
            N_MONTHS,
        ):
            raise RuntimeError(
                f"Worker {worker_id}: "
                f"gap_counts shape mismatch."
            )

        assigned_month_indices = np.array(
            assigned_month_indices,
            dtype=np.int16,
        )

        # Blocks relevant to THIS worker only.

        worker_active_mask = np.any(
            gap_counts[
                :,
                assigned_month_indices,
            ] > 0,
            axis=1,
        )

        active_block_ids = np.flatnonzero(
            worker_active_mask
        )

        progress = load_worker_progress(
            worker_id,
            n_blocks,
        )

        remaining_block_ids = [
            int(block_id)
            for block_id in active_block_ids
            if not progress[
                block_id
            ]
        ]

        month_labels = [
            (
                f"{MONTH_KEYS[int(i)][0]}-"
                f"{MONTH_KEYS[int(i)][1]:02d}"
            )
            for i in assigned_month_indices
        ]

        print(
            f"\n[Worker {worker_id}] "
            f"months={len(month_labels)}, "
            f"active_blocks={len(active_block_ids):,}, "
            f"remaining={len(remaining_block_ids):,}"
        )

        append_log(
            f"Worker {worker_id} start | "
            f"months={month_labels} | "
            f"active={len(active_block_ids)} | "
            f"remaining={len(remaining_block_ids)}"
        )

        # Open shared read-only data independently in each process.

        stack_ds = gdal.Open(
            str(STACK_VRT_PATH),
            gdal.GA_ReadOnly,
        )

        if stack_ds is None:
            raise RuntimeError(
                f"Worker {worker_id}: "
                f"cannot open 120-band VRT."
            )

        mask_reader = WorkerMaskReader()

        outputs = WorkerOutputManager(
            assigned_month_indices
        )

        checkpoint_counter = 0

        sparse_target_count = 0
        dense_target_count = 0

        started = time.time()

        try:

            iterator = tqdm(
                remaining_block_ids,
                desc=f"W{worker_id}",
                position=(
                    WORKER_PROGRESS_POSITION_OFFSET
                    + worker_id
                ),
                leave=True,
            )

            for block_id in iterator:

                block = blocks[
                    block_id
                ]

                (
                    xoff,
                    yoff,
                    block_width,
                    block_height,
                ) = block

                # Only the assigned target months that actually
                # contain QTP-interior gaps in this spatial block.
                local_counts = gap_counts[
                    block_id,
                    assigned_month_indices,
                ]

                local_active = np.flatnonzero(
                    local_counts > 0
                )

                if local_active.size == 0:

                    progress[
                        block_id
                    ] = True

                    continue

                target_month_indices = (
                    assigned_month_indices[
                        local_active
                    ]
                )

                # Read 120-month reference stack once per
                # worker-active spatial block.

                halo = add_halo(
                    block,
                    width,
                    height,
                )

                hx, hy, hw, hh = halo

                stack = stack_ds.ReadAsArray(
                    hx,
                    hy,
                    hw,
                    hh,
                )

                if stack is None:
                    raise RuntimeError(
                        f"Worker {worker_id}: "
                        f"120-band read failed "
                        f"at block {block_id}"
                    )

                stack = stack.astype(
                    np.uint8,
                    copy=False,
                )

                if (
                    stack.ndim != 3
                    or stack.shape[0] != N_MONTHS
                ):
                    raise RuntimeError(
                        f"Worker {worker_id}: "
                        f"unexpected stack shape {stack.shape}"
                    )

                cy = yoff - hy
                cx = xoff - hx

                core = (
                    cy,
                    cx,
                    block_height,
                    block_width,
                )

                qtp_core = mask_reader.read(
                    (
                        xoff,
                        yoff,
                        block_width,
                        block_height,
                    )
                )

                core_stack = stack[
                    :,
                    cy:
                    cy + block_height,
                    cx:
                    cx + block_width,
                ]

                # Original K=4 prior logic.
                (
                    valid_prefix,
                    water_prefix,
                ) = build_prior_prefix(
                    core_stack
                )

                # Process ONLY this worker's active target months.

                for month_index in target_month_indices:

                    month_index = int(
                        month_index
                    )

                    target = MONTH_KEYS[
                        month_index
                    ]

                    measured_gap_count = int(
                        gap_counts[
                            block_id,
                            month_index,
                        ]
                    )

                    if (
                        measured_gap_count
                        <= SPARSE_LIMIT
                    ):
                        sparse_target_count += 1
                    else:
                        dense_target_count += 1

                    (
                        result,
                        n_gap,
                        filled,
                        remaining,
                    ) = fill_target(
                        stack=stack,
                        target=target,
                        core=core,
                        qtp_core=qtp_core,
                        valid_prefix=valid_prefix,
                        water_prefix=water_prefix,
                    )

                    if (
                        result is not None
                        and filled > 0
                    ):

                        outputs.band(
                            month_index
                        ).WriteArray(
                            result,
                            xoff,
                            yoff,
                        )

                    del result

                # This block was successfully computed.
                # It is NOT persisted yet.

                progress[
                    block_id
                ] = True

                checkpoint_counter += 1

                del stack
                del core_stack
                del qtp_core
                del valid_prefix
                del water_prefix

                # SAFE CHECKPOINT:
                # Crash before this -> blocks recomputed next run.
                # Crash after this  -> TIFF data are already flushed.

                if (
                    checkpoint_counter
                    >= CHECKPOINT_EVERY
                ):

                    outputs.flush()

                    save_worker_progress(
                        worker_id,
                        progress,
                    )

                    checkpoint_counter = 0

            # Final safe checkpoint.
            outputs.flush()

            save_worker_progress(
                worker_id,
                progress,
            )

        finally:

            outputs.close()
            mask_reader.close()
            stack_ds = None

            gc.collect()

        elapsed = (
            time.time()
            - started
        )

        append_log(
            f"Worker {worker_id} DONE | "
            f"elapsed_h={elapsed/3600:.3f} | "
            f"sparse_targets={sparse_target_count} | "
            f"dense_targets={dense_target_count}"
        )

        return {
            "worker_id": worker_id,
            "status": "success",
            "elapsed_hours": elapsed / 3600,
            "sparse_targets": sparse_target_count,
            "dense_targets": dense_target_count,
            "remaining_started": len(
                remaining_block_ids
            ),
        }

    except Exception as error:

        message = (
            f"WORKER {worker_id} FAILED\n"
            f"{repr(error)}\n\n"
            f"{traceback.format_exc()}"
        )

        append_error(
            message
        )

        return {
            "worker_id": worker_id,
            "status": "failed",
            "error": repr(error),
        }


def validate_outputs(reference):

    print(
        "\nValidating final 120 outputs..."
    )

    for key in tqdm(
        MONTH_KEYS,
        desc="Final validation",
    ):

        path = output_path(
            key
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing output:\n{path}"
            )

        current = raster_info(
            path
        )

        if (
            current[:2] != reference[:2]
            or not np.allclose(
                current[2],
                reference[2],
                rtol=0,
                atol=1e-9,
            )
            or current[3] != reference[3]
        ):
            raise ValueError(
                f"Final grid mismatch:\n{path}"
            )

    print(
        "All 120 final output grids: PASS"
    )


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        + "#" * 100
    )

    print(
        "QTP STGF - SAFE 4-PROCESS + AUTO-RESUME"
    )

    print(
        "#" * 100
    )

    print(
        "Formal parameters:"
    )

    print(
        f"K={K}, "
        f"spatial_year={SPATIAL_YEAR_RANGE}, "
        f"temporal_year={TEMPORAL_YEAR_SPAN}, "
        f"temporal_month={TEMPORAL_MONTH_SPAN}, "
        f"R={R}"
    )

    print(
        f"\nProcesses: {N_WORKERS}"
    )

    print(
        f"OpenCV threads / process: "
        f"{OPENCV_THREADS_PER_WORKER}"
    )

    print(
        f"GDAL threads / process: "
        f"{GDAL_THREADS_PER_WORKER}"
    )

    print(
        f"Checkpoint every "
        f"{CHECKPOINT_EVERY} active blocks"
    )

    # Input discovery / alignment.

    catalog = build_catalog()

    print(
        f"\nFound months: "
        f"{len(catalog)}"
    )

    reference = check_inputs(
        catalog
    )

    (
        width,
        height,
        transform,
        projection,
    ) = reference

    print(
        f"\nRaster: "
        f"{width:,} x {height:,}"
    )

    blocks, n_block_x, n_block_y = build_blocks(
        width,
        height,
    )

    n_blocks = len(
        blocks
    )

    print(
        f"Processing blocks: "
        f"{n_blocks:,}"
    )

    # Load measured gap workload.

    gap_counts = load_gap_counts(
        width=width,
        height=height,
        n_blocks=n_blocks,
    )

    positive = gap_counts[
        gap_counts > 0
    ]

    print(
        f"Active block-months: "
        f"{positive.size:,}"
    )

    if positive.size:

        sparse_count = int(
            np.count_nonzero(
                positive <= SPARSE_LIMIT
            )
        )

        print(
            f"SPARSE_LIMIT={SPARSE_LIMIT:,}: "
            f"{100*sparse_count/positive.size:.2f}% "
            f"of active block-months use sparse route."
        )

    # Build read-only stack VRT before spawning workers.

    print(
        "\nBuilding 120-band VRT..."
    )

    build_stack_vrt(
        catalog
    )

    # Initialize outputs before spawning workers.
    # On resume, existing TIFFs are preserved.

    prepare_outputs(
        catalog
    )

    # Workload-balanced unique month groups.

    (
        worker_groups,
        worker_loads,
        month_loads,
    ) = assign_months_to_workers(
        gap_counts
    )

    print(
        "\nWorker month assignment "
        "(unique months, load-balanced):"
    )

    for worker_id, group in enumerate(
        worker_groups
    ):

        labels = [
            (
                f"{MONTH_KEYS[i][0]}-"
                f"{MONTH_KEYS[i][1]:02d}"
            )
            for i in group
        ]

        print(
            f"Worker {worker_id}: "
            f"{len(group)} months | "
            f"estimated gap pixels="
            f"{int(worker_loads[worker_id]):,}"
        )

        print(
            "  "
            + ", ".join(labels)
        )

    append_log(
        "Starting 4-process run. "
        f"worker_loads="
        f"{[int(x) for x in worker_loads]}"
    )

    # Free main-process reference before spawn.
    del positive
    del month_loads

    # gap_counts is no longer needed in main after assignment;
    # workers load the NPZ independently.
    del gap_counts

    gc.collect()

    # Spawn exactly four safe workers.

    ctx = mp.get_context(
        "spawn"
    )

    args = [
        (
            worker_id,
            worker_groups[
                worker_id
            ],
            width,
            height,
            n_blocks,
        )
        for worker_id in range(
            N_WORKERS
        )
    ]

    started = time.time()

    with ctx.Pool(
        processes=N_WORKERS
    ) as pool:

        results = pool.starmap(
            worker_main,
            args,
        )

    elapsed = (
        time.time()
        - started
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "WORKER RESULTS"
    )

    print(
        "=" * 100
    )

    failed = []

    for result in results:

        print(
            result
        )

        if result.get(
            "status"
        ) != "success":

            failed.append(
                result
            )

    if failed:

        raise RuntimeError(
            "One or more workers failed. "
            "Rerun the SAME script after fixing the cause; "
            "completed worker/block checkpoints will be reused.\n"
            f"{failed}"
        )

    print(
        f"\n4-process elapsed: "
        f"{elapsed / 3600:.2f} h"
    )

    # Final metadata validation.

    validate_outputs(
        reference
    )

    append_log(
        f"ALL DONE | "
        f"elapsed_h={elapsed/3600:.3f}"
    )

    print(
        "\n"
        + "#" * 100
    )

    print(
        "ALL DONE"
    )

    print(
        "#" * 100
    )

    print(
        "Output:"
    )

    print(
        OUTPUT_ROOT
    )

    print(
        "\nResume files:"
    )

    for worker_id in range(
        N_WORKERS
    ):

        print(
            worker_progress_path(
                worker_id
            )
        )

    print(
        "\nIf interrupted:"
    )

    print(
        "Simply rerun this exact script. "
        "Existing TIFFs are preserved and each worker resumes "
        "from its last safely flushed checkpoint."
    )


if __name__ == "__main__":

    mp.freeze_support()

    # Spawn is the safe multiprocessing method on Windows.
    main()
