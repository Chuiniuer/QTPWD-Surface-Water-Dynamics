# -*- coding: utf-8 -*-

"""
QTP annual surface-water F/H and dynamic-type calculation
2016-2025

Input monthly raster values:
    0 = non-water
    1 = water
    2 = invalid, excluded from F/H calculation

Outputs for each year:
    F_YEAR.tif       float32, nodata = -9999
    H_YEAR.tif       float32, nodata = -9999
    Class_YEAR.tif   uint8,   nodata = 255

Class codes:
    0   = Non-water
    1   = Perennial water
    2   = Wet intermittent water
    3   = Dry intermittent water
    254 = Valid but not matched by strict F-H thresholds
    255 = NoData

Classification thresholds:
    Perennial:
        F > 0.92 and H < 0.12

    Wet intermittent:
        F >= 0.50 and H >= 0.12

    Dry intermittent:
        0 < F < 0.50 and H >= 0.12
"""

import os
import numpy as np
import rasterio

from rasterio.windows import Window
from tqdm import tqdm
from numba import njit, prange


INPUT_ROOT = (
    r"K:\Monthly surface water dynamics TIB"
    r"\supervised classification"
    r"\step3_gap_filling"
    r"\QTP_STGF_gap_filled_final_4proc"
)

OUTPUT_ROOT = r"K:\QTP_FH_results"

YEARS = list(range(2016, 2026))

# Recommended for very large rasters.
# 512 usually gives a reasonable memory/speed balance.
BLOCK_SIZE = 512

# Input coding
INPUT_INVALID = 2

# Output nodata
FLOAT_NODATA = -9999.0
CLASS_NODATA = 255

# Valid pixel but does not fall into any of the strict RSE classes
CLASS_UNCLASSIFIED = 254


def get_year_files(year):

    files = []

    for month in range(1, 13):

        path = os.path.join(
            INPUT_ROOT,
            str(year),
            f"month{month:02d}",
            f"QTP_water_mask_STGF_{year}_{month:02d}.tif"
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing monthly raster:\n{path}"
            )

        files.append(path)

    return files


def check_rasters(srcs, files):

    ref = srcs[0]

    for i, src in enumerate(srcs[1:], start=1):

        if src.width != ref.width or src.height != ref.height:
            raise ValueError(
                "Raster dimensions mismatch:\n"
                f"Reference: {files[0]}\n"
                f"Problem:   {files[i]}"
            )

        if src.transform != ref.transform:
            raise ValueError(
                "Raster transform mismatch:\n"
                f"{files[i]}"
            )

        if src.crs != ref.crs:
            raise ValueError(
                "Raster CRS mismatch:\n"
                f"{files[i]}"
            )


def calculate_F(stack):
    """
    stack shape:
        (12, rows, cols)

    Values:
        0 = non-water
        1 = water
        2 = invalid

    Invalid observations are excluded.

    F = number of water observations / number of valid observations
    """

    valid = stack != INPUT_INVALID
    water = stack == 1

    valid_count = np.sum(valid, axis=0)
    water_count = np.sum(water, axis=0)

    F = np.full(
        valid_count.shape,
        FLOAT_NODATA,
        dtype=np.float32
    )

    mask = valid_count > 0

    F[mask] = (
        water_count[mask].astype(np.float32)
        /
        valid_count[mask].astype(np.float32)
    )

    return F


@njit(parallel=True, cache=True)
def calculate_H(stack):
    """
    Heterogeneity index:

        H = - sum(p_j * log(p_j)) / log(N)

    where each segment j is a continuous run of equal valid states,
    and:

        p_j = segment_length / N

    Values == 2 are removed before calculating H.

    If:
        no valid observations -> nodata
        only one valid observation -> H = 0

    The latter is mathematically treated as no observable transition.
    """

    T, rows, cols = stack.shape

    output = np.full(
        (rows, cols),
        FLOAT_NODATA,
        dtype=np.float32
    )

    total_pixels = rows * cols

    for idx in prange(total_pixels):

        r = idx // cols
        c = idx % cols

        # Count valid observations

        n = 0

        for t in range(T):

            value = stack[t, r, c]

            if value == 0 or value == 1:
                n += 1

        # No valid observation
        if n == 0:
            continue

        # Only one valid observation:
        # no temporal transition can be measured.
        if n == 1:
            output[r, c] = 0.0
            continue

        # Calculate segment lengths WITHOUT allocating seq

        first_found = False

        previous_value = 0
        segment_length = 0

        entropy = 0.0

        for t in range(T):

            value = stack[t, r, c]

            # completely ignore value 2
            if value != 0 and value != 1:
                continue

            if not first_found:

                previous_value = value
                segment_length = 1
                first_found = True

            else:

                if value == previous_value:

                    segment_length += 1

                else:

                    p = segment_length / n

                    entropy -= p * np.log(p)

                    previous_value = value
                    segment_length = 1

        # Last segment
        if segment_length > 0:

            p = segment_length / n

            entropy -= p * np.log(p)

        output[r, c] = entropy / np.log(n)

    return output


def classify_water(F, H):
    """
    Output codes:

        0   non-water
        1   perennial water
        2   wet intermittent water
        3   dry intermittent water
        254 valid but not matched by strict thresholds
        255 nodata
    """

    cls = np.full(
        F.shape,
        CLASS_NODATA,
        dtype=np.uint8
    )

    valid = (
        (F != FLOAT_NODATA)
        &
        (H != FLOAT_NODATA)
    )

    # All valid cells initially marked as valid-but-unclassified.
    cls[valid] = CLASS_UNCLASSIFIED

    # 0. Non-water

    cls[
        valid
        &
        (F == 0)
    ] = 0

    # 1. Perennial water

    cls[
        valid
        &
        (F > 0.92)
        &
        (H < 0.12)
    ] = 1

    # 2. Wet intermittent water

    cls[
        valid
        &
        (F >= 0.50)
        &
        (H >= 0.12)
    ] = 2

    # 3. Dry intermittent water

    cls[
        valid
        &
        (F > 0)
        &
        (F < 0.50)
        &
        (H >= 0.12)
    ] = 3

    return cls


def create_float_output(path, ref):

    profile = ref.profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=FLOAT_NODATA,
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="YES"
    )

    with rasterio.open(path, "w", **profile):
        pass


def create_class_output(path, ref):

    profile = ref.profile.copy()

    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=CLASS_NODATA,
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="YES"
    )

    with rasterio.open(path, "w", **profile):
        pass


def generate_windows(width, height, block_size):

    for row_off in range(0, height, block_size):

        win_height = min(
            block_size,
            height - row_off
        )

        for col_off in range(0, width, block_size):

            win_width = min(
                block_size,
                width - col_off
            )

            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=win_width,
                height=win_height
            )


def process_year(year):

    print("\n" + "=" * 70)
    print(f"Processing year: {year}")
    print("=" * 70)

    files = get_year_files(year)

    # Open the 12 monthly rasters only once

    srcs = [
        rasterio.open(path)
        for path in files
    ]

    try:

        check_rasters(srcs, files)

        ref = srcs[0]

        width = ref.width
        height = ref.height

        # Output directory

        out_dir = os.path.join(
            OUTPUT_ROOT,
            str(year)
        )

        os.makedirs(
            out_dir,
            exist_ok=True
        )

        F_path = os.path.join(
            out_dir,
            f"F_{year}.tif"
        )

        H_path = os.path.join(
            out_dir,
            f"H_{year}.tif"
        )

        C_path = os.path.join(
            out_dir,
            f"Class_{year}.tif"
        )

        # Create outputs independently

        if not os.path.exists(F_path):
            create_float_output(F_path, ref)

        if not os.path.exists(H_path):
            create_float_output(H_path, ref)

        if not os.path.exists(C_path):
            create_class_output(C_path, ref)

        # Number of windows

        n_rows = (
            height + BLOCK_SIZE - 1
        ) // BLOCK_SIZE

        n_cols = (
            width + BLOCK_SIZE - 1
        ) // BLOCK_SIZE

        total_windows = n_rows * n_cols

        print(f"Raster size : {width} × {height}")
        print(f"Block size  : {BLOCK_SIZE} × {BLOCK_SIZE}")
        print(f"Total blocks: {total_windows}")

        # Open output datasets

        with (
            rasterio.open(F_path, "r+") as dst_F,
            rasterio.open(H_path, "r+") as dst_H,
            rasterio.open(C_path, "r+") as dst_C
        ):

            windows = generate_windows(
                width,
                height,
                BLOCK_SIZE
            )

            for win in tqdm(
                windows,
                total=total_windows,
                desc=f"{year}",
                unit="block"
            ):

                # Read 12 months

                stack = np.empty(
                    (
                        12,
                        int(win.height),
                        int(win.width)
                    ),
                    dtype=np.uint8
                )

                for i, src in enumerate(srcs):

                    stack[i] = src.read(
                        1,
                        window=win
                    )

                # F

                F = calculate_F(stack)

                # H

                H = calculate_H(stack)

                # Class

                water_class = classify_water(
                    F,
                    H
                )

                # Write results

                dst_F.write(
                    F,
                    1,
                    window=win
                )

                dst_H.write(
                    H,
                    1,
                    window=win
                )

                dst_C.write(
                    water_class,
                    1,
                    window=win
                )

        print(f"\nFinished {year}")
        print(f"F     : {F_path}")
        print(f"H     : {H_path}")
        print(f"Class : {C_path}")

    finally:

        for src in srcs:
            src.close()


if __name__ == "__main__":

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True
    )

    print("=" * 70)
    print("QTP annual F / H / water-type calculation")
    print("=" * 70)

    print("\nInput coding:")
    print("  0   = non-water")
    print("  1   = water")
    print("  2   = invalid")

    print("\nOutput class coding:")
    print("  0   = non-water")
    print("  1   = perennial water")
    print("  2   = wet intermittent water")
    print("  3   = dry intermittent water")
    print("  254 = valid but unclassified by strict thresholds")
    print("  255 = nodata")

    print("\nYears:")
    print(YEARS)

    # Compile numba once using a tiny dummy array
    # Prevent the first real block from appearing to hang.

    print("\nCompiling H function...")

    dummy = np.zeros(
        (12, 2, 2),
        dtype=np.uint8
    )

    _ = calculate_H(dummy)

    print("Numba compilation completed.")

    # Process annual data only

    for year in YEARS:

        process_year(year)

    print("\n" + "=" * 70)
    print("ALL YEARS FINISHED")
    print("=" * 70)
