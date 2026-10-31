# -*- coding: utf-8 -*-
r"""
QTP first-5-years vs last-5-years transition analysis.

Reads existing annual F/H rasters only; does NOT recalculate annual F/H.

Input root:
K:\Monthly surface water dynamics TIB\supervised classification\
step4_analysis\QTP_FH_results

Calculates:
- mean F/H for 2016-2020
- revised classification for 2016-2020
- mean F/H for 2021-2025
- revised classification for 2021-2025
- transition from 2016-2020 to 2021-2025
- transition summary and legend CSV

Revised classification:
0   Non-water
1   Perennial
2   Wet intermittent
3   Dry intermittent
255 NoData

Rules:
F == 0 -> Non-water
F > 0.92 and H < 0.12 -> Perennial
remaining F >= 0.50 -> Wet intermittent
remaining 0 < F < 0.50 -> Dry intermittent

No class 4.
"""

import os
import csv
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


ROOT = (
    r"K:\Monthly surface water dynamics TIB"
    r"\supervised classification"
    r"\step4_analysis"
    r"\QTP_FH_results"
)

OUTPUT_DIR = os.path.join(ROOT, "multi_year_mean_and_transition")

YEARS = list(range(2016, 2026))
FIRST5_YEARS = [2016, 2017, 2018, 2019, 2020]
LAST5_YEARS = [2021, 2022, 2023, 2024, 2025]

BLOCK_SIZE = 512
FLOAT_NODATA = -9999.0
CLASS_NODATA = 255

F_PERENNIAL = 0.92
H_THRESHOLD = 0.12
F_WET = 0.50

CLASS_NAMES = {
    0: "Non-water",
    1: "Perennial water",
    2: "Wet intermittent water",
    3: "Dry intermittent water",
}


def resolve_annual_file(year, variable):
    candidates = [
        os.path.join(ROOT, str(year), f"{variable}_{year}.tif"),
        os.path.join(ROOT, f"{variable}_{year}.tif"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Cannot find {variable}_{year}.tif.\n"
        "Tried:\n  " + "\n  ".join(candidates)
    )


def build_file_table():
    return {
        year: {
            "F": resolve_annual_file(year, "F"),
            "H": resolve_annual_file(year, "H"),
        }
        for year in YEARS
    }


def same_transform(a, b, atol=1e-12):
    return np.allclose(tuple(a), tuple(b), rtol=0.0, atol=atol)


def check_consistency(srcs):
    ref = srcs[YEARS[0]]["F"]

    for year in YEARS:
        for variable in ("F", "H"):
            src = srcs[year][variable]

            if src.width != ref.width or src.height != ref.height:
                raise ValueError(f"Raster size mismatch: {year} {variable}")

            if src.crs != ref.crs:
                raise ValueError(f"CRS mismatch: {year} {variable}")

            if not same_transform(src.transform, ref.transform):
                raise ValueError(f"Transform mismatch: {year} {variable}")

    return ref


def generate_windows(width, height, block_size):
    for row_off in range(0, height, block_size):
        win_h = min(block_size, height - row_off)

        for col_off in range(0, width, block_size):
            win_w = min(block_size, width - col_off)

            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=win_w,
                height=win_h
            )


def get_float_profile(ref):
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
    return profile


def get_class_profile(ref):
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
    return profile


def valid_annual_pair(F, H):
    return (
        np.isfinite(F)
        & np.isfinite(H)
        & (F != FLOAT_NODATA)
        & (H != FLOAT_NODATA)
        & (F >= 0.0)
        & (F <= 1.0)
        & (H >= 0.0)
        & (H <= 1.0)
    )


def calculate_period_mean(F_stack, H_stack, indices):
    shape = F_stack.shape[1:]

    sum_F = np.zeros(shape, dtype=np.float64)
    sum_H = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.uint8)

    for idx in indices:
        F = F_stack[idx]
        H = H_stack[idx]
        valid = valid_annual_pair(F, H)

        sum_F[valid] += F[valid]
        sum_H[valid] += H[valid]
        count[valid] += 1

    mean_F = np.full(shape, FLOAT_NODATA, dtype=np.float32)
    mean_H = np.full(shape, FLOAT_NODATA, dtype=np.float32)

    valid_period = count > 0

    mean_F[valid_period] = (
        sum_F[valid_period] / count[valid_period]
    ).astype(np.float32)

    mean_H[valid_period] = (
        sum_H[valid_period] / count[valid_period]
    ).astype(np.float32)

    return mean_F, mean_H


def classify_revised(F, H):
    cls = np.full(F.shape, CLASS_NODATA, dtype=np.uint8)

    valid = (
        np.isfinite(F)
        & np.isfinite(H)
        & (F != FLOAT_NODATA)
        & (H != FLOAT_NODATA)
        & (F >= 0.0)
        & (F <= 1.0)
        & (H >= 0.0)
        & (H <= 1.0)
    )

    nonwater = valid & np.isclose(F, 0.0, rtol=0.0, atol=1e-7)
    cls[nonwater] = 0

    perennial = valid & (F > F_PERENNIAL) & (H < H_THRESHOLD)
    cls[perennial] = 1

    remaining_water = valid & (~nonwater) & (~perennial)

    cls[remaining_water & (F >= F_WET)] = 2
    cls[remaining_water & (F > 0.0) & (F < F_WET)] = 3

    return cls


def calculate_transition(first_class, last_class):
    transition = np.full(
        first_class.shape,
        CLASS_NODATA,
        dtype=np.uint8
    )

    valid = (
        (first_class != CLASS_NODATA)
        & (last_class != CLASS_NODATA)
    )

    transition[valid] = (
        first_class[valid].astype(np.uint16) * 10
        + last_class[valid].astype(np.uint16)
    ).astype(np.uint8)

    return transition


def get_output_paths():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    return {
        "first_F": os.path.join(OUTPUT_DIR, "F_mean_2016_2020.tif"),
        "first_H": os.path.join(OUTPUT_DIR, "H_mean_2016_2020.tif"),
        "first_C": os.path.join(
            OUTPUT_DIR,
            "Class_mean_2016_2020_revised.tif"
        ),
        "last_F": os.path.join(OUTPUT_DIR, "F_mean_2021_2025.tif"),
        "last_H": os.path.join(OUTPUT_DIR, "H_mean_2021_2025.tif"),
        "last_C": os.path.join(
            OUTPUT_DIR,
            "Class_mean_2021_2025_revised.tif"
        ),
        "transition": os.path.join(
            OUTPUT_DIR,
            "Transition_2016_2020_to_2021_2025_revised.tif"
        ),
        "summary": os.path.join(
            OUTPUT_DIR,
            "transition_summary_2016_2020_to_2021_2025.csv"
        ),
        "legend": os.path.join(
            OUTPUT_DIR,
            "transition_code_legend_2016_2020_to_2021_2025.csv"
        ),
    }


def write_transition_legend(path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow([
            "transition_code",
            "start_class",
            "start_name",
            "end_class",
            "end_name",
            "meaning"
        ])

        for start in range(4):
            for end in range(4):
                code = start * 10 + end

                writer.writerow([
                    code,
                    start,
                    CLASS_NAMES[start],
                    end,
                    CLASS_NAMES[end],
                    f"{CLASS_NAMES[start]} -> {CLASS_NAMES[end]}"
                ])

        writer.writerow([
            255, "", "NoData", "", "NoData",
            "NoData at first or last period"
        ])


def write_transition_summary(path, counts):
    valid_total = sum(
        n for code, n in counts.items()
        if code != CLASS_NODATA
    )

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow([
            "transition_code",
            "start_class",
            "start_name",
            "end_class",
            "end_name",
            "pixel_count",
            "percent_of_valid_pixels"
        ])

        for code in sorted(counts):
            n = counts[code]

            if code == CLASS_NODATA:
                writer.writerow([
                    255, "", "NoData", "", "NoData", n, ""
                ])
                continue

            start = code // 10
            end = code % 10

            if start not in CLASS_NAMES or end not in CLASS_NAMES:
                continue

            pct = (
                n / valid_total * 100.0
                if valid_total > 0
                else 0.0
            )

            writer.writerow([
                code,
                start,
                CLASS_NAMES[start],
                end,
                CLASS_NAMES[end],
                n,
                f"{pct:.6f}"
            ])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    annual_files = build_file_table()

    print("=" * 72)
    print("QTP 2016-2020 vs 2021-2025 transition analysis")
    print("=" * 72)

    srcs = {}

    try:
        for year in YEARS:
            srcs[year] = {
                "F": rasterio.open(annual_files[year]["F"]),
                "H": rasterio.open(annual_files[year]["H"]),
            }

        ref = check_consistency(srcs)

        width = ref.width
        height = ref.height
        paths = get_output_paths()

        float_profile = get_float_profile(ref)
        class_profile = get_class_profile(ref)

        first_indices = [YEARS.index(y) for y in FIRST5_YEARS]
        last_indices = [YEARS.index(y) for y in LAST5_YEARS]

        n_rows = (height + BLOCK_SIZE - 1) // BLOCK_SIZE
        n_cols = (width + BLOCK_SIZE - 1) // BLOCK_SIZE
        total_blocks = n_rows * n_cols

        transition_counts = defaultdict(int)

        with (
            rasterio.open(paths["first_F"], "w", **float_profile) as out_first_F,
            rasterio.open(paths["first_H"], "w", **float_profile) as out_first_H,
            rasterio.open(paths["first_C"], "w", **class_profile) as out_first_C,
            rasterio.open(paths["last_F"], "w", **float_profile) as out_last_F,
            rasterio.open(paths["last_H"], "w", **float_profile) as out_last_H,
            rasterio.open(paths["last_C"], "w", **class_profile) as out_last_C,
            rasterio.open(
                paths["transition"],
                "w",
                **class_profile
            ) as out_transition
        ):

            windows = generate_windows(
                width,
                height,
                BLOCK_SIZE
            )

            for win in tqdm(
                windows,
                total=total_blocks,
                desc="Processing",
                unit="block"
            ):
                rows = int(win.height)
                cols = int(win.width)

                F_stack = np.empty(
                    (len(YEARS), rows, cols),
                    dtype=np.float32
                )

                H_stack = np.empty(
                    (len(YEARS), rows, cols),
                    dtype=np.float32
                )

                for i, year in enumerate(YEARS):
                    F_stack[i] = srcs[year]["F"].read(
                        1,
                        window=win
                    )

                    H_stack[i] = srcs[year]["H"].read(
                        1,
                        window=win
                    )

                first_F, first_H = calculate_period_mean(
                    F_stack,
                    H_stack,
                    first_indices
                )

                last_F, last_H = calculate_period_mean(
                    F_stack,
                    H_stack,
                    last_indices
                )

                first_C = classify_revised(first_F, first_H)
                last_C = classify_revised(last_F, last_H)

                transition = calculate_transition(
                    first_C,
                    last_C
                )

                values, nums = np.unique(
                    transition,
                    return_counts=True
                )

                for value, n in zip(values, nums):
                    transition_counts[int(value)] += int(n)

                out_first_F.write(first_F, 1, window=win)
                out_first_H.write(first_H, 1, window=win)
                out_first_C.write(first_C, 1, window=win)

                out_last_F.write(last_F, 1, window=win)
                out_last_H.write(last_H, 1, window=win)
                out_last_C.write(last_C, 1, window=win)

                out_transition.write(
                    transition,
                    1,
                    window=win
                )

        write_transition_legend(paths["legend"])
        write_transition_summary(
            paths["summary"],
            transition_counts
        )

        print("\nDONE")
        print(f"Output folder:\n{OUTPUT_DIR}")
        print("\nTransition:")
        print(paths["transition"])

    finally:
        for year in srcs:
            for variable in srcs[year]:
                try:
                    srcs[year][variable].close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
