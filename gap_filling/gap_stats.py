# -*- coding: utf-8 -*-

"""Count missing pixels by block and month."""

from pathlib import Path
import math
import csv
import time

import numpy as np
from osgeo import gdal
from tqdm import tqdm


INPUT_ROOT = Path(
    r"K:\Monthly surface water dynamics TIB"
    r"\supervised classification"
    r"\step2_semantic_segmentation"
    r"\inference"
    r"\QTP_water_mask_clipped"
)

QTP_MASK_PATH = (
    INPUT_ROOT
    / "_QTP_fixed_mask.tif"
)

OUTPUT_ROOT = Path(
    r"K:\Monthly surface water dynamics TIB"
    r"\supervised classification"
    r"\step3_gap_filling"
    r"\QTP_gap_statistics"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


START_YEAR = 2016
END_YEAR = 2025

MONTH_KEYS = [
    (year, month)
    for year in range(
        START_YEAR,
        END_YEAR + 1,
    )
    for month in range(
        1,
        13,
    )
]

N_MONTHS = len(
    MONTH_KEYS
)


BLOCK_SIZE = 1024

SCAN_TILE_SIZE = 8192

GAP = np.uint8(2)

GDAL_CACHE_GB = 8


gdal.UseExceptions()

gdal.SetCacheMax(
    GDAL_CACHE_GB
    * 1024
    * 1024
    * 1024
)

gdal.SetConfigOption(
    "GDAL_NUM_THREADS",
    "ALL_CPUS",
)


def find_month_file(
    year,
    month,
):

    candidates = [
        INPUT_ROOT
        / (
            f"QTP_water_mask_"
            f"{year}_"
            f"{month:02d}_clip.tif"
        ),

        INPUT_ROOT
        / (
            f"QTP_water_mask_"
            f"{year}_"
            f"{month:02d}.tif"
        ),
    ]

    for path in candidates:

        if path.exists():
            return path

    return None


def build_catalog():

    catalog = {}

    for key in MONTH_KEYS:

        path = find_month_file(
            *key
        )

        if path is None:

            raise FileNotFoundError(
                f"Missing month: "
                f"{key[0]}-"
                f"{key[1]:02d}"
            )

        catalog[
            key
        ] = path

    return catalog


def raster_info(
    path,
):

    ds = gdal.Open(
        str(path),
        gdal.GA_ReadOnly,
    )

    if ds is None:

        raise RuntimeError(
            f"Cannot open:\n{path}"
        )

    result = (
        ds.RasterXSize,
        ds.RasterYSize,
        tuple(
            ds.GetGeoTransform()
        ),
        ds.GetProjection(),
    )

    ds = None

    return result


def count_gap_per_block(
    gap_mask,
):

    """Count missing pixels within each processing block."""

    h, w = (
        gap_mask.shape
    )

    ny = math.ceil(
        h / BLOCK_SIZE
    )

    nx = math.ceil(
        w / BLOCK_SIZE
    )

    padded_h = (
        ny * BLOCK_SIZE
    )

    padded_w = (
        nx * BLOCK_SIZE
    )

    if (
        padded_h != h
        or padded_w != w
    ):

        padded = np.zeros(
            (
                padded_h,
                padded_w,
            ),
            dtype=np.uint8,
        )

        padded[
            :h,
            :w
        ] = gap_mask.astype(
            np.uint8,
            copy=False,
        )

    else:

        padded = gap_mask.astype(
            np.uint8,
            copy=False,
        )

    view = padded.reshape(
        ny,
        BLOCK_SIZE,
        nx,
        BLOCK_SIZE,
    )

    counts = view.sum(
        axis=(1, 3),
        dtype=np.uint32,
    )

    return counts


def main():

    print(
        "\n"
        + "#" * 100
    )

    print(
        "QTP GAP DENSITY STATISTICS"
    )

    print(
        "#" * 100
    )

    catalog = build_catalog()

    reference = raster_info(
        catalog[
            MONTH_KEYS[0]
        ]
    )

    width = reference[0]
    height = reference[1]

    print(
        f"Raster size: "
        f"{width:,} × "
        f"{height:,}"
    )

    print(
        f"Pixels/month: "
        f"{width * height:,}"
    )

    n_block_x = math.ceil(
        width
        / BLOCK_SIZE
    )

    n_block_y = math.ceil(
        height
        / BLOCK_SIZE
    )

    n_blocks = (
        n_block_x
        * n_block_y
    )

    print(
        f"1024 blocks: "
        f"{n_blocks:,}"
    )

    # Open datasets

    mask_ds = gdal.Open(
        str(QTP_MASK_PATH),
        gdal.GA_ReadOnly,
    )

    if mask_ds is None:

        raise RuntimeError(
            f"Cannot open:\n"
            f"{QTP_MASK_PATH}"
        )

    mask_band = (
        mask_ds.GetRasterBand(1)
    )

    month_datasets = {}
    month_bands = {}

    for key in MONTH_KEYS:

        ds = gdal.Open(
            str(
                catalog[
                    key
                ]
            ),
            gdal.GA_ReadOnly,
        )

        if ds is None:

            raise RuntimeError(
                f"Cannot open:\n"
                f"{catalog[key]}"
            )

        month_datasets[
            key
        ] = ds

        month_bands[
            key
        ] = (
            ds.GetRasterBand(1)
        )

    # Matrix:
    # rows = spatial blocks
    # cols = months

    gap_counts = np.zeros(
        (
            n_blocks,
            N_MONTHS,
        ),
        dtype=np.uint32,
    )

    # Large tiles

    scan_nx = math.ceil(
        width
        / SCAN_TILE_SIZE
    )

    scan_ny = math.ceil(
        height
        / SCAN_TILE_SIZE
    )

    total_tiles = (
        scan_nx
        * scan_ny
    )

    print(
        f"Large scan tiles: "
        f"{total_tiles}"
    )

    started = time.time()

    pbar = tqdm(
        total=total_tiles,
        desc="Gap statistics",
    )

    for sy in range(
        scan_ny
    ):

        yoff = (
            sy
            * SCAN_TILE_SIZE
        )

        h = min(
            SCAN_TILE_SIZE,
            height - yoff,
        )

        for sx in range(
            scan_nx
        ):

            xoff = (
                sx
                * SCAN_TILE_SIZE
            )

            w = min(
                SCAN_TILE_SIZE,
                width - xoff,
            )

            # QTP mask

            qtp = mask_band.ReadAsArray(
                xoff,
                yoff,
                w,
                h,
            )

            if qtp is None:

                raise RuntimeError(
                    "QTP mask read failed."
                )

            inside_qtp = (
                qtp != 0
            )

            # Global block offsets

            global_block_y0 = (
                yoff // BLOCK_SIZE
            )

            global_block_x0 = (
                xoff // BLOCK_SIZE
            )

            local_ny = math.ceil(
                h / BLOCK_SIZE
            )

            local_nx = math.ceil(
                w / BLOCK_SIZE
            )

            if not np.any(
                inside_qtp
            ):

                del qtp
                del inside_qtp

                pbar.update(1)

                continue

            # Months

            for month_index, key in enumerate(
                MONTH_KEYS
            ):

                arr = month_bands[
                    key
                ].ReadAsArray(
                    xoff,
                    yoff,
                    w,
                    h,
                )

                if arr is None:

                    raise RuntimeError(
                        f"Read failed: "
                        f"{key}"
                    )

                gap_mask = (
                    (arr == GAP)
                    & inside_qtp
                )

                if not np.any(
                    gap_mask
                ):

                    del arr
                    del gap_mask

                    continue

                local_counts = (
                    count_gap_per_block(
                        gap_mask
                    )
                )


                for local_by in range(
                    local_ny
                ):

                    global_by = (
                        global_block_y0
                        + local_by
                    )

                    if (
                        global_by
                        >= n_block_y
                    ):
                        continue

                    for local_bx in range(
                        local_nx
                    ):

                        global_bx = (
                            global_block_x0
                            + local_bx
                        )

                        if (
                            global_bx
                            >= n_block_x
                        ):
                            continue

                        block_id = (
                            global_by
                            * n_block_x
                            + global_bx
                        )

                        gap_counts[
                            block_id,
                            month_index,
                        ] = (
                            local_counts[
                                local_by,
                                local_bx,
                            ]
                        )

                del arr
                del gap_mask
                del local_counts

            del qtp
            del inside_qtp

            pbar.update(1)

    pbar.close()

    # Close input

    mask_band = None
    mask_ds = None

    month_bands.clear()
    month_datasets.clear()

    # Save raw count matrix

    np.savez_compressed(
        OUTPUT_ROOT
        / "gap_counts_block_month.npz",

        gap_counts=gap_counts,

        block_size=np.int32(
            BLOCK_SIZE
        ),

        width=np.int64(
            width
        ),

        height=np.int64(
            height
        ),

        n_block_x=np.int32(
            n_block_x
        ),

        n_block_y=np.int32(
            n_block_y
        ),
    )

    # Global active block-month values

    positive = (
        gap_counts[
            gap_counts > 0
        ]
    )

    total_combinations = (
        n_blocks
        * N_MONTHS
    )

    active_combinations = int(
        positive.size
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "GLOBAL GAP STATISTICS"
    )

    print(
        "=" * 100
    )

    print(
        f"block × month combinations: "
        f"{total_combinations:,}"
    )

    print(
        f"Active combinations: "
        f"{active_combinations:,}"
    )

    print(
        f"Zero-gap combinations: "
        f"{total_combinations - active_combinations:,}"
    )

    print(
        f"Active ratio: "
        f"{100 * active_combinations / total_combinations:.2f}%"
    )

    if positive.size == 0:

        print(
            "No QTP internal gap found."
        )

        return

    # Percentiles

    percentiles = [
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
        99.5,
        99.9,
    ]

    values = np.percentile(
        positive,
        percentiles,
    )

    print(
        "\nGap pixels per ACTIVE block-month:"
    )

    for p, value in zip(
        percentiles,
        values,
    ):

        print(
            f"P{p:<5} = "
            f"{value:,.1f}"
        )

    print(
        f"Mean   = "
        f"{positive.mean():,.1f}"
    )

    print(
        f"Max    = "
        f"{positive.max():,}"
    )

    # Density

    full_block_pixels = (
        BLOCK_SIZE
        * BLOCK_SIZE
    )

    median = np.percentile(
        positive,
        50,
    )

    p90 = np.percentile(
        positive,
        90,
    )

    p95 = np.percentile(
        positive,
        95,
    )

    print(
        "\nEquivalent percentage of "
        "a full 1024×1024 block:"
    )

    print(
        f"Median = "
        f"{100 * median / full_block_pixels:.5f}%"
    )

    print(
        f"P90    = "
        f"{100 * p90 / full_block_pixels:.5f}%"
    )

    print(
        f"P95    = "
        f"{100 * p95 / full_block_pixels:.5f}%"
    )

    # Threshold counts

    thresholds = [
        1,
        10,
        50,
        100,
        250,
        500,
        1000,
        2000,
        5000,
        10000,
        25000,
        50000,
        100000,
        250000,
        500000,
    ]

    print(
        "\nCumulative distribution:"
    )

    distribution_rows = []

    for threshold in thresholds:

        n = int(
            np.count_nonzero(
                positive
                <= threshold
            )
        )

        pct = (
            100
            * n
            / positive.size
        )

        print(
            f"gap <= "
            f"{threshold:>7,}: "
            f"{n:>10,} "
            f"({pct:6.2f}%)"
        )

        distribution_rows.append(
            (
                threshold,
                n,
                pct,
            )
        )

    # Suggested sparse thresholds

    for threshold in [
        500,
        1000,
        2000,
        5000,
        10000,
    ]:

        sparse_n = np.count_nonzero(
            positive
            <= threshold
        )

        print(
            f"\nIf SPARSE_LIMIT={threshold:,}:"
        )

        print(
            f"  sparse combinations = "
            f"{sparse_n:,} "
            f"({100*sparse_n/positive.size:.2f}%)"
        )

        print(
            f"  dense combinations  = "
            f"{positive.size-sparse_n:,} "
            f"({100*(positive.size-sparse_n)/positive.size:.2f}%)"
        )

    # Monthly CSV

    monthly_csv = (
        OUTPUT_ROOT
        / "monthly_gap_statistics.csv"
    )

    with open(
        monthly_csv,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "year",
                "month",
                "total_gap_pixels",
                "active_blocks",
                "mean_gap_per_active_block",
                "median_gap_per_active_block",
                "p90_gap_per_active_block",
                "p95_gap_per_active_block",
                "max_gap_per_block",
            ]
        )

        for month_index, key in enumerate(
            MONTH_KEYS
        ):

            month_counts = (
                gap_counts[
                    :,
                    month_index,
                ]
            )

            positive_month = (
                month_counts[
                    month_counts > 0
                ]
            )

            total_gap = int(
                month_counts.sum(
                    dtype=np.uint64
                )
            )

            active_blocks = int(
                positive_month.size
            )

            if active_blocks:

                mean_gap = (
                    positive_month.mean()
                )

                median_gap = (
                    np.percentile(
                        positive_month,
                        50,
                    )
                )

                p90_gap = (
                    np.percentile(
                        positive_month,
                        90,
                    )
                )

                p95_gap = (
                    np.percentile(
                        positive_month,
                        95,
                    )
                )

                max_gap = int(
                    positive_month.max()
                )

            else:

                mean_gap = 0
                median_gap = 0
                p90_gap = 0
                p95_gap = 0
                max_gap = 0

            writer.writerow(
                [
                    key[0],
                    key[1],
                    total_gap,
                    active_blocks,
                    mean_gap,
                    median_gap,
                    p90_gap,
                    p95_gap,
                    max_gap,
                ]
            )

    # Threshold CSV

    threshold_csv = (
        OUTPUT_ROOT
        / "gap_count_distribution.csv"
    )

    with open(
        threshold_csv,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "gap_threshold",
                "n_active_block_months_le_threshold",
                "percentage",
            ]
        )

        for row in distribution_rows:

            writer.writerow(
                row
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
        f"Finished in "
        f"{elapsed / 60:.1f} min"
    )

    print(
        "Saved:"
    )

    print(
        OUTPUT_ROOT
        / "gap_counts_block_month.npz"
    )

    print(
        monthly_csv
    )

    print(
        threshold_csv
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":

    main()