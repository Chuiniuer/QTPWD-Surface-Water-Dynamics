# -*- coding: utf-8 -*-
"""Aggregate aligned 10 m water masks to a fixed 25 km grid."""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

INPUT_ROOT = Path(
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step3_gap_filling\QTPWD_alberts_equal_area"
)

OUTPUT_ROOT = Path(
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step4_analysis\QTP_water_area_25km_block"
)

MONTHLY_DIR = OUTPUT_ROOT / "monthly"
TEMPLATE_PATH = OUTPUT_ROOT / "QTP_25km_grid_template.tif"
LOG_PATH = OUTPUT_ROOT / "QTP_25km_processing_log.csv"

SOURCE_RES_M = 10.0
TARGET_RES_M = 25_000.0
FACTOR = int(round(TARGET_RES_M / SOURCE_RES_M))  # 2500

WATER_VALUE = 1
NONWATER_VALUE = 0
INVALID_VALUE = 2

PIXEL_AREA_KM2 = (SOURCE_RES_M * SOURCE_RES_M) / 1_000_000.0  # 0.0001 km²
SKIP_COMPLETED = True

EXPECTED_YEARS = range(2016, 2026)

DATE_PATTERNS = [
    re.compile(r"(?<!\d)(20(?:1[6-9]|2[0-5]))[_-](0[1-9]|1[0-2])(?!\d)"),
    re.compile(r"(?<!\d)(20(?:1[6-9]|2[0-5]))(0[1-9]|1[0-2])(?!\d)"),
]


def extract_year_month(path: Path):
    for pat in DATE_PATTERNS:
        m = pat.search(path.stem)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def discover_monthly_rasters():
    found = {}
    for p in INPUT_ROOT.rglob("*.tif"):
        ym = extract_year_month(p)
        if ym is None:
            continue
        if ym in found:
            raise RuntimeError(
                f"Duplicate monthly raster for {ym[0]}-{ym[1]:02d}:\n"
                f"  {found[ym]}\n"
                f"  {p}"
            )
        found[ym] = p

    if not found:
        raise FileNotFoundError(f"No monthly GeoTIFFs found under:\n{INPUT_ROOT}")

    return dict(sorted(found.items()))


def same_crs(wkt1: str, wkt2: str) -> bool:
    s1 = osr.SpatialReference()
    s2 = osr.SpatialReference()
    s1.ImportFromWkt(wkt1)
    s2.ImportFromWkt(wkt2)
    return bool(s1.IsSame(s2))


def get_reference_info(path: Path):
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open reference raster:\n{path}")

    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    if not proj:
        raise RuntimeError("Reference raster has no CRS.")
    if abs(gt[2]) > 1e-12 or abs(gt[4]) > 1e-12:
        raise RuntimeError("Rotated source raster is not supported.")

    xres = abs(gt[1])
    yres = abs(gt[5])

    if abs(xres - SOURCE_RES_M) > 0.05 or abs(yres - SOURCE_RES_M) > 0.05:
        raise RuntimeError(
            f"Expected ~10 m source pixels, got {xres} x {yres} m."
        )

    width = ds.RasterXSize
    height = ds.RasterYSize

    out_width = (width + FACTOR - 1) // FACTOR
    out_height = (height + FACTOR - 1) // FACTOR

    # Fixed grid anchored to the upper-left corner of the first source raster.
    out_gt = (
        gt[0],
        TARGET_RES_M,
        0.0,
        gt[3],
        0.0,
        -TARGET_RES_M,
    )

    info = {
        "projection": proj,
        "src_gt": gt,
        "src_width": width,
        "src_height": height,
        "out_gt": out_gt,
        "out_width": out_width,
        "out_height": out_height,
    }

    ds = None
    return info


def validate_source(path: Path, ref):
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open:\n{path}")

    if ds.RasterXSize != ref["src_width"] or ds.RasterYSize != ref["src_height"]:
        raise RuntimeError(
            f"Raster dimensions differ from reference:\n{path}"
        )

    if not same_crs(ds.GetProjection(), ref["projection"]):
        raise RuntimeError(f"CRS differs from reference:\n{path}")

    gt = ds.GetGeoTransform()
    for i, (a, b) in enumerate(zip(gt, ref["src_gt"])):
        if abs(a - b) > 1e-6:
            raise RuntimeError(
                f"Pixel alignment differs at geotransform element {i}:\n{path}"
            )

    ds = None


def create_template(ref):
    if TEMPLATE_PATH.exists():
        return

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        str(TEMPLATE_PATH),
        ref["out_width"],
        ref["out_height"],
        1,
        gdal.GDT_Float32,
        options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
    )

    ds.SetProjection(ref["projection"])
    ds.SetGeoTransform(ref["out_gt"])

    b = ds.GetRasterBand(1)
    b.SetNoDataValue(np.nan)
    b.Fill(np.nan)
    b.SetDescription("Fixed_25km_grid_template")

    ds.FlushCache()
    ds = None


def aggregate_month(src_path: Path, out_path: Path, ref):
    src = gdal.Open(str(src_path), gdal.GA_ReadOnly)
    sb = src.GetRasterBand(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    drv = gdal.GetDriverByName("GTiff")
    dst = drv.Create(
        str(out_path),
        ref["out_width"],
        ref["out_height"],
        2,
        gdal.GDT_Float32,
        options=[
            "TILED=YES",
            "COMPRESS=LZW",
            "BIGTIFF=IF_SAFER",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
        ],
    )

    dst.SetProjection(ref["projection"])
    dst.SetGeoTransform(ref["out_gt"])

    b_area = dst.GetRasterBand(1)
    b_valid = dst.GetRasterBand(2)

    # Explicit NaN NoData.
    b_area.SetNoDataValue(np.nan)
    b_valid.SetNoDataValue(np.nan)

    b_area.SetDescription("WaterArea_km2")
    b_valid.SetDescription("ValidFraction")

    area = np.full(
        (ref["out_height"], ref["out_width"]),
        np.nan,
        dtype=np.float32,
    )
    valid_fraction = np.full_like(area, np.nan)

    for out_y in range(ref["out_height"]):
        yoff = out_y * FACTOR
        ysize = min(FACTOR, ref["src_height"] - yoff)

        for out_x in range(ref["out_width"]):
            xoff = out_x * FACTOR
            xsize = min(FACTOR, ref["src_width"] - xoff)

            arr = sb.ReadAsArray(xoff, yoff, xsize, ysize)
            if arr is None:
                raise RuntimeError(
                    f"Read failed at x={xoff}, y={yoff}, "
                    f"size={xsize}x{ysize}\n{src_path}"
                )

            water_count = np.count_nonzero(arr == WATER_VALUE)
            nonwater_count = np.count_nonzero(arr == NONWATER_VALUE)
            valid_count = water_count + nonwater_count
            covered_count = arr.size

            # All-invalid output cell -> NaN.
            if valid_count == 0:
                continue

            # Partial-invalid cells retain observed water area only.
            area[out_y, out_x] = water_count * PIXEL_AREA_KM2
            valid_fraction[out_y, out_x] = valid_count / covered_count

    b_area.WriteArray(area)
    b_valid.WriteArray(valid_fraction)

    dst.SetMetadata({
        "SOURCE_RESOLUTION_M": str(SOURCE_RES_M),
        "TARGET_RESOLUTION_M": str(TARGET_RES_M),
        "AGGREGATION_FACTOR": str(FACTOR),
        "SOURCE_WATER_VALUE": str(WATER_VALUE),
        "SOURCE_NONWATER_VALUE": str(NONWATER_VALUE),
        "SOURCE_INVALID_VALUE": str(INVALID_VALUE),
        "PIXEL_AREA_KM2": str(PIXEL_AREA_KM2),
        "AGGREGATION_METHOD": "Direct integer block aggregation",
        "NODATA": "NaN",
        "INVALID_POLICY":
            "All-invalid coarse cell -> NaN; partial-invalid cell retains "
            "observed water area and reports ValidFraction.",
        "WATER_AREA_NOTE":
            "Observed water area only; no extrapolation for invalid pixels.",
    })

    b_area.FlushCache()
    b_valid.FlushCache()
    dst.FlushCache()

    src = None
    dst = None


def quick_qc(out_path: Path):
    ds = gdal.Open(str(out_path), gdal.GA_ReadOnly)

    area = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    valid = ds.GetRasterBand(2).ReadAsArray().astype(np.float64)

    result = {
        "Total_water_area_km2": float(np.nansum(area)),
        "Valid_cells": int(np.count_nonzero(~np.isnan(area))),
        "NaN_cells": int(np.count_nonzero(np.isnan(area))),
        "Min_valid_fraction": float(np.nanmin(valid)),
        "Mean_valid_fraction": float(np.nanmean(valid)),
        "Max_valid_fraction": float(np.nanmax(valid)),
    }

    ds = None
    return result


def main():
    print("=" * 84)
    print("QTP FAST 10 m -> 25 km direct block aggregation")
    print("=" * 84)
    print(f"Full output cell = {FACTOR} x {FACTOR} source pixels")
    print(f"One 10 m water pixel = {PIXEL_AREA_KM2:.7f} km2")
    print("All-invalid 25 km cell -> NaN")

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input folder not found:\n{INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

    monthly = discover_monthly_rasters()

    expected = {(y, m) for y in EXPECTED_YEARS for m in range(1, 13)}
    missing = sorted(expected - set(monthly))

    print(f"Found monthly rasters: {len(monthly)}")
    if missing:
        print("WARNING - missing months:")
        print(", ".join(f"{y}-{m:02d}" for y, m in missing))

    first_ym, first_path = next(iter(monthly.items()))
    ref = get_reference_info(first_path)
    create_template(ref)

    print(f"Reference: {first_ym[0]}-{first_ym[1]:02d}")
    print(f"Source size: {ref['src_width']} x {ref['src_height']}")
    print(f"25 km grid: {ref['out_width']} x {ref['out_height']}")
    print(f"Template: {TEMPLATE_PATH}")

    rows = []

    for i, ((year, month), src_path) in enumerate(monthly.items(), 1):
        out_dir = MONTHLY_DIR / str(year)
        out_path = out_dir / f"QTP_water_area_25km_{year}_{month:02d}.tif"

        started = datetime.now()

        if SKIP_COMPLETED and out_path.exists():
            status = "SKIPPED"
            error = ""
            qc = {}
            print(
                f"[{i:03d}/{len(monthly):03d}] "
                f"{year}-{month:02d} SKIPPED"
            )
        else:
            try:
                validate_source(src_path, ref)
                aggregate_month(src_path, out_path, ref)
                qc = quick_qc(out_path)

                status = "DONE"
                error = ""

                print(
                    f"[{i:03d}/{len(monthly):03d}] "
                    f"{year}-{month:02d} DONE | "
                    f"water={qc['Total_water_area_km2']:.2f} km2 | "
                    f"NaN_cells={qc['NaN_cells']} | "
                    f"mean_valid={qc['Mean_valid_fraction']:.4f}"
                )

            except Exception as e:
                status = "ERROR"
                error = repr(e)
                qc = {}

                print(
                    f"[{i:03d}/{len(monthly):03d}] "
                    f"{year}-{month:02d} ERROR"
                )
                print(error)

        rows.append({
            "Year": year,
            "Month": month,
            "Input": str(src_path),
            "Output": str(out_path),
            "Status": status,
            "Total_water_area_km2": qc.get("Total_water_area_km2", ""),
            "Valid_cells": qc.get("Valid_cells", ""),
            "NaN_cells": qc.get("NaN_cells", ""),
            "Min_valid_fraction": qc.get("Min_valid_fraction", ""),
            "Mean_valid_fraction": qc.get("Mean_valid_fraction", ""),
            "Max_valid_fraction": qc.get("Max_valid_fraction", ""),
            "Started": started.isoformat(timespec="seconds"),
            "Finished": datetime.now().isoformat(timespec="seconds"),
            "Error": error,
        })

        with open(LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print("\nFinished.")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Monthly rasters: {MONTHLY_DIR}")
    print(f"Processing log: {LOG_PATH}")


if __name__ == "__main__":
    main()
