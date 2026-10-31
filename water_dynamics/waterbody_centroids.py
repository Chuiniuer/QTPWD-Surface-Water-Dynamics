# -*- coding: utf-8 -*-
r"""
QTP waterbody polygon -> centroid point
Minimal fast version using GDAL/OGR only.

Input:
E:\phd_l3\QTPWD_vector\<year>\QTP_waterbody_<year>.gdb
Layer:
WB_YYYY_MM

Output:
K:\Monthly surface water dynamics TIB\supervised classification\step4_analysis
\QTP_waterbody_points

Each output point keeps only:
- Area_km2
- Point geometry

Notes:
- Uses ST_Centroid(Shape)
- No ArcGIS/arcpy
- No PointOnSurface
- No intersection
- No area recalculation
- No spatial index
- One GeoPackage per month
- Existing output files are skipped
"""

from pathlib import Path
import time
from osgeo import gdal

gdal.UseExceptions()

# 1. Paths

INPUT_ROOT = Path(r"E:\phd_l3\QTPWD_vector")

OUTPUT_ROOT = Path(
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step4_analysis\QTP_waterbody_points"
)

YEARS = range(2016, 2026)
MONTHS = range(1, 13)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# 2. Main

print("=" * 80)
print("QTP polygon -> centroid point")
print("Keep only: Area_km2 + point geometry")
print("No spatial index / no area calculation / no intersection")
print("=" * 80)

for year in YEARS:

    gdb = (
        INPUT_ROOT
        / str(year)
        / f"QTP_waterbody_{year}.gdb"
    )

    if not gdb.exists():
        print(f"[MISSING GDB] {gdb}")
        continue

    src_ds = gdal.OpenEx(
        str(gdb),
        gdal.OF_VECTOR | gdal.OF_READONLY
    )

    if src_ds is None:
        print(f"[ERROR] Cannot open: {gdb}")
        continue

    print()
    print(f"===== {year} =====")

    for month in MONTHS:

        src_layer_name = f"WB_{year}_{month:02d}"

        src_lyr = src_ds.GetLayerByName(
            src_layer_name
        )

        # fallback for names without leading zero
        if src_lyr is None:
            src_layer_name = f"WB_{year}_{month}"
            src_lyr = src_ds.GetLayerByName(
                src_layer_name
            )

        if src_lyr is None:
            print(
                f"[MISSING] {year}-{month:02d}"
            )
            continue

        # Check Area_km2 field
        defn = src_lyr.GetLayerDefn()

        if defn.GetFieldIndex("Area_km2") < 0:
            print(
                f"[ERROR] {src_layer_name} "
                f"missing Area_km2"
            )
            continue

        out_file = (
            OUTPUT_ROOT
            / f"QTP_waterbody_points_{year}_{month:02d}.gpkg"
        )

        out_layer_name = (
            f"points_{year}_{month:02d}"
        )

        # completed output -> skip
        if out_file.exists():
            print(
                f"[SKIP] {year}-{month:02d}"
            )
            continue

        print(
            f"[RUN ] {year}-{month:02d}",
            flush=True
        )

        t0 = time.time()

        # FileGDB geometry field is Shape in your data.
        sql = (
            f'SELECT '
            f'"Area_km2", '
            f'ST_Centroid("Shape") AS geometry '
            f'FROM "{src_layer_name}"'
        )

        try:
            options = gdal.VectorTranslateOptions(
                format="GPKG",
                SQLStatement=sql,
                SQLDialect="SQLite",
                layerName=out_layer_name,
                geometryType="POINT",
                layerCreationOptions=[
                    "SPATIAL_INDEX=NO"
                ]
            )

            out_ds = gdal.VectorTranslate(
                str(out_file),
                src_ds,
                options=options
            )

            if out_ds is None:
                raise RuntimeError(
                    "gdal.VectorTranslate returned None"
                )

            out_ds = None

            elapsed_min = (
                time.time() - t0
            ) / 60.0

            print(
                f"[DONE] {year}-{month:02d} | "
                f"{elapsed_min:.2f} min"
            )

        except Exception as e:

            print(
                f"[ERROR] {year}-{month:02d} | "
                f"{type(e).__name__}: {e}"
            )

            # Remove incomplete output
            if out_file.exists():
                try:
                    out_file.unlink()
                except Exception:
                    pass

    src_ds = None


print()
print("=" * 80)
print("Finished")
print(f"Output: {OUTPUT_ROOT}")
print("=" * 80)
