# -*- coding: utf-8 -*-
"""
QTP monthly waterbody statistics (2016–2025)

FINAL DEFINITIONS
-----------------
Total surface-water area:
    All detected connected water patches are included.

Waterbody count:
    Only connected waterbodies with Area_km2 >= 0.1 are counted.

Size-class counts:
    Small count : 0.1 <= A < 1 km2
    Medium count: 1 <= A < 100 km2
    Large count : A >= 100 km2

Size-class areas:
    Small area  : A < 1 km2      (includes <0.1 km2 patches)
    Medium area : 1 <= A < 100 km2
    Large area  : A >= 100 km2

Therefore:
    Small area + Medium area + Large area = Total surface-water area
but:
    Small count + Medium count + Large count = Waterbody count (A >= 0.1)
"""

from pathlib import Path
import gc
import time
import numpy as np
import pandas as pd
import geopandas as gpd

# PATHS

POINT_ROOT = Path(
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step4_analysis\QTP_waterbody_points"
)

WATERSHED_SHP = Path(
    r"D:\phdl2\QTP surface water dynamics\regions\QTP_watershed.shp"
)

OUTPUT_ROOT = Path(
    r"K:\Monthly surface water dynamics TIB\supervised classification"
    r"\step4_analysis\QTP_waterbody_statistics_0p1"
)

# SETTINGS

YEARS = range(2016, 2026)
MONTHS = range(1, 13)

AREA_FIELD = "Area_km2"
BASIN_FIELD = "Name"

COUNT_THRESHOLD = 0.1

# If interrupted, completed months can be skipped.
SKIP_COMPLETED = True

PART_QTP = OUTPUT_ROOT / "_parts_qtp"
PART_WATERSHED = OUTPUT_ROOT / "_parts_watershed"

PART_QTP.mkdir(parents=True, exist_ok=True)
PART_WATERSHED.mkdir(parents=True, exist_ok=True)


# FILE HELPERS

def point_file(year, month):
    return POINT_ROOT / f"QTP_waterbody_points_{year}_{month:02d}.gpkg"


def point_layer(year, month):
    return f"points_{year}_{month:02d}"


def qtp_part(year, month):
    return PART_QTP / f"{year}_{month:02d}.csv"


def watershed_part(year, month):
    return PART_WATERSHED / f"{year}_{month:02d}.csv"


# STATISTICS

def calculate_statistics(values):
    """
    Count threshold and area classes deliberately use different lower limits:
    - counts exclude A < 0.1 km2
    - areas retain A < 0.1 km2 inside the Small class
    """

    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]

    # ---------------- count masks ----------------
    below_count_threshold = a < 0.1
    small_count_mask = (a >= 0.1) & (a < 1.0)
    medium_mask = (a >= 1.0) & (a < 100.0)
    large_mask = a >= 100.0

    counted_waterbody_mask = (
        small_count_mask | medium_mask | large_mask
    )

    # ---------------- area masks ----------------
    small_area_mask = a < 1.0
    medium_area_mask = medium_mask
    large_area_mask = large_mask

    result = {
        # all connected water patches
        "All_water_patch_count": int(a.size),
        "Total_surface_water_area_km2": float(
            a.sum(dtype=np.float64)
        ),

        # diagnostic: patches below counting threshold
        "Below_0p1_patch_count": int(
            np.count_nonzero(below_count_threshold)
        ),
        "Below_0p1_area_km2": float(
            a[below_count_threshold].sum(dtype=np.float64)
        ),

        # waterbody count, threshold >=0.1 km2
        "Waterbody_count_ge_0p1": int(
            np.count_nonzero(counted_waterbody_mask)
        ),
        "Waterbody_area_ge_0p1_km2": float(
            a[counted_waterbody_mask].sum(dtype=np.float64)
        ),

        # size-class counts
        "Small_waterbody_count": int(
            np.count_nonzero(small_count_mask)
        ),
        "Medium_waterbody_count": int(
            np.count_nonzero(medium_mask)
        ),
        "Large_waterbody_count": int(
            np.count_nonzero(large_mask)
        ),

        # size-class areas
        # NOTE: small area includes ALL A < 1 km2, including A < 0.1
        "Small_waterbody_area_km2": float(
            a[small_area_mask].sum(dtype=np.float64)
        ),
        "Medium_waterbody_area_km2": float(
            a[medium_area_mask].sum(dtype=np.float64)
        ),
        "Large_waterbody_area_km2": float(
            a[large_area_mask].sum(dtype=np.float64)
        ),
    }

    # CONSISTENCY CHECKS

    # all patches = below threshold + counted waterbodies
    if (
        result["Below_0p1_patch_count"]
        + result["Waterbody_count_ge_0p1"]
        != result["All_water_patch_count"]
    ):
        raise RuntimeError(
            "Count consistency check failed."
        )

    # counted waterbodies = small + medium + large counts
    if (
        result["Small_waterbody_count"]
        + result["Medium_waterbody_count"]
        + result["Large_waterbody_count"]
        != result["Waterbody_count_ge_0p1"]
    ):
        raise RuntimeError(
            "Size-class count consistency check failed."
        )

    # small + medium + large AREA must equal total area
    if not np.isclose(
        result["Small_waterbody_area_km2"]
        + result["Medium_waterbody_area_km2"]
        + result["Large_waterbody_area_km2"],
        result["Total_surface_water_area_km2"],
        rtol=1e-10,
        atol=1e-7
    ):
        raise RuntimeError(
            "Size-class area consistency check failed."
        )

    # below-threshold area + >=0.1 area must equal total area
    if not np.isclose(
        result["Below_0p1_area_km2"]
        + result["Waterbody_area_ge_0p1_km2"],
        result["Total_surface_water_area_km2"],
        rtol=1e-10,
        atol=1e-7
    ):
        raise RuntimeError(
            "Threshold-area consistency check failed."
        )

    return result


# LOAD WATERSHEDS

def load_watersheds():

    reference = None

    for year in YEARS:
        for month in MONTHS:
            if point_file(year, month).exists():
                reference = (year, month)
                break
        if reference:
            break

    if reference is None:
        raise FileNotFoundError(
            f"No point GeoPackage found under:\n{POINT_ROOT}"
        )

    year, month = reference

    sample = gpd.read_file(
        point_file(year, month),
        layer=point_layer(year, month),
        rows=1,
        engine="pyogrio"
    )

    point_crs = sample.crs

    if point_crs is None:
        raise RuntimeError(
            "Point data have no CRS."
        )

    watersheds = gpd.read_file(
        WATERSHED_SHP,
        columns=[BASIN_FIELD, "geometry"],
        engine="pyogrio",
        use_arrow=True
    )

    if watersheds.crs is None:
        raise RuntimeError(
            "Watershed shapefile has no CRS."
        )

    if watersheds.crs != point_crs:
        watersheds = watersheds.to_crs(
            point_crs
        )

    watersheds = watersheds[
        [BASIN_FIELD, "geometry"]
    ].copy()

    watersheds[BASIN_FIELD] = (
        watersheds[BASIN_FIELD].astype(str)
    )

    if watersheds[BASIN_FIELD].duplicated().any():
        watersheds = watersheds.dissolve(
            by=BASIN_FIELD,
            as_index=False
        )

    print("Watersheds:")
    for name in watersheds[BASIN_FIELD]:
        print("  ", name)

    return watersheds


# PROCESS ONE MONTH

def process_month(
    year,
    month,
    watersheds
):

    start = time.time()

    # Read point data

    points = gpd.read_file(
        point_file(year, month),
        layer=point_layer(year, month),
        columns=[AREA_FIELD, "geometry"],
        engine="pyogrio",
        use_arrow=True
    )

    points[AREA_FIELD] = pd.to_numeric(
        points[AREA_FIELD],
        errors="coerce"
    )

    invalid = int(
        points[AREA_FIELD].isna().sum()
    )

    print(
        f"  Read: {len(points):,} patches"
    )

    if invalid:
        print(
            f"  WARNING: invalid Area_km2 = "
            f"{invalid:,}"
        )

    # QTP statistics

    qtp_stats = calculate_statistics(
        points[AREA_FIELD].to_numpy(
            dtype=np.float64,
            copy=False
        )
    )

    qtp_df = pd.DataFrame([{
        "Year": year,
        "Month": month,
        **qtp_stats
    }])

    print(
        "  QTP | "
        f"count>=0.1: "
        f"{qtp_stats['Waterbody_count_ge_0p1']:,} | "
        f"small: "
        f"{qtp_stats['Small_waterbody_count']:,} | "
        f"medium: "
        f"{qtp_stats['Medium_waterbody_count']:,} | "
        f"large: "
        f"{qtp_stats['Large_waterbody_count']:,}"
    )

    print(
        "      | "
        f"small area(<1): "
        f"{qtp_stats['Small_waterbody_area_km2']:,.3f} | "
        f"medium area: "
        f"{qtp_stats['Medium_waterbody_area_km2']:,.3f} | "
        f"large area: "
        f"{qtp_stats['Large_waterbody_area_km2']:,.3f} | "
        f"total: "
        f"{qtp_stats['Total_surface_water_area_km2']:,.3f} km2"
    )

    # Spatial join: centroid -> watershed

    t = time.time()

    joined = gpd.sjoin(
        points[[AREA_FIELD, "geometry"]],
        watersheds[
            [BASIN_FIELD, "geometry"]
        ],
        how="left",
        predicate="within"
    )

    if len(joined) != len(points):
        raise RuntimeError(
            f"Spatial join changed row count: "
            f"{len(points):,} -> {len(joined):,}. "
            "Watershed polygons may overlap."
        )

    assigned = joined[
        BASIN_FIELD
    ].notna()

    print(
        f"  Watershed join: "
        f"{time.time() - t:.1f} s | "
        f"assigned={int(assigned.sum()):,} | "
        f"outside={int((~assigned).sum()):,}"
    )

    # Watershed statistics

    rows = []

    for name in watersheds[
        BASIN_FIELD
    ].tolist():

        values = joined.loc[
            joined[BASIN_FIELD].eq(name),
            AREA_FIELD
        ].to_numpy(
            dtype=np.float64,
            copy=False
        )

        rows.append({
            "Year": year,
            "Month": month,
            BASIN_FIELD: name,
            **calculate_statistics(values)
        })

    watershed_df = pd.DataFrame(rows)

    # watershed cross-check
    if (
        int(
            watershed_df[
                "All_water_patch_count"
            ].sum()
        )
        != int(assigned.sum())
    ):
        raise RuntimeError(
            "Watershed patch-count "
            "consistency check failed."
        )

    assigned_area = float(
        joined.loc[
            assigned,
            AREA_FIELD
        ].sum()
    )

    watershed_area = float(
        watershed_df[
            "Total_surface_water_area_km2"
        ].sum()
    )

    if not np.isclose(
        assigned_area,
        watershed_area,
        rtol=1e-10,
        atol=1e-7
    ):
        raise RuntimeError(
            "Watershed area "
            "consistency check failed."
        )

    # Save checkpoints

    qtp_df.to_csv(
        qtp_part(year, month),
        index=False,
        encoding="utf-8-sig"
    )

    watershed_df.to_csv(
        watershed_part(year, month),
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"[DONE] {year}-{month:02d} | "
        f"{time.time() - start:.1f} s"
    )

    del points
    del joined
    del qtp_df
    del watershed_df

    gc.collect()


# MERGE RESULTS

def merge_results():

    qtp_files = sorted(
        PART_QTP.glob("*.csv")
    )

    watershed_files = sorted(
        PART_WATERSHED.glob("*.csv")
    )

    if not qtp_files:
        raise RuntimeError(
            "No QTP monthly results."
        )

    if not watershed_files:
        raise RuntimeError(
            "No watershed monthly results."
        )

    qtp = pd.concat(
        [
            pd.read_csv(f)
            for f in qtp_files
        ],
        ignore_index=True
    ).sort_values(
        ["Year", "Month"]
    ).reset_index(drop=True)

    watershed = pd.concat(
        [
            pd.read_csv(f)
            for f in watershed_files
        ],
        ignore_index=True
    ).sort_values(
        ["Year", "Month", BASIN_FIELD]
    ).reset_index(drop=True)

    qtp_csv = (
        OUTPUT_ROOT
        / "QTP_monthly_waterbody_statistics.csv"
    )

    watershed_csv = (
        OUTPUT_ROOT
        / "Watershed_monthly_waterbody_statistics.csv"
    )

    excel_file = (
        OUTPUT_ROOT
        / "QTP_waterbody_statistics_2016_2025.xlsx"
    )

    qtp.to_csv(
        qtp_csv,
        index=False,
        encoding="utf-8-sig"
    )

    watershed.to_csv(
        watershed_csv,
        index=False,
        encoding="utf-8-sig"
    )

    with pd.ExcelWriter(
        excel_file,
        engine="openpyxl"
    ) as writer:

        qtp.to_excel(
            writer,
            sheet_name="QTP_monthly",
            index=False
        )

        watershed.to_excel(
            writer,
            sheet_name="Watershed_monthly",
            index=False
        )

        definitions = pd.DataFrame({
            "Metric": [
                "Total surface-water area",
                "Waterbody count",
                "Small waterbody count",
                "Medium waterbody count",
                "Large waterbody count",
                "Small waterbody area",
                "Medium waterbody area",
                "Large waterbody area",
                "Below-threshold diagnostic"
            ],
            "Definition": [
                "All detected connected water patches",
                "A >= 0.1 km2",
                "0.1 <= A < 1 km2",
                "1 <= A < 100 km2",
                "A >= 100 km2",
                "A < 1 km2, including patches below 0.1 km2",
                "1 <= A < 100 km2",
                "A >= 100 km2",
                "A < 0.1 km2; excluded from counts but retained in small-area and total-area statistics"
            ]
        })

        definitions.to_excel(
            writer,
            sheet_name="Definitions",
            index=False
        )

    print()
    print("Final outputs:")
    print(" ", qtp_csv)
    print(" ", watershed_csv)
    print(" ", excel_file)


# MAIN

def main():

    print("=" * 78)
    print("QTP MONTHLY WATERBODY STATISTICS")
    print("COUNT:")
    print("  Small  = 0.1 <= A < 1 km2")
    print("  Medium = 1 <= A < 100 km2")
    print("  Large  = A >= 100 km2")
    print("AREA:")
    print("  Small  = A < 1 km2")
    print("  Medium = 1 <= A < 100 km2")
    print("  Large  = A >= 100 km2")
    print("  Total  = ALL detected water patches")
    print("=" * 78)

    watersheds = load_watersheds()

    failed = []
    missing = []

    for year in YEARS:
        for month in MONTHS:

            path = point_file(
                year,
                month
            )

            if not path.exists():
                print(
                    f"[MISSING] "
                    f"{year}-{month:02d}"
                )
                missing.append(
                    (year, month)
                )
                continue

            if (
                SKIP_COMPLETED
                and qtp_part(
                    year,
                    month
                ).exists()
                and watershed_part(
                    year,
                    month
                ).exists()
            ):
                print(
                    f"[SKIP] "
                    f"{year}-{month:02d}"
                )
                continue

            print()
            print(
                f"[RUN ] "
                f"{year}-{month:02d}"
            )

            try:
                process_month(
                    year,
                    month,
                    watersheds
                )

            except Exception as e:
                print(
                    f"[ERROR] "
                    f"{year}-{month:02d} | "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

                failed.append(
                    (
                        year,
                        month,
                        str(e)
                    )
                )

                gc.collect()

    merge_results()

    print()
    print("=" * 78)

    if missing:
        print(
            "Missing months:",
            missing
        )

    if failed:
        print(
            "Failed months:"
        )

        for item in failed:
            print(
                " ",
                item
            )

    if not missing and not failed:
        print(
            "All 120 months "
            "completed successfully."
        )

    print("=" * 78)


if __name__ == "__main__":
    main()
