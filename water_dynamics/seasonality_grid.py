# -*- coding: utf-8 -*-
"""
Rebuild QTP 25-km seasonality products in the specified Albers Equal Area CRS.

Workflow
--------
Existing monthly 25-km rasters
    Band 1: WaterArea_km2
    Band 2: ValidFraction

        ↓

Reproject to a NEW native 25 km × 25 km Albers grid

Target CRS:
    Projection              Albers Equal Area
    Geographic CRS          WGS84
    Central Meridian        100°
    Latitude of Origin      36°
    Standard Parallel 1     25°
    Standard Parallel 2     47°
    False Easting           0 m
    False Northing          0 m
    Unit                    metre

        ↓

Recalculate:
    1. Relative contribution of seasonality (Cseason)
    2. Relative change in seasonality (RelDec)

Established validity rules are retained:
    Monthly valid: ValidFraction >= 0.90
    Valid year: >= 10 valid months
    Valid 10-year grid: >= 8 valid years

IMPORTANT
---------
1. No 10-m classification raster is reprocessed.
2. No valid-fraction correction is applied to WaterArea.
3. MeanArea >= 0.1 km² is ONLY used for centre-point output.
4. ddof = 0 throughout.
5. RelDec is based on OLS trend of annual intra-annual SD.
"""

from pathlib import Path
import math
import csv
import shutil

import numpy as np
from osgeo import gdal, osr
import geopandas as gpd
from shapely.geometry import Point


# ============================================================
# 0. GDAL SETTINGS
# ============================================================

gdal.UseExceptions()

gdal.SetConfigOption(
    "GDAL_NUM_THREADS",
    "4"
)

gdal.SetConfigOption(
    "GDAL_CACHEMAX",
    "1024"
)


# ============================================================
# 1. INPUT / OUTPUT
# ============================================================

ROOT = Path(
    r"K:\Monthly surface water dynamics TIB"
    r"\supervised classification\step4_analysis"
    r"\QTP_water_area_25km_block"
)

MONTHLY_ROOT = ROOT / "monthly"

OUTPUT_ROOT = Path(
    r"K:\Monthly surface water dynamics TIB"
    r"\supervised classification\step4_analysis"
    r"\QTP_water_area_25km_QTP_Albers"
)

MONTHLY_OUT = OUTPUT_ROOT / "monthly"

SEASONALITY_OUT = (
    OUTPUT_ROOT
    / "seasonality_grid"
)

CHANGE_OUT = (
    OUTPUT_ROOT
    / "seasonality_change"
)

MONTHLY_OUT.mkdir(
    parents=True,
    exist_ok=True
)

SEASONALITY_OUT.mkdir(
    parents=True,
    exist_ok=True
)

CHANGE_OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. YEARS / MONTHS
# ============================================================

YEARS = list(
    range(2016, 2026)
)

MONTHS = list(
    range(1, 13)
)


# ============================================================
# 3. FIXED ANALYSIS PARAMETERS
# ============================================================

TARGET_RES = 25000.0

VALID_FRACTION_THRESHOLD = 0.90

MIN_VALID_MONTHS_PER_YEAR = 10

MIN_VALID_YEARS = 8

POINT_MIN_MEAN_AREA = 0.1

ZERO_TOL = 1e-12


# ============================================================
# 4. TARGET CRS
# ============================================================

TARGET_PROJ4 = (
    "+proj=aea "
    "+lat_1=25 "
    "+lat_2=47 "
    "+lat_0=36 "
    "+lon_0=100 "
    "+x_0=0 "
    "+y_0=0 "
    "+datum=WGS84 "
    "+units=m "
    "+no_defs"
)

target_srs = osr.SpatialReference()
target_srs.ImportFromProj4(
    TARGET_PROJ4
)

TARGET_WKT = target_srs.ExportToWkt()


# ============================================================
# 5. FILE HELPERS
# ============================================================

def input_file(year, month):

    return (
        MONTHLY_ROOT
        / str(year)
        / f"QTP_water_area_25km_{year}_{month:02d}.tif"
    )


def output_file(year, month):

    folder = (
        MONTHLY_OUT
        / str(year)
    )

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    return (
        folder
        / f"QTP_water_area_25km_{year}_{month:02d}.tif"
    )


# ============================================================
# 6. CHECK INPUTS
# ============================================================

def check_inputs():

    missing = []

    for year in YEARS:

        for month in MONTHS:

            path = input_file(
                year,
                month
            )

            if not path.exists():
                missing.append(path)

    if missing:

        print("\nMissing files:")

        for path in missing:
            print(path)

        raise FileNotFoundError(
            f"{len(missing)} monthly input files are missing."
        )

    print(
        "Input check: 120/120 monthly rasters found."
    )


# ============================================================
# 7. SOURCE RASTER BOUNDS
# ============================================================

def raster_bounds(ds):

    gt = ds.GetGeoTransform()

    width = ds.RasterXSize
    height = ds.RasterYSize

    corners = [
        (0, 0),
        (width, 0),
        (0, height),
        (width, height)
    ]

    xy = []

    for px, py in corners:

        x = (
            gt[0]
            + px * gt[1]
            + py * gt[2]
        )

        y = (
            gt[3]
            + px * gt[4]
            + py * gt[5]
        )

        xy.append(
            (x, y)
        )

    xs = [
        p[0]
        for p in xy
    ]

    ys = [
        p[1]
        for p in xy
    ]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys)
    )


# ============================================================
# 8. TRANSFORM SOURCE EXTENT TO TARGET CRS
# ============================================================

def get_target_extent(reference_file):

    ds = gdal.Open(
        str(reference_file)
    )

    if ds is None:
        raise RuntimeError(
            f"Cannot open: {reference_file}"
        )

    source_wkt = ds.GetProjection()

    if not source_wkt:
        raise RuntimeError(
            "Source raster has no CRS."
        )

    xmin, ymin, xmax, ymax = (
        raster_bounds(ds)
    )

    source_srs = osr.SpatialReference()
    source_srs.ImportFromWkt(
        source_wkt
    )

    # --------------------------------------------------------
    # Force traditional GIS axis order
    # --------------------------------------------------------

    source_srs.SetAxisMappingStrategy(
        osr.OAMS_TRADITIONAL_GIS_ORDER
    )

    target_srs_local = (
        osr.SpatialReference()
    )

    target_srs_local.ImportFromProj4(
        TARGET_PROJ4
    )

    target_srs_local.SetAxisMappingStrategy(
        osr.OAMS_TRADITIONAL_GIS_ORDER
    )

    transformer = (
        osr.CoordinateTransformation(
            source_srs,
            target_srs_local
        )
    )

    # --------------------------------------------------------
    # Densify source boundary.
    #
    # Using only four corners is not sufficiently robust
    # when transforming between projected CRSs.
    # --------------------------------------------------------

    n = 101

    boundary = []

    for x in np.linspace(
        xmin,
        xmax,
        n
    ):
        boundary.append(
            (x, ymin)
        )
        boundary.append(
            (x, ymax)
        )

    for y in np.linspace(
        ymin,
        ymax,
        n
    ):
        boundary.append(
            (xmin, y)
        )
        boundary.append(
            (xmax, y)
        )

    tx = []
    ty = []

    for x, y in boundary:

        p = transformer.TransformPoint(
            float(x),
            float(y)
        )

        tx.append(
            p[0]
        )

        ty.append(
            p[1]
        )

    raw_xmin = min(tx)
    raw_xmax = max(tx)

    raw_ymin = min(ty)
    raw_ymax = max(ty)

    # --------------------------------------------------------
    # Snap outward to exact 25-km multiples.
    #
    # This creates a deterministic native 25-km target grid.
    # --------------------------------------------------------

    target_xmin = (
        math.floor(
            raw_xmin / TARGET_RES
        )
        * TARGET_RES
    )

    target_xmax = (
        math.ceil(
            raw_xmax / TARGET_RES
        )
        * TARGET_RES
    )

    target_ymin = (
        math.floor(
            raw_ymin / TARGET_RES
        )
        * TARGET_RES
    )

    target_ymax = (
        math.ceil(
            raw_ymax / TARGET_RES
        )
        * TARGET_RES
    )

    ds = None

    return (
        target_xmin,
        target_ymin,
        target_xmax,
        target_ymax
    )


# ============================================================
# 9. CREATE SINGLE-BAND TEMPORARY VRT
# ============================================================

def create_band_vrt(
    source_file,
    band_number,
    vrt_file
):

    options = gdal.BuildVRTOptions(
        bandList=[
            band_number
        ]
    )

    vrt = gdal.BuildVRT(
        str(vrt_file),
        [
            str(source_file)
        ],
        options=options
    )

    if vrt is None:
        raise RuntimeError(
            f"Failed to create VRT: {vrt_file}"
        )

    vrt.FlushCache()
    vrt = None


# ============================================================
# 10. REPROJECT ONE MONTH
# ============================================================

def reproject_month(
    year,
    month,
    target_extent
):

    src = input_file(
        year,
        month
    )

    dst = output_file(
        year,
        month
    )

    if dst.exists():

        print(
            f"[SKIP] {year}-{month:02d}"
        )

        return

    temp_dir = (
        OUTPUT_ROOT
        / "_temp"
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    area_vrt = (
        temp_dir
        / f"area_{year}_{month:02d}.vrt"
    )

    valid_vrt = (
        temp_dir
        / f"valid_{year}_{month:02d}.vrt"
    )

    area_tmp = (
        temp_dir
        / f"area_{year}_{month:02d}.tif"
    )

    valid_tmp = (
        temp_dir
        / f"valid_{year}_{month:02d}.tif"
    )

    # ========================================================
    # Band 1: WaterArea_km2
    # ========================================================

    create_band_vrt(
        src,
        1,
        area_vrt
    )

    xmin, ymin, xmax, ymax = (
        target_extent
    )

    area_options = (
        gdal.WarpOptions(
            format="GTiff",

            dstSRS=TARGET_WKT,

            outputBounds=(
                xmin,
                ymin,
                xmax,
                ymax
            ),

            xRes=TARGET_RES,
            yRes=TARGET_RES,

            resampleAlg="sum",

            dstNodata=np.nan,

            outputType=gdal.GDT_Float64,

            multithread=True,

            creationOptions=[
                "TILED=YES",
                "COMPRESS=DEFLATE",
                "BIGTIFF=IF_SAFER"
            ]
        )
    )

    result = gdal.Warp(
        str(area_tmp),
        str(area_vrt),
        options=area_options
    )

    if result is None:
        raise RuntimeError(
            f"Area reprojection failed: "
            f"{year}-{month:02d}"
        )

    result.FlushCache()
    result = None

    # ========================================================
    # Band 2: ValidFraction
    # ========================================================

    create_band_vrt(
        src,
        2,
        valid_vrt
    )

    valid_options = (
        gdal.WarpOptions(
            format="GTiff",

            dstSRS=TARGET_WKT,

            outputBounds=(
                xmin,
                ymin,
                xmax,
                ymax
            ),

            xRes=TARGET_RES,
            yRes=TARGET_RES,

            resampleAlg="average",

            dstNodata=np.nan,

            outputType=gdal.GDT_Float64,

            multithread=True,

            creationOptions=[
                "TILED=YES",
                "COMPRESS=DEFLATE",
                "BIGTIFF=IF_SAFER"
            ]
        )
    )

    result = gdal.Warp(
        str(valid_tmp),
        str(valid_vrt),
        options=valid_options
    )

    if result is None:
        raise RuntimeError(
            f"ValidFraction reprojection failed: "
            f"{year}-{month:02d}"
        )

    result.FlushCache()
    result = None

    # ========================================================
    # Combine into final two-band raster
    # ========================================================

    area_ds = gdal.Open(
        str(area_tmp)
    )

    valid_ds = gdal.Open(
        str(valid_tmp)
    )

    area = (
        area_ds
        .GetRasterBand(1)
        .ReadAsArray()
        .astype(np.float64)
    )

    valid = (
        valid_ds
        .GetRasterBand(1)
        .ReadAsArray()
        .astype(np.float64)
    )

    # Numerical safety only
    finite_valid = np.isfinite(valid)

    valid[
        finite_valid
    ] = np.clip(
        valid[
            finite_valid
        ],
        0.0,
        1.0
    )

    driver = gdal.GetDriverByName(
        "GTiff"
    )

    out_ds = driver.Create(
        str(dst),

        area_ds.RasterXSize,
        area_ds.RasterYSize,

        2,

        gdal.GDT_Float64,

        options=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "BIGTIFF=IF_SAFER"
        ]
    )

    out_ds.SetGeoTransform(
        area_ds.GetGeoTransform()
    )

    out_ds.SetProjection(
        TARGET_WKT
    )

    # --------------------------------------------------------
    # Band 1
    # --------------------------------------------------------

    b1 = out_ds.GetRasterBand(1)

    b1.WriteArray(
        area
    )

    b1.SetDescription(
        "WaterArea_km2"
    )

    b1.SetNoDataValue(
        np.nan
    )

    # --------------------------------------------------------
    # Band 2
    # --------------------------------------------------------

    b2 = out_ds.GetRasterBand(2)

    b2.WriteArray(
        valid
    )

    b2.SetDescription(
        "ValidFraction"
    )

    b2.SetNoDataValue(
        np.nan
    )

    out_ds.FlushCache()

    out_ds = None
    area_ds = None
    valid_ds = None

    # ========================================================
    # Delete temporary files
    # ========================================================

    for f in [
        area_vrt,
        valid_vrt,
        area_tmp,
        valid_tmp
    ]:

        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass

    print(
        f"[DONE] {year}-{month:02d}"
    )


# ============================================================
# 11. REPROJECT ALL 120 MONTHS
# ============================================================

def reproject_all():

    reference = input_file(
        YEARS[0],
        MONTHS[0]
    )

    extent = get_target_extent(
        reference
    )

    xmin, ymin, xmax, ymax = (
        extent
    )

    cols = int(
        round(
            (xmax - xmin)
            / TARGET_RES
        )
    )

    rows = int(
        round(
            (ymax - ymin)
            / TARGET_RES
        )
    )

    print()
    print("=" * 70)
    print("TARGET GRID")
    print("=" * 70)

    print(
        f"xmin = {xmin:.0f}"
    )

    print(
        f"ymin = {ymin:.0f}"
    )

    print(
        f"xmax = {xmax:.0f}"
    )

    print(
        f"ymax = {ymax:.0f}"
    )

    print(
        f"resolution = "
        f"{TARGET_RES:.0f} m"
    )

    print(
        f"rows = {rows}"
    )

    print(
        f"cols = {cols}"
    )

    print()

    for year in YEARS:

        for month in MONTHS:

            reproject_month(
                year,
                month,
                extent
            )

    return extent


# ============================================================
# 12. LOAD 120-MONTH DATA CUBE
# ============================================================

def load_monthly_cube():

    first = output_file(
        YEARS[0],
        MONTHS[0]
    )

    ds = gdal.Open(
        str(first)
    )

    rows = ds.RasterYSize
    cols = ds.RasterXSize

    gt = ds.GetGeoTransform()

    projection = ds.GetProjection()

    ds = None

    ntime = (
        len(YEARS)
        * len(MONTHS)
    )

    area_cube = np.full(
        (
            ntime,
            rows,
            cols
        ),
        np.nan,
        dtype=np.float64
    )

    valid_cube = np.full(
        (
            ntime,
            rows,
            cols
        ),
        np.nan,
        dtype=np.float64
    )

    t = 0

    for year in YEARS:

        for month in MONTHS:

            path = output_file(
                year,
                month
            )

            ds = gdal.Open(
                str(path)
            )

            area = (
                ds
                .GetRasterBand(1)
                .ReadAsArray()
                .astype(np.float64)
            )

            valid = (
                ds
                .GetRasterBand(2)
                .ReadAsArray()
                .astype(np.float64)
            )

            # ------------------------------------------------
            # Established monthly validity rule
            # ------------------------------------------------

            good = (
                np.isfinite(area)
                &
                np.isfinite(valid)
                &
                (
                    valid
                    >= VALID_FRACTION_THRESHOLD
                )
            )

            area_cube[
                t
            ][
                good
            ] = area[
                good
            ]

            valid_cube[
                t
            ] = valid

            ds = None

            t += 1

    return (
        area_cube,
        valid_cube,
        gt,
        projection
    )


# ============================================================
# 13. CALCULATE ANNUAL METRICS
# ============================================================

def calculate_annual_metrics(
    area_cube
):

    n_years = len(
        YEARS
    )

    rows = area_cube.shape[1]
    cols = area_cube.shape[2]

    annual_mean = np.full(
        (
            n_years,
            rows,
            cols
        ),
        np.nan,
        dtype=np.float64
    )

    annual_sd = np.full(
        (
            n_years,
            rows,
            cols
        ),
        np.nan,
        dtype=np.float64
    )

    valid_month_count = np.zeros(
        (
            n_years,
            rows,
            cols
        ),
        dtype=np.int16
    )

    for yi, year in enumerate(
        YEARS
    ):

        start = yi * 12
        end = start + 12

        block = area_cube[
            start:end
        ]

        count = np.sum(
            np.isfinite(block),
            axis=0
        )

        valid_month_count[
            yi
        ] = count

        valid_year = (
            count
            >= MIN_VALID_MONTHS_PER_YEAR
        )

        # ----------------------------------------------------
        # Avoid warnings for completely empty pixels
        # ----------------------------------------------------

        with np.errstate(
            invalid="ignore",
            divide="ignore"
        ):

            mean_temp = np.nanmean(
                block,
                axis=0
            )

            sd_temp = np.nanstd(
                block,
                axis=0,
                ddof=0
            )

        annual_mean[
            yi
        ][
            valid_year
        ] = mean_temp[
            valid_year
        ]

        annual_sd[
            yi
        ][
            valid_year
        ] = sd_temp[
            valid_year
        ]

    return (
        annual_mean,
        annual_sd,
        valid_month_count
    )


# ============================================================
# 14. CSEASON
# ============================================================

def calculate_cseason(
    area_cube,
    annual_mean,
    annual_sd
):

    valid_years = np.sum(
        np.isfinite(annual_sd),
        axis=0
    )

    eligible = (
        valid_years
        >= MIN_VALID_YEARS
    )

    rows = annual_sd.shape[1]
    cols = annual_sd.shape[2]

    sigma_bar = np.full(
        (
            rows,
            cols
        ),
        np.nan,
        dtype=np.float64
    )

    delta = np.full(
        (
            rows,
            cols
        ),
        np.nan,
        dtype=np.float64
    )

    cseason = np.full(
        (
            rows,
            cols
        ),
        np.nan,
        dtype=np.float64
    )

    mean_area = np.full(
        (
            rows,
            cols
        ),
        np.nan,
        dtype=np.float64
    )

    with np.errstate(
        invalid="ignore",
        divide="ignore"
    ):

        sigma_temp = np.nanmean(
            annual_sd,
            axis=0
        )

        delta_temp = np.nanstd(
            annual_mean,
            axis=0,
            ddof=0
        )

        area_temp = np.nanmean(
            area_cube,
            axis=0
        )

    sigma_bar[
        eligible
    ] = sigma_temp[
        eligible
    ]

    delta[
        eligible
    ] = delta_temp[
        eligible
    ]

    mean_area[
        eligible
    ] = area_temp[
        eligible
    ]

    denominator = (
        sigma_bar
        + delta
    )

    good = (
        eligible
        &
        np.isfinite(denominator)
        &
        (
            denominator
            > ZERO_TOL
        )
    )

    cseason[
        good
    ] = (
        100.0
        * sigma_bar[
            good
        ]
        / denominator[
            good
        ]
    )

    return (
        cseason,
        sigma_bar,
        delta,
        mean_area,
        valid_years
    )


# ============================================================
# 15. RELATIVE SEASONALITY CHANGE
# ============================================================

def calculate_change(
    annual_sd,
    mean_area
):

    rows = annual_sd.shape[1]
    cols = annual_sd.shape[2]

    valid_years = np.sum(
        np.isfinite(annual_sd),
        axis=0
    )

    mean_seas = np.full(
        (
            rows,
            cols
        ),
        np.nan,
        dtype=np.float64
    )

    slope_year = np.full(
        (
            rows,
            cols
        ),
        np.nan,
        dtype=np.float64
    )

    abs_dec = np.full(
        (
            rows,
            cols
        ),
        np.nan,
        dtype=np.float64
    )

    rel_dec = np.full(
        (
            rows,
            cols
        ),
        np.nan,
        dtype=np.float64
    )

    years_array = np.asarray(
        YEARS,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Grid-by-grid OLS.
    #
    # Only 10 annual observations maximum, so this loop is
    # inexpensive relative to raster reprojection.
    # --------------------------------------------------------

    for row in range(rows):

        for col in range(cols):

            y = annual_sd[
                :,
                row,
                col
            ]

            good = np.isfinite(y)

            n = int(
                np.sum(good)
            )

            if n < MIN_VALID_YEARS:
                continue

            x = years_array[
                good
            ]

            yy = y[
                good
            ]

            ms = float(
                np.mean(yy)
            )

            mean_seas[
                row,
                col
            ] = ms

            # -----------------------------------------------
            # OLS slope
            # -----------------------------------------------

            x_centered = (
                x
                - np.mean(x)
            )

            y_centered = (
                yy
                - np.mean(yy)
            )

            denominator = np.sum(
                x_centered ** 2
            )

            if denominator <= 0:
                continue

            beta = (
                np.sum(
                    x_centered
                    * y_centered
                )
                / denominator
            )

            slope_year[
                row,
                col
            ] = beta

            abs_dec[
                row,
                col
            ] = (
                beta
                * 10.0
            )

            if (
                np.isfinite(ms)
                and ms > ZERO_TOL
            ):

                rel_dec[
                    row,
                    col
                ] = (
                    beta
                    / ms
                    * 100.0
                    * 10.0
                )

    return (
        mean_seas,
        slope_year,
        abs_dec,
        rel_dec,
        valid_years
    )


# ============================================================
# 16. GRID CENTRE
# ============================================================

def grid_center(
    row,
    col,
    gt
):

    x = (
        gt[0]
        + (
            col + 0.5
        ) * gt[1]
        + (
            row + 0.5
        ) * gt[2]
    )

    y = (
        gt[3]
        + (
            col + 0.5
        ) * gt[4]
        + (
            row + 0.5
        ) * gt[5]
    )

    return x, y


# ============================================================
# 17. CSEASON CLASS
# ============================================================

def dominance_class(
    value
):

    if not np.isfinite(value):
        return None

    if value > 50:
        return "Seasonal"

    if value < 50:
        return "Nonseason"

    return "Equal"


# ============================================================
# 18. CHANGE CLASS
# ============================================================

def change_class(
    value
):

    if not np.isfinite(value):
        return None

    if value > 5:
        return "RapidInc"

    if value > 0:
        return "Increase"

    if value < -5:
        return "RapidDec"

    if value < 0:
        return "Decrease"

    return "NoChange"


# ============================================================
# 19. WRITE SEASONALITY SHAPEFILE
# ============================================================

def write_seasonality_shapefile(
    cseason,
    sigma_bar,
    delta,
    mean_area,
    valid_years,
    gt
):

    records = []

    rows, cols = (
        cseason.shape
    )

    grid_id = 0

    for row in range(rows):

        for col in range(cols):

            grid_id += 1

            cs = cseason[
                row,
                col
            ]

            ma = mean_area[
                row,
                col
            ]

            # ------------------------------------------------
            # Centre-point output filter only
            # ------------------------------------------------

            if not np.isfinite(cs):
                continue

            if not np.isfinite(ma):
                continue

            if ma < POINT_MIN_MEAN_AREA:
                continue

            x, y = grid_center(
                row,
                col,
                gt
            )

            records.append({
                "GridID": grid_id,
                "Row": row,
                "Col": col,
                "Cseason": float(cs),
                "SigmaBar": float(
                    sigma_bar[
                        row,
                        col
                    ]
                ),
                "Delta": float(
                    delta[
                        row,
                        col
                    ]
                ),
                "MeanArea": float(ma),
                "ValidYrs": int(
                    valid_years[
                        row,
                        col
                    ]
                ),
                "Dominance": (
                    dominance_class(cs)
                ),
                "geometry": Point(
                    x,
                    y
                )
            })

    gdf = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=TARGET_WKT
    )

    shp = (
        SEASONALITY_OUT
        / "QTP_25km_seasonality_centres.shp"
    )

    gdf.to_file(
        shp,
        driver="ESRI Shapefile",
        encoding="UTF-8"
    )

    csv_file = (
        SEASONALITY_OUT
        / "QTP_25km_seasonality_centres.csv"
    )

    gdf.drop(
        columns="geometry"
    ).to_csv(
        csv_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"Seasonality points: "
        f"{len(gdf):,}"
    )

    return shp


# ============================================================
# 20. WRITE CHANGE SHAPEFILE
# ============================================================

def write_change_shapefile(
    mean_seas,
    slope_year,
    abs_dec,
    rel_dec,
    mean_area,
    valid_years,
    gt
):

    records = []

    rows, cols = (
        rel_dec.shape
    )

    grid_id = 0

    for row in range(rows):

        for col in range(cols):

            grid_id += 1

            rd = rel_dec[
                row,
                col
            ]

            ma = mean_area[
                row,
                col
            ]

            if not np.isfinite(rd):
                continue

            if not np.isfinite(ma):
                continue

            # ------------------------------------------------
            # Centre-point output filter only
            # ------------------------------------------------

            if ma < POINT_MIN_MEAN_AREA:
                continue

            x, y = grid_center(
                row,
                col,
                gt
            )

            records.append({
                "GridID": grid_id,
                "Row": row,
                "Col": col,
                "ValidYrs": int(
                    valid_years[
                        row,
                        col
                    ]
                ),
                "MeanArea": float(ma),
                "MeanSeas": float(
                    mean_seas[
                        row,
                        col
                    ]
                ),
                "SlopeYr": float(
                    slope_year[
                        row,
                        col
                    ]
                ),
                "AbsDec": float(
                    abs_dec[
                        row,
                        col
                    ]
                ),
                "RelDec": float(rd),
                "Class": (
                    change_class(rd)
                ),
                "geometry": Point(
                    x,
                    y
                )
            })

    gdf = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=TARGET_WKT
    )

    shp = (
        CHANGE_OUT
        / "QTP_25km_seasonality_change_centres.shp"
    )

    gdf.to_file(
        shp,
        driver="ESRI Shapefile",
        encoding="UTF-8"
    )

    csv_file = (
        CHANGE_OUT
        / "QTP_25km_seasonality_change_centres.csv"
    )

    gdf.drop(
        columns="geometry"
    ).to_csv(
        csv_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"Seasonality-change points: "
        f"{len(gdf):,}"
    )

    return shp


# ============================================================
# 21. WRITE README
# ============================================================

def write_readme():

    readme = OUTPUT_ROOT / "README.txt"

    text = f"""
QTP 25 km Albers seasonality results
===================================

Target grid: WGS 1984 Albers Equal Area, 25,000 m resolution;
central meridian 100 degrees, origin 36 degrees, standard parallels 25 and 47 degrees.

Input: {MONTHLY_ROOT}
Monthly bands: WaterArea_km2 and ValidFraction (2016–2025).
Water area is reprojected using GDAL sum and valid fraction using GDAL average.
No valid-fraction area extrapolation is applied.

Valid months have ValidFraction >= {VALID_FRACTION_THRESHOLD}.
Valid years have at least {MIN_VALID_MONTHS_PER_YEAR} valid months.
Grid cells require at least {MIN_VALID_YEARS} valid years.

SigmaBar is the mean within-year population standard deviation of monthly area;
Delta is the population standard deviation of annual mean area (ddof=0).
Cseason = 100 * SigmaBar / (SigmaBar + Delta).
Values above 50 are seasonal; below 50 are non-seasonal; 50 is equal.

Yearly intra-annual standard deviations are regressed against year using OLS.
SlopeYr is the annual slope; AbsDec = 10 * SlopeYr;
RelDec = 100 * AbsDec / MeanSeas, where MeanSeas is the mean within-year SD.
RelDec > 5 is RapidInc, 0 < RelDec <= 5 Increase,
-5 <= RelDec < 0 Decrease, RelDec < -5 RapidDec and 0 NoChange.

The MeanArea >= {POINT_MIN_MEAN_AREA} km2 threshold applies only to centre-point output.
Monthly rasters and Cseason/RelDec calculations are not filtered by this threshold.

Outputs:
seasonality_grid/QTP_25km_seasonality_centres.shp
seasonality_change/QTP_25km_seasonality_change_centres.shp
monthly/YYYY/ (reprojected monthly rasters)
"""

    readme.write_text(
        text.strip(),
        encoding="utf-8"
    )


# ============================================================
# 22. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "QTP 25-km NEW ALBERS SEASONALITY ANALYSIS"
    )
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    check_inputs()

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    reproject_all()

    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------

    (
        area_cube,
        valid_cube,
        gt,
        projection
    ) = load_monthly_cube()

    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    (
        annual_mean,
        annual_sd,
        valid_month_count
    ) = calculate_annual_metrics(
        area_cube
    )

    # --------------------------------------------------------
    # Step 5
    # --------------------------------------------------------

    (
        cseason,
        sigma_bar,
        delta,
        mean_area,
        valid_years_cseason
    ) = calculate_cseason(
        area_cube,
        annual_mean,
        annual_sd
    )

    # --------------------------------------------------------
    # Step 6
    # --------------------------------------------------------

    (
        mean_seas,
        slope_year,
        abs_dec,
        rel_dec,
        valid_years_change
    ) = calculate_change(
        annual_sd,
        mean_area
    )

    # --------------------------------------------------------
    # Step 7
    # --------------------------------------------------------

    shp_a = (
        write_seasonality_shapefile(
            cseason,
            sigma_bar,
            delta,
            mean_area,
            valid_years_cseason,
            gt
        )
    )

    # --------------------------------------------------------
    # Step 8
    # --------------------------------------------------------

    shp_b = (
        write_change_shapefile(
            mean_seas,
            slope_year,
            abs_dec,
            rel_dec,
            mean_area,
            valid_years_change,
            gt
        )
    )

    # --------------------------------------------------------
    # Step 9
    # --------------------------------------------------------

    write_readme()

    # --------------------------------------------------------
    # Clean empty temporary directory
    # --------------------------------------------------------

    temp_dir = (
        OUTPUT_ROOT
        / "_temp"
    )

    if temp_dir.exists():

        try:
            shutil.rmtree(
                temp_dir
            )
        except Exception:
            pass

    print()
    print("=" * 70)
    print("COMPLETED")
    print("=" * 70)

    print()
    print("A:")
    print(shp_a)

    print()
    print("B:")
    print(shp_b)

    print()


if __name__ == "__main__":
    main()