# -*- coding: utf-8 -*-

"""Mosaic regional water masks and clip them to a common QTP grid."""


# 0. Imports


import re
import csv
import math
import traceback
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from osgeo import gdal, ogr, osr


# 1. Paths


INFERENCE_ROOT = Path(
    r"K:\Monthly surface water dynamics TIB"
    r"\supervised classification"
    r"\step2_semantic_segmentation"
    r"\inference"
)


REGION_ROOT = (
    INFERENCE_ROOT
    / "FULL_DATA_FGSA_UNET_200EPOCHS_SAFE_FP32"
)


# Stage 1 output


MERGED_ROOT = (
    INFERENCE_ROOT
    / "QTP_water_mask_merged"
)

MERGED_VRT_ROOT = (
    MERGED_ROOT
    / "_temp_vrt"
)


# Stage 2 output


CLIPPED_ROOT = (
    INFERENCE_ROOT
    / "QTP_water_mask_clipped"
)

# QTP boundary
QTP_SHP = Path(
    r"D:\phdl2\QTP surface water dynamics"
    r"\regions"
    r"\QTP.shp"
)


# Logs


MERGE_LOG = (
    MERGED_ROOT
    / "merge_log.csv"
)

CLIP_LOG = (
    CLIPPED_ROOT
    / "clip_log.csv"
)

ERROR_LOG = (
    INFERENCE_ROOT
    / "QTP_mosaic_clip_errors.txt"
)


# 2. Settings


INVALID_VALUE = 2

BLOCK_SIZE = 512

GDAL_CACHE_MB = 4096
GDAL_THREADS = "ALL_CPUS"


RUN_STAGE_1_MOSAIC = True
RUN_STAGE_2_CLIP = True


SKIP_EXISTING_MERGED = True
SKIP_EXISTING_CLIPPED = True


CHECK_CLIPPED_ALIGNMENT = True


# 3. GDAL configuration


gdal.UseExceptions()

gdal.SetConfigOption(
    "GDAL_NUM_THREADS",
    GDAL_THREADS,
)

gdal.SetConfigOption(
    "CPL_NUM_THREADS",
    GDAL_THREADS,
)

gdal.SetConfigOption(
    "GDAL_CACHEMAX",
    str(GDAL_CACHE_MB),
)


# 4. Create directories


MERGED_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

MERGED_VRT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

CLIPPED_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# 5. General helpers


def now_string():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def append_error(message):

    with open(
        ERROR_LOG,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            f"\n[{now_string()}]\n"
        )

        f.write(message)

        f.write("\n")


def append_csv(
    path,
    fieldnames,
    row,
):

    path = Path(path)

    write_header = (
        not path.exists()
    )

    with open(
        path,
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(
            row
        )


# 6. SRS helpers


def make_srs(wkt):

    if not wkt:
        return None

    srs = osr.SpatialReference()

    srs.ImportFromWkt(
        wkt
    )

    try:

        srs.SetAxisMappingStrategy(
            osr.OAMS_TRADITIONAL_GIS_ORDER
        )

    except Exception:
        pass

    return srs


def srs_is_same(
    wkt1,
    wkt2,
):

    srs1 = make_srs(
        wkt1
    )

    srs2 = make_srs(
        wkt2
    )

    if (
        srs1 is None
        or srs2 is None
    ):

        return False

    return bool(
        srs1.IsSame(
            srs2
        )
    )


# 7. Raster metadata


def inspect_raster(path):

    ds = gdal.Open(
        str(path),
        gdal.GA_ReadOnly,
    )

    if ds is None:

        raise RuntimeError(
            f"Cannot open raster:\n"
            f"{path}"
        )

    gt = ds.GetGeoTransform()

    info = {

        "path": Path(path),

        "xsize": ds.RasterXSize,
        "ysize": ds.RasterYSize,

        "gt": tuple(gt),

        "projection": (
            ds.GetProjection()
        ),

        "xres": abs(
            gt[1]
        ),

        "yres": abs(
            gt[5]
        ),

        "xmin": gt[0],

        "ymax": gt[3],

        "xmax": (
            gt[0]
            + ds.RasterXSize * gt[1]
        ),

        "ymin": (
            gt[3]
            + ds.RasterYSize * gt[5]
        ),
    }

    ds = None

    return info


# 8. Discover regional masks


def discover_regional_masks():

    if not REGION_ROOT.exists():

        raise FileNotFoundError(
            f"REGION_ROOT does not exist:\n"
            f"{REGION_ROOT}"
        )

    pattern = re.compile(
        r"^mask_fgsa_unet_r"
        r"(\d+)_"
        r"(\d{4})_"
        r"(\d{2})"
        r"\.tif$",
        re.IGNORECASE,
    )

    monthly_files = (
        defaultdict(list)
    )

    print(
        "\nSearching regional masks..."
    )

    for path in REGION_ROOT.rglob(
        "mask_fgsa_unet_r*.tif"
    ):

        match = pattern.match(
            path.name
        )

        if match is None:
            continue

        region_id = int(
            match.group(1)
        )

        year = int(
            match.group(2)
        )

        month = int(
            match.group(3)
        )

        if not (
            1 <= month <= 12
        ):
            continue

        monthly_files[
            (year, month)
        ].append(
            (
                region_id,
                path,
            )
        )

    for key in monthly_files:

        monthly_files[key].sort(
            key=lambda x: x[0]
        )

    return monthly_files


# 9. STAGE 1
# Direct mosaic — no fixed XY, no clipping


def mosaic_one_month(
    year,
    month,
    region_files,
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"[MERGE] "
        f"{year}-{month:02d}"
    )

    print(
        f"Regions: "
        f"{len(region_files)}"
    )

    output_path = (
        MERGED_ROOT
        / (
            f"QTP_water_mask_"
            f"{year}_"
            f"{month:02d}.tif"
        )
    )

    vrt_path = (
        MERGED_VRT_ROOT
        / (
            f"QTP_water_mask_"
            f"{year}_"
            f"{month:02d}.vrt"
        )
    )

    
    # Skip existing
    

    if (
        SKIP_EXISTING_MERGED
        and output_path.exists()
    ):

        print(
            "[SKIP MERGE]"
        )

        return

    input_paths = [
        str(path)
        for _, path
        in region_files
    ]

    if not input_paths:

        return

    if vrt_path.exists():

        try:
            vrt_path.unlink()
        except Exception:
            pass

    
    # Build VRT
    #
    
    #
    
    
    

    vrt_options = (
        gdal.BuildVRTOptions(

            resolution="highest",

            resampleAlg="nearest",

            srcNodata=(
                INVALID_VALUE
            ),

            VRTNodata=(
                INVALID_VALUE
            ),

            addAlpha=False,
        )
    )

    vrt_ds = gdal.BuildVRT(
        str(vrt_path),
        input_paths,
        options=vrt_options,
    )

    if vrt_ds is None:

        raise RuntimeError(
            f"BuildVRT failed: "
            f"{year}-{month:02d}"
        )

    vrt_ds.FlushCache()

    vrt_ds = None

    
    # VRT -> GeoTIFF
    

    translate_options = (
        gdal.TranslateOptions(

            format="GTiff",

            outputType=(
                gdal.GDT_Byte
            ),

            noData=(
                INVALID_VALUE
            ),

            creationOptions=[

                "COMPRESS=LZW",

                "TILED=YES",

                f"BLOCKXSIZE={BLOCK_SIZE}",
                f"BLOCKYSIZE={BLOCK_SIZE}",

                "BIGTIFF=IF_SAFER",

                f"NUM_THREADS={GDAL_THREADS}",
            ],
        )
    )

    out_ds = gdal.Translate(
        str(output_path),
        str(vrt_path),
        options=translate_options,
    )

    if out_ds is None:

        raise RuntimeError(
            f"Translate failed: "
            f"{year}-{month:02d}"
        )

    xsize = (
        out_ds.RasterXSize
    )

    ysize = (
        out_ds.RasterYSize
    )

    out_ds.FlushCache()

    out_ds = None

    
    # Delete VRT
    

    try:

        if vrt_path.exists():
            vrt_path.unlink()

    except Exception:
        pass

    size_gb = (
        output_path.stat().st_size
        / 1024 ** 3
    )

    print(
        f"[MERGED] "
        f"{xsize} x {ysize} | "
        f"{size_gb:.3f} GB"
    )

    append_csv(

        MERGE_LOG,

        [
            "time",
            "year",
            "month",
            "regions",
            "xsize",
            "ysize",
            "size_gb",
            "status",
        ],

        {
            "time": now_string(),
            "year": year,
            "month": f"{month:02d}",
            "regions": len(region_files),
            "xsize": xsize,
            "ysize": ysize,
            "size_gb": (
                f"{size_gb:.3f}"
            ),
            "status": "success",
        },
    )


# 10. Discover merged files


def discover_merged_files():

    pattern = re.compile(
        r"^QTP_water_mask_"
        r"(\d{4})_"
        r"(\d{2})"
        r"\.tif$",
        re.IGNORECASE,
    )

    files = []

    for path in MERGED_ROOT.glob(
        "QTP_water_mask_????_??.tif"
    ):

        match = pattern.match(
            path.name
        )

        if match is None:
            continue

        year = int(
            match.group(1)
        )

        month = int(
            match.group(2)
        )

        files.append(
            (
                year,
                month,
                path,
            )
        )

    files.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    return files


# 11. QTP geometry extent in raster CRS


def get_qtp_extent_in_raster_crs(
    shp_path,
    raster_projection,
):

    """Transform the QTP boundary extent into the raster CRS."""

    ds = ogr.Open(
        str(shp_path)
    )

    if ds is None:

        raise RuntimeError(
            f"Cannot open QTP shapefile:\n"
            f"{shp_path}"
        )

    layer = ds.GetLayer(0)

    source_srs = (
        layer.GetSpatialRef()
    )

    if source_srs is None:

        raise RuntimeError(
            "QTP.shp has no projection."
        )

    target_srs = make_srs(
        raster_projection
    )

    try:

        source_srs.SetAxisMappingStrategy(
            osr.OAMS_TRADITIONAL_GIS_ORDER
        )

    except Exception:
        pass

    transform = None

    if not source_srs.IsSame(
        target_srs
    ):

        transform = (
            osr.CoordinateTransformation(
                source_srs,
                target_srs,
            )
        )

    xmin = None
    ymin = None
    xmax = None
    ymax = None

    layer.ResetReading()

    for feature in layer:

        geometry = (
            feature.GetGeometryRef()
        )

        if geometry is None:
            continue

        geom = geometry.Clone()

        if transform is not None:

            geom.Transform(
                transform
            )

        env = geom.GetEnvelope()

        # xmin, xmax, ymin, ymax
        gxmin = env[0]
        gxmax = env[1]
        gymin = env[2]
        gymax = env[3]

        if xmin is None:

            xmin = gxmin
            xmax = gxmax
            ymin = gymin
            ymax = gymax

        else:

            xmin = min(
                xmin,
                gxmin,
            )

            xmax = max(
                xmax,
                gxmax,
            )

            ymin = min(
                ymin,
                gymin,
            )

            ymax = max(
                ymax,
                gymax,
            )

    ds = None

    if xmin is None:

        raise RuntimeError(
            "Cannot obtain QTP geometry extent."
        )

    return (
        xmin,
        ymin,
        xmax,
        ymax,
    )


# 12. Build ONE fixed clipping grid


def build_fixed_clip_grid(
    reference_raster,
):

    """Build a fixed clipping grid aligned to the reference mosaic."""

    info = inspect_raster(
        reference_raster
    )

    xres = info["xres"]
    yres = info["yres"]

    gt = info["gt"]

    origin_x = gt[0]
    origin_y = gt[3]

    (
        qtp_xmin,
        qtp_ymin,
        qtp_xmax,
        qtp_ymax,
    ) = get_qtp_extent_in_raster_crs(
        QTP_SHP,
        info["projection"],
    )

    
    # Align bbox to source pixel grid
    

    xmin = (
        origin_x
        + math.floor(
            (
                qtp_xmin
                - origin_x
            )
            / xres
        )
        * xres
    )

    xmax = (
        origin_x
        + math.ceil(
            (
                qtp_xmax
                - origin_x
            )
            / xres
        )
        * xres
    )

    ymax = (
        origin_y
        - math.floor(
            (
                origin_y
                - qtp_ymax
            )
            / yres
        )
        * yres
    )

    ymin = (
        origin_y
        - math.ceil(
            (
                origin_y
                - qtp_ymin
            )
            / yres
        )
        * yres
    )

    xsize = int(
        round(
            (
                xmax - xmin
            )
            / xres
        )
    )

    ysize = int(
        round(
            (
                ymax - ymin
            )
            / yres
        )
    )

    gt_out = (
        xmin,
        xres,
        0.0,
        ymax,
        0.0,
        -yres,
    )

    return {

        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,

        "xres": xres,
        "yres": yres,

        "xsize": xsize,
        "ysize": ysize,

        "gt": gt_out,

        "projection": (
            info["projection"]
        ),
    }


# 13. Strict alignment validation


def validate_clipped_grid(
    path,
    grid,
):

    ds = gdal.Open(
        str(path),
        gdal.GA_ReadOnly,
    )

    if ds is None:

        return (
            False,
            "Cannot open output."
        )

    
    # Raster size
    

    if (
        ds.RasterXSize
        != grid["xsize"]
    ):

        ds = None

        return (
            False,
            "RasterXSize mismatch."
        )

    if (
        ds.RasterYSize
        != grid["ysize"]
    ):

        ds = None

        return (
            False,
            "RasterYSize mismatch."
        )

    
    # GeoTransform
    

    gt = tuple(
        ds.GetGeoTransform()
    )

    ref_gt = (
        grid["gt"]
    )

    for i in range(6):

        tolerance = max(
            abs(ref_gt[i]),
            1.0,
        ) * 1e-10

        if abs(
            gt[i]
            - ref_gt[i]
        ) > tolerance:

            ds = None

            return (
                False,
                (
                    f"GeoTransform[{i}] "
                    f"mismatch."
                ),
            )

    
    # Projection
    

    projection = (
        ds.GetProjection()
    )

    if not srs_is_same(
        projection,
        grid["projection"],
    ):

        ds = None

        return (
            False,
            "Projection mismatch."
        )

    ds = None

    return (
        True,
        "PASS",
    )


# 14. STAGE 2
# Unified shapefile clipping


def clip_one_month(
    year,
    month,
    input_path,
    fixed_grid,
):

    print(
        "\n"
        + "-" * 90
    )

    print(
        f"[CLIP] "
        f"{year}-{month:02d}"
    )

    output_path = (
        CLIPPED_ROOT
        / (
            f"QTP_water_mask_"
            f"{year}_"
            f"{month:02d}_clip.tif"
        )
    )

    
    # Existing
    

    if (
        SKIP_EXISTING_CLIPPED
        and output_path.exists()
    ):

        if CHECK_CLIPPED_ALIGNMENT:

            passed, message = (
                validate_clipped_grid(
                    output_path,
                    fixed_grid,
                )
            )

            if not passed:

                raise RuntimeError(
                    f"Existing clipped raster "
                    f"is not aligned:\n"
                    f"{message}"
                )

        print(
            "[SKIP CLIP]"
        )

        return

    
    # Warp
    #
    
    #
    # cropToCutline = False
    #
    
    #
    
    
    
    
    #
    
    #
    
    

    warp_options = (
        gdal.WarpOptions(

            format="GTiff",

            # QTP polygon
            cutlineDSName=str(
                QTP_SHP
            ),

            
            cropToCutline=False,

            
            outputBounds=(

                fixed_grid["xmin"],
                fixed_grid["ymin"],
                fixed_grid["xmax"],
                fixed_grid["ymax"],
            ),

            
            width=(
                fixed_grid["xsize"]
            ),

            height=(
                fixed_grid["ysize"]
            ),

            dstSRS=(
                fixed_grid["projection"]
            ),

            srcNodata=(
                INVALID_VALUE
            ),

            dstNodata=(
                INVALID_VALUE
            ),

            resampleAlg="near",

            outputType=(
                gdal.GDT_Byte
            ),

            multithread=True,

            warpOptions=[

                f"NUM_THREADS={GDAL_THREADS}",

                
                
                "INIT_DEST=NO_DATA",
            ],

            creationOptions=[

                "COMPRESS=LZW",

                "TILED=YES",

                f"BLOCKXSIZE={BLOCK_SIZE}",
                f"BLOCKYSIZE={BLOCK_SIZE}",

                "BIGTIFF=IF_SAFER",

                f"NUM_THREADS={GDAL_THREADS}",
            ],
        )
    )

    out_ds = gdal.Warp(
        str(output_path),
        str(input_path),
        options=warp_options,
    )

    if out_ds is None:

        raise RuntimeError(
            f"gdal.Warp failed:\n"
            f"{input_path}"
        )

    band = (
        out_ds.GetRasterBand(1)
    )

    band.SetNoDataValue(
        INVALID_VALUE
    )

    out_ds.FlushCache()

    out_ds = None

    
    # Strict alignment check
    

    if CHECK_CLIPPED_ALIGNMENT:

        passed, message = (
            validate_clipped_grid(
                output_path,
                fixed_grid,
            )
        )

        if not passed:

            raise RuntimeError(
                f"STRICT ALIGNMENT CHECK FAILED:\n"
                f"{output_path}\n"
                f"{message}"
            )

    size_gb = (
        output_path.stat().st_size
        / 1024 ** 3
    )

    print(
        f"[CLIPPED] "
        f"{fixed_grid['xsize']} x "
        f"{fixed_grid['ysize']} | "
        f"{size_gb:.3f} GB | "
        f"ALIGNMENT PASS"
    )

    append_csv(

        CLIP_LOG,

        [
            "time",
            "year",
            "month",
            "xsize",
            "ysize",
            "size_gb",
            "alignment",
            "status",
        ],

        {
            "time": now_string(),

            "year": year,

            "month": (
                f"{month:02d}"
            ),

            "xsize": (
                fixed_grid[
                    "xsize"
                ]
            ),

            "ysize": (
                fixed_grid[
                    "ysize"
                ]
            ),

            "size_gb": (
                f"{size_gb:.3f}"
            ),

            "alignment": "PASS",

            "status": "success",
        },
    )


# 15. Main


def main():

    print(
        "\n"
        + "#" * 100
    )

    print(
        "QTP WATER MASK"
    )

    print(
        "STAGE 1: FAST MOSAIC"
    )

    print(
        "STAGE 2: STRICT UNIFIED CLIP"
    )

    print(
        "#" * 100
    )

    
    # Stage 1
    

    if RUN_STAGE_1_MOSAIC:

        monthly_files = (
            discover_regional_masks()
        )

        if not monthly_files:

            raise RuntimeError(
                "No regional masks found."
            )

        all_keys = sorted(
            monthly_files.keys()
        )

        print(
            f"\nFound "
            f"{len(all_keys)} "
            f"year-months."
        )

        merge_success = 0
        merge_failed = 0

        for (
            year,
            month,
        ) in all_keys:

            try:

                mosaic_one_month(

                    year=year,

                    month=month,

                    region_files=(
                        monthly_files[
                            (year, month)
                        ]
                    ),
                )

                merge_success += 1

            except Exception as e:

                merge_failed += 1

                message = (
                    f"[MERGE FAILED] "
                    f"{year}-{month:02d}\n"
                    f"{repr(e)}\n"
                    f"{traceback.format_exc()}"
                )

                print(
                    message
                )

                append_error(
                    message
                )

                continue

        print(
            "\n"
            + "=" * 100
        )

        print(
            "STAGE 1 FINISHED"
        )

        print(
            f"Success / skipped: "
            f"{merge_success}"
        )

        print(
            f"Failed: "
            f"{merge_failed}"
        )

    
    # Stage 2
    

    if RUN_STAGE_2_CLIP:

        merged_files = (
            discover_merged_files()
        )

        if not merged_files:

            raise RuntimeError(
                "No merged raster files found."
            )

        
        # Reference mosaic
        #
        
        

        reference_path = (
            merged_files[0][2]
        )

        print(
            "\nReference mosaic:"
        )

        print(
            reference_path
        )

        
        
        #
        
        

        fixed_grid = (
            build_fixed_clip_grid(
                reference_path
            )
        )

        print(
            "\n"
            + "=" * 100
        )

        print(
            "FIXED FINAL CLIP GRID"
        )

        print(
            "=" * 100
        )

        print(
            "Extent:"
        )

        print(
            (
                fixed_grid["xmin"],
                fixed_grid["ymin"],
                fixed_grid["xmax"],
                fixed_grid["ymax"],
            )
        )

        print(
            "\nResolution:"
        )

        print(
            fixed_grid["xres"],
            fixed_grid["yres"],
        )

        print(
            "\nSize:"
        )

        print(
            fixed_grid["xsize"],
            "x",
            fixed_grid["ysize"],
        )

        print(
            "\nGeoTransform:"
        )

        print(
            fixed_grid["gt"]
        )

        print(
            "=" * 100
        )

        clip_success = 0
        clip_failed = 0

        for (
            year,
            month,
            input_path,
        ) in merged_files:

            try:

                clip_one_month(

                    year=year,

                    month=month,

                    input_path=(
                        input_path
                    ),

                    fixed_grid=(
                        fixed_grid
                    ),
                )

                clip_success += 1

            except Exception as e:

                clip_failed += 1

                message = (
                    f"[CLIP FAILED] "
                    f"{year}-{month:02d}\n"
                    f"{repr(e)}\n"
                    f"{traceback.format_exc()}"
                )

                print(
                    message
                )

                append_error(
                    message
                )

                continue

        print(
            "\n"
            + "=" * 100
        )

        print(
            "STAGE 2 FINISHED"
        )

        print(
            f"Success / skipped: "
            f"{clip_success}"
        )

        print(
            f"Failed: "
            f"{clip_failed}"
        )

        print(
            "\nALL FINAL CLIPPED FILES "
            "MUST HAVE:"
        )

        print(
            f"RasterXSize = "
            f"{fixed_grid['xsize']}"
        )

        print(
            f"RasterYSize = "
            f"{fixed_grid['ysize']}"
        )

        print(
            "GeoTransform ="
        )

        print(
            fixed_grid["gt"]
        )

    print(
        "\n"
        + "#" * 100
    )

    print(
        "ALL TASKS FINISHED"
    )

    print(
        "#" * 100
    )

    print(
        "\nMerged output:"
    )

    print(
        MERGED_ROOT
    )

    print(
        "\nFinal aligned clipped output:"
    )

    print(
        CLIPPED_ROOT
    )


# 16. Entry


if __name__ == "__main__":
    main()