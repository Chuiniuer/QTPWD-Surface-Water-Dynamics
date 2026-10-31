# -*- coding: utf-8 -*-
"""
Export labelled and unlabelled water objects to Shapefiles for manual review.
"""

from pathlib import Path
from datetime import datetime
import json
import shutil
import time
import fiona

# Settings

PROCESS_ALL_2018 = True   # False: January only; True: all 2018 months.
YEAR = 2018
MONTH = 1

BASE = Path(
    r"K:\Monthly surface water dynamics TIB"
    r"\supervised classification\step5_lake_classification"
)

INPUT_ROOT = BASE / "monthly_lake_river_attributes"
OUTPUT_ROOT = BASE / "monthly_lake_river_edit_shp"

BATCH_SIZE = 5000
PRINT_EVERY = 100000

# Preserve edits unless rebuilding is explicitly enabled.
REBUILD_CHANGED = False

VERSION = "1.0"

# Shapefile fields are limited to 10 characters.
FIELD_MAP = {
    "obj_id": "obj_id",
    "year": "year",
    "month": "month",
    "area_km2": "area_km2",
    "label": "label",
    "label_state": "lbl_state",
    "res_review": "res_review",
    "ref_valid": "ref_valid",
    "river_frac": "river_frac",
    "lake_frac": "lake_frac",
    "conflict_frac": "conf_frac",
    "perim_m": "perim_m",
    "outer_m": "outer_m",
    "elong": "elong",
    "compact": "compact",
    "rect_fill": "rect_fill",
    "solidity": "solidity",
    "width_m": "width_m",
    "hole_free_c": "fill_comp",
    "hole_frac": "hole_frac",
    "n_parts": "n_parts",
    "n_holes": "n_holes",
}

INTEGER_FIELDS = {
    "obj_id", "year", "month", "label",
    "res_review", "n_parts", "n_holes",
}

# ========================================


def log(message):
    print(
        f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}",
        flush=True,
    )


def file_stamp(path):
    path = Path(path)
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def output_stamps(folder):
    """Record the required Shapefile components."""
    result = {}
    for stem in ("labeled", "unlabeled"):
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            path = folder / f"{stem}{suffix}"
            if not path.is_file():
                return None
            st = path.stat()
            result[path.name] = {
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }
    return result


def export_month(year, month):
    source = (
        INPUT_ROOT / str(year)
        / f"WB_{year}_{month:02d}_attributes.gpkg"
    )
    final_dir = OUTPUT_ROOT / str(year) / f"{month:02d}"
    working_dir = final_dir.with_name(final_dir.name + "_working")
    marker = final_dir / "completed.json"

    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")

    signature = {
        "version": VERSION,
        "input": file_stamp(source),
        "fields": FIELD_MAP,
    }

    # Skip completed exports and preserve manual edits.
    if marker.is_file():
        old = json.loads(marker.read_text(encoding="utf-8"))
        current = output_stamps(final_dir)

        if old.get("signature") == signature and current is not None:
            if current == old.get("outputs"):
                log(f"[SKIP] {year}-{month:02d} already completed")
            else:
                log(
                    f"[SKIP] {year}-{month:02d} already exported; "
                    "keeping existing Shapefiles that may have been edited"
                )
            return

    if final_dir.exists() and not REBUILD_CHANGED:
        raise RuntimeError(
            f"Output exists, but input changed or export is incomplete: {final_dir}\n"
            "Preserve manual edits before rebuilding; "
            "set REBUILD_CHANGED = True to rebuild."
        )

    # Regenerate an interrupted temporary export.
    if working_dir.exists():
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    layers = fiona.listlayers(source)
    if "waterbodies" in layers:
        layer = "waterbodies"
    elif len(layers) == 1:
        layer = layers[0]
    else:
        raise ValueError(f"Cannot determine input layer: {layers}")

    start = time.monotonic()
    total = labeled = unlabeled = 0
    last_print = 0

    with fiona.open(source, layer=layer) as src:
        available = src.schema["properties"]
        if "label" not in available:
            raise ValueError("Input is missing the label field.")
        if "obj_id" not in available:
            raise ValueError("Input is missing the obj_id field.")
        if not src.crs_wkt and not src.crs:
            raise ValueError("Input has no coordinate reference system.")

        selected = {
            original: short
            for original, short in FIELD_MAP.items()
            if original in available
        }

        properties = {}
        for original, short in selected.items():
            if original == "label_state":
                properties[short] = "str:40"
            elif original in INTEGER_FIELDS:
                properties[short] = "int:18"
            else:
                properties[short] = "float:24.12"

        schema = {
            # Shapefile Polygon can store single- and multipart geometries.
            "geometry": "Polygon",
            "properties": properties,
        }

        options = {
            "driver": "ESRI Shapefile",
            "schema": schema,
            "crs_wkt": src.crs_wkt,
            "encoding": "UTF-8",
            # Stop before exceeding the Shapefile size limit.
            "2GB_LIMIT": "YES",
        }

        log(f"[START] {year}-{month:02d}; {len(src):,} objects")

        with fiona.open(
            working_dir / "labeled.shp", "w", **options
        ) as dst_labeled, fiona.open(
            working_dir / "unlabeled.shp", "w", **options
        ) as dst_unlabeled:

            batches = [[], []]
            writers = [dst_labeled, dst_unlabeled]

            for feature in src:
                attrs = dict(feature["properties"])
                value = attrs.get("label")

                if value in (1, 2):
                    group = 0
                    labeled += 1
                elif value is None or value == 0:
                    group = 1
                    unlabeled += 1
                else:
                    raise ValueError(
                        f"Unexpected label={value}, "
                        f"obj_id={attrs.get('obj_id')}; export stopped."
                    )

                geom = feature["geometry"]
                if geom is not None and geom["type"] not in (
                    "Polygon", "MultiPolygon"
                ):
                    raise ValueError(
                        f"Non-polygon geometry: {geom['type']}, "
                        f"obj_id={attrs.get('obj_id')}"
                    )

                batches[group].append({
                    "geometry": geom,
                    "properties": {
                        short: attrs.get(original)
                        for original, short in selected.items()
                    },
                })

                if len(batches[group]) >= BATCH_SIZE:
                    writers[group].writerecords(batches[group])
                    batches[group].clear()

                total += 1
                if total - last_print >= PRINT_EVERY:
                    log(
                        f"[PROGRESS] {total:,} ; "
                        f"labelled {labeled:,}, unlabelled {unlabeled:,} ; "
                        f"elapsed {(time.monotonic()-start)/60:.1f} min"
                    )
                    last_print = total

            for writer, batch in zip(writers, batches):
                if batch:
                    writer.writerecords(batch)

    # Verify exported object counts.
    with fiona.open(working_dir / "labeled.shp") as ds:
        actual_labeled = len(ds)
    with fiona.open(working_dir / "unlabeled.shp") as ds:
        actual_unlabeled = len(ds)

    if (actual_labeled, actual_unlabeled) != (labeled, unlabeled):
        raise RuntimeError("Output object counts differ; no completion marker written.")

    mapping_lines = [
        "Source field\tShapefile field",
        *[f"{a}\t{b}" for a, b in selected.items()],
    ]
    (working_dir / "字段对照表.tsv").write_text(
        "\n".join(mapping_lines), encoding="utf-8-sig"
    )

    (working_dir / "说明.txt").write_text(
        f"Month: {year}-{month:02d}\n"
        f"Input: {source}\n"
        f"Labelled objects: {labeled:,}; unlabelled objects: {unlabeled:,}.\n"
        "Edit label: 1 = lake, 2 = river, 0 = undetermined.\n"
        "Use obj_id to join to the source GeoPackage; do not use exported FID.\n"
        "Keep all Shapefile components together.\n",
        encoding="utf-8-sig",
    )

    # Back up existing outputs before rebuilding.
    if final_dir.exists():
        backup = final_dir.with_name(
            final_dir.name + "_backup_"
            + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        final_dir.rename(backup)
        log(f"[BACKUP] Previous output: {backup}")

    working_dir.rename(final_dir)

    outputs = output_stamps(final_dir)
    if outputs is None:
        raise RuntimeError("Shapefile components are incomplete; no completion marker written.")

    record = {
        "signature": signature,
        "outputs": outputs,
        "total": total,
        "labeled": labeled,
        "unlabeled": unlabeled,
        "seconds": time.monotonic() - start,
    }
    marker.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log(
        f"[DONE] {year}-{month:02d}: "
        f"labelled {labeled:,}; unlabelled {unlabeled:,}"
    )
    log(f"[OUTPUT] {final_dir}")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    months = range(1, 13) if PROCESS_ALL_2018 else [MONTH]

    for month in months:
        export_month(YEAR, month)

    log("All exports completed.")


if __name__ == "__main__":
    main()