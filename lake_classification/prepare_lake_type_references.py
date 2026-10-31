# -*- coding: utf-8 -*-
"""Merge glacier outlines and project glacier and permafrost references to the study CRS."""
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import geopandas as gpd
from pyproj import CRS
import fiona

BASE = Path(r'K:\Monthly surface water dynamics TIB\supervised classification\step5_lake_classification')
GLACIER_ROOT = BASE / 'glacier' / 'applied'
GLACIERS = [GLACIER_ROOT / name for name in (
    'RGI2000-v7.0-G-13_central_asia.shp',
    'RGI2000-v7.0-G-14_south_asia_west.shp',
    'RGI2000-v7.0-G-15_south_asia_east.shp',
)]
MERGED = GLACIER_ROOT / 'merged' / 'RGI7_regions_13_14_15.gpkg'
PERMAFROST = BASE / 'permafrost' / 'UiO_PEX_PERZONES_5.0_20181128_2000_2016_NH' / 'UiO_PEX_PERZONES_5.0_20181128_2000_2016_NH.shp'
# Use the CRS of the monthly water bodies.
CRS_REFERENCE = Path(r'E:\phd_l3\QTP_reservoir_separation\water_without_reservoirs\2018\WB_2018_01.gpkg')
OUTPUT = BASE / 'lake_type_references_QTP_Albers'
REBUILD = False
VERSION = '1.0'

def log(message):
    print(f'{datetime.now():%H:%M:%S} {message}', flush=True)

def stamps(paths):
    records = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        parts = [path.with_suffix(s) for s in ('.shp', '.shx', '.dbf', '.prj', '.cpg')] if path.suffix.lower() == '.shp' else [path]
        for part in parts:
            if part.exists():
                st = part.stat()
                records.append([str(part), st.st_size, st.st_mtime_ns])
    return records

def target_crs():
    if not CRS_REFERENCE.is_file():
        raise FileNotFoundError(f'CRS reference file does not exist: {CRS_REFERENCE}; update CRS_REFERENCE.')
    layers = fiona.listlayers(CRS_REFERENCE)
    layer = 'waterbodies' if 'waterbodies' in layers else layers[0]
    with fiona.open(CRS_REFERENCE, layer=layer) as src:
        if not src.crs_wkt and not src.crs:
            raise ValueError('Monthly water body reference has no CRS.')
        target = CRS.from_user_input(src.crs_wkt or src.crs)
    if not target.is_projected or any(abs(axis.unit_conversion_factor - 1) > 1e-9 for axis in target.axis_info[:2]):
        raise ValueError('Reference CRS must be projected in metres.')
    # Check the expected projected CRS.
    expected = CRS.from_proj4('+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs')
    if not target.equals(expected, ignore_axis_order=True):
        log('[WARNING] Reference CRS differs from the expected Albers CRS; using the water body CRS.')
    log('[CRS] ' + target.to_wkt())
    return target

def process(name, paths, target, glacier=False):
    out = OUTPUT / f'{name}.gpkg'
    marker = OUTPUT / f'{name}_completed.json'
    signature = {'version': VERSION, 'inputs': stamps(paths), 'target_wkt': target.to_wkt()}
    if out.exists() and marker.exists() and not REBUILD:
        old = json.loads(marker.read_text(encoding='utf-8'))
        st = out.stat()
        if old.get('signature') == signature and old.get('output_stamp') == [st.st_size, st.st_mtime_ns]:
            log(f'[SKIP] {out}')
            return
    if out.exists() and not REBUILD:
        raise RuntimeError(f'Existing output has an inconsistent cache: {out}. Set REBUILD=True to rebuild.')
    frames = []
    for index, path in enumerate(paths):
        log(f'[READ] {path}')
        frame = gpd.read_file(path)
        if frame.crs is None:
            raise ValueError(f'Input has no CRS: {path}')
        if frame.geometry.isna().any() or frame.geometry.is_empty.any():
            raise ValueError(f'Input contains empty geometries: {path}; inspect the input.')
        if glacier and len(paths) == 3:
            frame['src_region'] = (13, 14, 15)[index]
        log(f'[PROJECT] {len(frame):,} features; projection only, with no clipping, dissolving or simplification')
        frame = frame.to_crs(target)
        frames.append(frame)
    result = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry=frames[0].geometry.name, crs=target)
    temp = OUTPUT / f'{name}_working.gpkg'
    if temp.exists():
        temp.unlink()
    log(f'[WRITE] {out.name}')
    result.to_file(temp, layer=name, driver='GPKG', index=False)
    with fiona.open(temp, layer=name) as src:
        if len(src) != len(result):
            raise RuntimeError('Output feature count does not match.')
        if not CRS.from_user_input(src.crs_wkt).equals(target, ignore_axis_order=True):
            raise RuntimeError('Output CRS does not match.')
    temp.replace(out)
    st = out.stat()
    marker.write_text(json.dumps({'signature': signature, 'count': len(result), 'output_stamp': [st.st_size, st.st_mtime_ns]}, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'[DONE] {out}; {len(result):,} features')

def main():
    log('[START] Preparing glacier and permafrost references')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = target_crs()
    # Reuse the merged file if available; otherwise combine the three regions.
    paths = [MERGED] if MERGED.is_file() else GLACIERS
    process('RGI7_glaciers_QTP_Albers', paths, target, glacier=True)
    process('Obu_permafrost_zones_QTP_Albers', [PERMAFROST], target)
    log('[FINISHED] References use the monthly water body CRS.')

if __name__ == '__main__':
    main()
