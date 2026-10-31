# -*- coding: utf-8 -*-
"""Prepare cached QTP river/lake references. Run directly with Python 3.9+.
Dependencies: numpy rasterio fiona shapely>=2 pyproj.
No monthly classification and no modification of source data.
Outputs are references, not verified river/lake classifications.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import sys
import time
import traceback

VERSION = '1.0.0'
ROOT = Path(r'D:\phdl2\QTP surface water dynamics')
DEFAULT_OUT = Path(r'K:\Monthly surface water dynamics TIB\supervised classification\step5_lake_classification\reference_data')
PRODUCTS = ['Combined_river_mask_10m', 'River_source_flags_10m',
            'HMA_lake_support_10m', 'Reference_validity_10m', 'HMA_river_mask_10m']
LOG = logging.getLogger('QTP')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rivers', type=Path, default=ROOT / 'Vector dataset of rivers' / 'Water_Vector' / 'Rivers.shp')
    p.add_argument('--lakes', type=Path, default=ROOT / 'Vector dataset of rivers' / 'Water_Vector' / 'lakes&reservoirs.shp')
    p.add_argument('--grwl', type=Path, default=ROOT / 'GRWL_water_line' / 'GRWL_QTP_waterway_mask_30m.tif')
    p.add_argument('--osm', type=Path, default=ROOT / 'OSM_water_layer' / 'OSM_water_line' / 'OSM_QTP_waterway_mask_30m.tif')
    p.add_argument('--boundary', type=Path, default=ROOT / 'regions' / 'QTP.shp')
    p.add_argument('--crs-reference', type=Path, default=Path(r'E:\phd_l3\QTPWD_vector\2016\QTP_waterbody_2016.gdb'))
    p.add_argument('--crs-layer', default='WB_2016_01')
    p.add_argument('--output', type=Path, default=DEFAULT_OUT)
    p.add_argument('--tile-size', type=int, default=2048, help='Processing tile pixels (multiple of 256).')
    p.add_argument('--threads', type=int, default=2, help='GDAL warp threads; not monthly worker count.')
    p.add_argument('--grwl-zero-is-background', action='store_true', help='Explicit override ONLY if GRWL metadata incorrectly labels background 0 as NoData.')
    p.add_argument('--osm-zero-is-background', action='store_true', help='Explicit override ONLY if OSM metadata incorrectly labels background 0 as NoData.')
    p.add_argument('--check-only', action='store_true', help='Inspect inputs/grid without processing data.')
    p.add_argument('--rebuild', action='store_true', help='Start a new cache generation; keep previous runs.')
    p.add_argument('--break-lock', action='store_true', help='Remove stale lock ONLY after confirming no previous instance is running.')
    return p.parse_args()


def dependencies():
    global np, fiona, rio, CRS, Transformer, shape, mapping, box, MultiPolygon
    global transform_geom, unary_union, make_valid, rasterize, WarpedVRT, Resampling, Window, Affine
    import numpy as np
    import fiona
    import rasterio as rio
    from pyproj import CRS, Transformer
    from shapely.geometry import shape, mapping, box, MultiPolygon
    from shapely.ops import transform as transform_geom, unary_union
    from shapely import make_valid
    from rasterio.features import rasterize
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from affine import Affine


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def stamp(path):
    path = Path(path)
    if path.is_dir():
        files = sorted(x for x in path.rglob('*') if x.is_file() and not x.name.endswith('.lock'))
    elif path.suffix.lower() == '.shp':
        files = sorted(x for x in path.parent.glob(path.stem + '.*') if x.is_file())
    else:
        files = [path]
        for tail in ['.msk', '.aux.xml']:
            x = Path(str(path) + tail)
            if x.exists():
                files.append(x)
    return [[str(x.resolve()), x.stat().st_size, x.stat().st_mtime_ns] for x in files]


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:24]


def polygons(g):
    if g is None or g.is_empty:
        return None
    if not g.is_valid:
        g = make_valid(g)
    if g.geom_type == 'Polygon':
        return MultiPolygon([g])
    if g.geom_type == 'MultiPolygon':
        return g
    if g.geom_type == 'GeometryCollection':
        parts = []
        for child in g.geoms:
            sub = polygons(child)
            if sub is not None:
                parts.extend(sub.geoms)
        return MultiPolygon(parts) if parts else None
    return None


def load_boundary(path, target):
    parts = []
    with fiona.open(path) as src:
        if not src.crs_wkt and not src.crs:
            raise ValueError('QTP boundary has no CRS')
        tr = Transformer.from_crs(src.crs_wkt or src.crs, target, always_xy=True)
        for f in src:
            if f['geometry']:
                g = polygons(transform_geom(tr.transform, shape(f['geometry'])))
                if g is not None:
                    parts.append(g)
    if not parts:
        raise ValueError('Empty QTP polygon boundary')
    return polygons(unary_union(parts))


def prepare_vector(source, output, boundary, target, base_key):
    key = digest([VERSION, base_key, stamp(source), 'vector-center-mask'])
    meta = output.with_suffix('.json')
    old = read_json(meta)
    if old and old.get('key') == key and output.exists() and old.get('output_stamp') == stamp(output):
        LOG.info('[CACHE] %s', output.name)
        return old
    temp = output.with_name(output.stem + '.working.gpkg')
    if temp.exists():
        temp.unlink()
    result = {'key': key, 'source': str(source), 'selected': 0, 'written': 0,
              'empty_or_nonpolygon': 0, 'invalid_repaired': 0}
    LOG.info('[VECTOR] Project and clip: %s', source)
    with fiona.open(source) as src:
        source_crs = CRS.from_user_input(src.crs_wkt or src.crs)
        tr = Transformer.from_crs(source_crs, target, always_xy=True)
        inv = Transformer.from_crs(target, source_crs, always_xy=True)
        bbox = inv.transform_bounds(*boundary.bounds, densify_pts=101)
        props = dict(src.schema['properties'])
        if 'q_refid' in props or 'q_srcfid' in props:
            raise ValueError('Source already has q_refid/q_srcfid; rename those source fields in a copy first.')
        props.update(q_refid='int64', q_srcfid='str')
        result['source_fields'] = dict(src.schema['properties'])
        result['source_crs'] = source_crs.to_wkt()
        batch = []
        last = time.monotonic()
        with fiona.open(temp, 'w', driver='GPKG', layer='reference',
                        schema={'geometry': 'MultiPolygon', 'properties': props},
                        crs_wkt=target.to_wkt(), SPATIAL_INDEX='YES') as dst:
            for feature in src.filter(bbox=bbox):
                result['selected'] += 1
                if not feature['geometry']:
                    result['empty_or_nonpolygon'] += 1
                    continue
                raw = shape(feature['geometry'])
                if not raw.is_valid:
                    result['invalid_repaired'] += 1
                g = polygons(raw)
                if g is None:
                    result['empty_or_nonpolygon'] += 1
                    continue
                g = polygons(transform_geom(tr.transform, g))
                if g is None or not boundary.intersects(g):
                    continue
                if not boundary.covers(g):
                    g = polygons(g.intersection(boundary))
                if g is None:
                    continue
                result['written'] += 1
                attrs = dict(feature['properties'])
                attrs.update(q_refid=result['written'], q_srcfid=str(feature['id']))
                batch.append({'geometry': mapping(g), 'properties': attrs})
                if len(batch) >= 1000:
                    dst.writerecords(batch)
                    batch.clear()
                if time.monotonic() - last > 15:
                    LOG.info('[VECTOR] selected=%s retained=%s', result['selected'], result['written'])
                    last = time.monotonic()
            if batch:
                dst.writerecords(batch)
    if not result['written']:
        raise ValueError('No polygon features intersect QTP: ' + str(source))
    os.replace(temp, output)
    result['output_stamp'] = stamp(output)
    atomic_json(meta, result)
    return result


def profile(width, height, transform, crs, count=1):
    return dict(driver='GTiff', width=width, height=height, count=count,
                dtype='uint8', crs=crs, transform=transform, nodata=255,
                tiled=True, blockxsize=256, blockysize=256,
                compress='DEFLATE', zlevel=6, predictor=2, BIGTIFF='YES')


def burn(src, bounds, transform, shape_out):
    # Fiona's GPKG spatial index limits reads to this processing tile.
    geoms = [(f['geometry'], 1) for f in src.filter(bbox=bounds) if f['geometry']]
    if not geoms:
        return np.zeros(shape_out, dtype='uint8')
    return rasterize(geoms, out_shape=shape_out, transform=transform,
                     fill=0, all_touched=False, dtype='uint8')


def raster_info(path, zero_override):
    with rio.open(path) as src:
        if src.count != 1 or not src.crs:
            raise ValueError('Expected one-band raster with CRS: ' + str(path))
        if src.nodata == 0 and not zero_override:
            raise ValueError(f'{path}: NoData=0 is ambiguous for a binary reference. '
                             'Confirm whether 0 means valid background before using the corresponding '
                             '--grwl-zero-is-background or --osm-zero-is-background option.')
        return {'path': str(path), 'crs': src.crs.to_wkt(), 'transform': list(src.transform),
                'width': src.width, 'height': src.height, 'dtype': src.dtypes[0],
                'nodata': str(src.nodata), 'zero_override': zero_override}


def raster_mask(vrt, window, inside, label):
    a = vrt.read(1, window=window, masked=True)
    valid = ~np.ma.getmaskarray(a) & inside
    vals = a.data
    bad = valid & (vals != 0) & (vals != 1)
    if bad.any():
        raise ValueError(f'{label}: expected processed binary 0/1 mask; found {np.unique(vals[bad])[:20]}. '
                         'Do not supply the original GRWL/OSM multitype product.')
    return (valid & (vals == 1)).astype('uint8'), valid


def tile_counts(data, inside):
    flags = data[1][inside]
    lake = data[2][inside] == 1
    river = data[0][inside] == 1
    validity = data[3][inside]
    return {'qtp_pixels': int(inside.sum()),
            'source_flags_pixels': np.bincount(flags, minlength=8).tolist(),
            'lake_pixels': int(lake.sum()), 'river_pixels': int(river.sum()),
            'lake_river_overlap_pixels': int((lake & river).sum()),
            'validity_pixels': np.bincount(validity, minlength=8).tolist()}


def process_tiles(args, run, key, grid, boundary, rivers, lakes):
    tiles = run / 'tiles'
    tiles.mkdir(exist_ok=True)
    width, height, transform, crs = grid
    total = math.ceil(width / args.tile_size) * math.ceil(height / args.tile_size)
    records = []
    with contextlib.ExitStack() as stack:
        rv = stack.enter_context(fiona.open(rivers, layer='reference'))
        lv = stack.enter_context(fiona.open(lakes, layer='reference'))
        vrts = []
        for path, override in [(args.grwl, args.grwl_zero_is_background), (args.osm, args.osm_zero_is_background)]:
            src = stack.enter_context(rio.open(path))
            options = dict(crs=crs, transform=transform, width=width, height=height,
                           nodata=255, dtype='uint8', resampling=Resampling.nearest,
                           warp_mem_limit=128, tolerance=0.001, NUM_THREADS=str(args.threads))
            if override:
                options['src_nodata'] = 255
            vrt = stack.enter_context(WarpedVRT(src, **options))
            vrts.append(vrt)
        index = 0
        last = time.monotonic()
        for row in range(0, height, args.tile_size):
            for col in range(0, width, args.tile_size):
                index += 1
                w, h = min(args.tile_size, width-col), min(args.tile_size, height-row)
                window = Window(col, row, w, h)
                tf = rio.windows.transform(window, transform)
                bounds = rio.windows.bounds(window, transform)
                tag = f'r{row:07d}_c{col:07d}'
                path = tiles / (tag + '.tif')
                meta = tiles / (tag + '.json')
                cached = read_json(meta)
                if cached and cached.get('key') == key and (cached.get('empty') or
                        (path.exists() and cached.get('stamp') == stamp(path))):
                    records.append(cached)
                else:
                    clipped = boundary.intersection(box(*bounds))
                    inside = (rasterize([(mapping(clipped), 1)], out_shape=(h, w), transform=tf,
                                        dtype='uint8', all_touched=False).astype(bool)
                              if not clipped.is_empty else np.zeros((h, w), bool))
                    record = {'key': key, 'row': row, 'col': col, 'width': w, 'height': h,
                              'file': str(path), 'empty': not bool(inside.any())}
                    if inside.any():
                        hma = burn(rv, bounds, tf, (h, w))
                        lake = burn(lv, bounds, tf, (h, w))
                        grwl, gv = raster_mask(vrts[0], window, inside, 'GRWL')
                        osm, ov = raster_mask(vrts[1], window, inside, 'OSM')
                        flags = hma | (grwl << 1) | (osm << 2)
                        validity = inside.astype('uint8') * 4 | gv.astype('uint8') | (ov.astype('uint8') << 1)
                        data = np.stack([(flags != 0).astype('uint8'), flags, lake, validity, hma])
                        data[:, ~inside] = 255
                        record['counts'] = tile_counts(data, inside)
                        temp = path.with_suffix('.working.tif')
                        with rio.open(temp, 'w', **profile(w, h, tf, crs, count=5)) as dst:
                            dst.write(data)
                            dst.update_tags(cache_key=key, version=VERSION)
                        os.replace(temp, path)
                        record['stamp'] = stamp(path)
                    atomic_json(meta, record)
                    records.append(record)
                if time.monotonic() - last > 15 or index == total or index == 1:
                    LOG.info('[TILES] %s/%s (completed tiles reused automatically)', index, total)
                    last = time.monotonic()
    return records


def export_products(run, key, grid, records):
    width, height, transform, crs = grid
    products = []
    for band, name in enumerate(PRODUCTS, 1):
        path = run / (name + '.tif')
        meta = path.with_suffix('.json')
        old = read_json(meta)
        if old and old.get('key') == key and path.exists() and old.get('stamp') == stamp(path):
            LOG.info('[CACHE] %s', path.name)
            products.append(str(path))
            continue
        LOG.info('[EXPORT] %s (tile copy only; no reprojection)', path.name)
        temp = path.with_suffix('.working.tif')
        last = time.monotonic()
        with rio.open(temp, 'w', **profile(width, height, transform, crs)) as dst:
            for i, r in enumerate(records, 1):
                window = Window(r['col'], r['row'], r['width'], r['height'])
                if r['empty']:
                    data = np.full((r['height'], r['width']), 255, dtype='uint8')
                else:
                    with rio.open(r['file']) as src:
                        data = src.read(band)
                dst.write(data, 1, window=window)
                if time.monotonic() - last > 15:
                    LOG.info('[EXPORT] %s %s/%s', name, i, len(records))
                    last = time.monotonic()
            dst.set_band_description(1, name)
            dst.update_tags(cache_key=key, version=VERSION, area_unit='pixel = 100 square metres',
                            semantics='Positive reference evidence; zero does not establish absence of a real river/lake.',
                            sampling='nearest and pixel-centre rasterization; no all_touched; no buffers')
        os.replace(temp, path)
        atomic_json(meta, {'key': key, 'stamp': stamp(path)})
        products.append(str(path))
    return products


def main(args):
    if args.tile_size < 256 or args.tile_size > 4096 or args.tile_size % 256:
        raise ValueError('--tile-size must be a multiple of 256 between 256 and 4096')
    if not 1 <= args.threads <= 8:
        raise ValueError('--threads must be 1..8')
    for name in ['rivers', 'lakes', 'grwl', 'osm', 'boundary', 'crs_reference']:
        path = getattr(args, name)
        if not path.exists():
            raise FileNotFoundError(f'{name}: {path}')
    with fiona.open(args.crs_reference, layer=args.crs_layer) as src:
        target = CRS.from_user_input(src.crs_wkt or src.crs)
    method = target.coordinate_operation.method_name if target.coordinate_operation else ''
    if not target.is_projected or 'albers' not in method.lower() or any(
            abs(ax.unit_conversion_factor - 1) > 1e-10 for ax in target.axis_info[:2]):
        raise ValueError('Monthly reference must use Albers with metre axes; CRS name alone is not used.')
    LOG.info('[CRS] %s; %s', target.name, method)
    infos = [raster_info(args.grwl, args.grwl_zero_is_background), raster_info(args.osm, args.osm_zero_is_background)]
    boundary = load_boundary(args.boundary, target)
    with rio.open(args.grwl) as src:
        if not CRS.from_user_input(src.crs).equals(target, ignore_axis_order=True):
            raise ValueError('GRWL template CRS differs from monthly Albers CRS; use the previously aligned GRWL product.')
        t = src.transform
        if abs(t.b) > 1e-9 or abs(t.d) > 1e-9 or t.a <= 0 or t.e >= 0:
            raise ValueError('GRWL template must be north-up.')
        if abs(t.a / 10 - round(t.a / 10)) > 1e-8 or abs(-t.e / 10 - round(-t.e / 10)) > 1e-8:
            raise ValueError('GRWL pixel dimensions must be integer multiples of 10 m.')
        xmin, ymin, xmax, ymax = boundary.bounds
        x0 = t.c + math.floor((xmin-t.c)/10)*10
        y0 = t.f + math.ceil((ymax-t.f)/10)*10
        width, height = math.ceil((xmax-x0)/10), math.ceil((y0-ymin)/10)
    transform = Affine(10, 0, x0, 0, -10, y0)
    grid = width, height, transform, target.to_wkt()
    params = {'version': VERSION, 'inputs': {k: stamp(getattr(args, k)) for k in
              ['rivers', 'lakes', 'grwl', 'osm', 'boundary']},
              'crs': target.to_wkt(), 'crs_reference': str(args.crs_reference), 'crs_layer': args.crs_layer,
              'grid': {'width': width, 'height': height, 'transform': list(transform)},
              'tile_size': args.tile_size, 'raster_inputs': infos,
              'all_touched': False, 'resampling': 'nearest', 'resolution_m': 10}
    key = digest(params)
    LOG.info('[GRID] %s x %s, 10 m; %.1f GB per uncompressed Byte layer', width, height, width*height/1e9)
    LOG.info('[GRID] No full-grid arrays will be allocated.')
    if args.check_only:
        atomic_json(args.output / 'input_check.json', params)
        LOG.info('[CHECK ONLY] Finished; no masks created.')
        return
    # Run folders isolate changed inputs and prevent stale complete products being mistaken for new ones.
    runs = args.output / 'runs'
    runs.mkdir(exist_ok=True)
    active = read_json(args.output / 'active_run.json')
    if not args.rebuild and active and active.get('key') == key and Path(active['run']).is_dir():
        run = Path(active['run'])
    else:
        suffix = time.strftime('_%Y%m%d_%H%M%S') if args.rebuild else ''
        run = runs / (key + suffix)
        run.mkdir(exist_ok=True)
        atomic_json(args.output / 'active_run.json', {'key': key, 'run': str(run)})
    atomic_json(run / 'parameters.json', params)
    LOG.info('[RUN] %s', run)
    vector_key = digest([VERSION, stamp(args.boundary), target.to_wkt(),
                         stamp(args.rivers), stamp(args.lakes)])
    # Vector cache is independent of processing tile size and raster inputs.
    vdir = args.output / 'vector_cache' / vector_key
    vdir.mkdir(parents=True, exist_ok=True)
    rivers, lakes = vdir / 'HMA_rivers_QTP.gpkg', vdir / 'HMA_lakes_reservoirs_QTP.gpkg'
    reports = [prepare_vector(args.rivers, rivers, boundary, target, vector_key),
               prepare_vector(args.lakes, lakes, boundary, target, vector_key)]
    records = process_tiles(args, run, key, grid, boundary, rivers, lakes)
    products = export_products(run, key, grid, records)
    totals = {'qtp_pixels': 0, 'river_pixels': 0, 'lake_pixels': 0, 'lake_river_overlap_pixels': 0,
              'source_flags_pixels': [0]*8, 'validity_pixels': [0]*8}
    for r in records:
        c = r.get('counts', {})
        for k in totals:
            if isinstance(totals[k], list):
                totals[k] = [x+y for x, y in zip(totals[k], c.get(k, [0]*8))]
            else:
                totals[k] += c.get(k, 0)
    flags = totals['source_flags_pixels']
    report = {'version': VERSION, 'key': key, 'status': 'COMPLETE', 'products': products,
              'reference_vectors': [str(rivers), str(lakes)], 'vectors': reports, 'parameters': params,
              'counts': totals,
              'areas_km2': {k.replace('_pixels', ''): v * 0.0001 for k, v in totals.items() if isinstance(v, int)},
              'source_coverage_km2': {name: sum(flags[i] for i in range(8) if i & bit)*0.0001
                                      for name, bit in [('HMA', 1), ('GRWL', 2), ('OSM', 4)]},
              'source_flag_areas_km2': [x * 0.0001 for x in flags],
              'notes': [
                  'Areas are centre-sampled on a 10 m Albers grid, not original vector areas.',
                  'Source bits: HMA=1, GRWL=2, OSM=4. Combined river=1 where any bit is present.',
                  'All mask files: outside QTP=255 (NoData). Inside QTP 0 means no positive reference evidence, not proven nonwater.',
                  'Validity bits: GRWL valid raster data=1; OSM valid raster data=2; inside QTP=4. Therefore inside codes are 4,5,6,7. This does NOT describe real-world mapping completeness.',
                  'HMA vector study coverage is not supplied: absence of a polygon is not a verified negative observation.',
                  'Lake support includes lakes and reservoirs; it does not establish natural-lake identity.',
                  'No buffers, vector simplification, lake-river erasure, slope filters or monthly classification.',
                  'The GRWL/OSM native resolution is not improved by nearest-neighbour resampling.',
                  'Incomplete raster tiles are recomputed. Interrupted final exports recopy cached tiles without reprojecting.',
                  'Vector preparation restarts its current layer if interrupted; already completed layers are reused.',
                  'Keep tile caches for restart. Do not run two instances in the same output directory.']}
    atomic_json(run / 'reference_report.json', report)
    atomic_json(args.output / 'reference_report.json', report)
    atomic_json(args.output / 'current_reference.json', {'status': 'COMPLETE', 'key': key, 'run': str(run),
                'products': dict(zip(PRODUCTS, products)), 'rivers_vector': str(rivers), 'lakes_vector': str(lakes)})
    LOG.info('[DONE] River union %.2f km2; lake reference %.2f km2; overlap %.2f km2',
             report['areas_km2']['river'], report['areas_km2']['lake'], report['areas_km2']['lake_river_overlap'])
    LOG.info('[DONE] Output index: %s', args.output / 'current_reference.json')


def entrypoint():
    print('[START] QTP reference preparation ' + VERSION, flush=True)
    print('[SCRIPT] ' + str(Path(__file__).resolve()), flush=True)
    print('[PYTHON] ' + sys.executable, flush=True)
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.output / 'processing.log', encoding='utf-8')])
    lock = args.output / 'processing.lock'
    if args.break_lock and lock.exists():
        lock.unlink()
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Output directory is locked. If the previous process is no longer running, use --break-lock.')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(json.dumps({'pid': os.getpid(), 'started': time.strftime('%Y-%m-%d %H:%M:%S')}))
        dependencies()
        with rio.Env(GDAL_CACHEMAX=256*1024*1024):
            main(args)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        entrypoint()
    except KeyboardInterrupt:
        print('[STOPPED] Finished tiles are cached; rerun to resume.', flush=True)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        if LOG.handlers or logging.getLogger().handlers:
            LOG.exception('[FAILED] Source data were not modified. Check processing.log.')
        sys.exit(1)
