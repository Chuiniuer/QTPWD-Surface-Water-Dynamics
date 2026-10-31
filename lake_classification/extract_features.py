# -*- coding: utf-8 -*-
"""Prepare shared rasters and extract monthly lake/river features.
Dependencies: numpy, pandas, shapely>=2, pyproj, rasterio and pyarrow.
Raster preparation also requires GDAL; reading a Shapefile CRS requires pyogrio.
No source geometry is modified.

Outputs: OUT_ROOT / year / WB_YYYY_MM / <input-and-settings fingerprint>/
         features/part-000000.parquet ... ; manifest.json ; DONE.json
Read a completed month with pandas.read_parquet(run_dir / 'features').
For millions of objects prefer pyarrow.dataset.dataset(...).to_batches().
Join to manifest.source_path + source_layer using source_fid, NOT output row order.
obj_id is preserved when present; otherwise obj_id = source_fid, explicitly recorded.
"""
from pathlib import Path
import os
import sys
import json
import hashlib
import sqlite3
import time
import traceback
from collections import OrderedDict
from contextlib import ExitStack, closing
from concurrent.futures import ProcessPoolExecutor, as_completed

# ====================== EDIT SETTINGS HERE ======================
BASE = Path(r'K:\Monthly surface water dynamics TIB\supervised classification\step5_lake_classification')
ATTR_ROOT = BASE / 'monthly_lake_river_attributes'
RAW_ROOT = Path(r'E:\phd_l3\QTP_reservoir_separation\water_without_reservoirs')
OUT_ROOT = Path(r'E:\phd_l3\QTP_monthly_lake_river_features')
TERRAIN_ROOT = Path(r'E:\phd_l3\terrain_data')
REFERENCE_ROOT = BASE / 'external_features_2018_01'  # Recursively find the existing distance caches

ALL_YEARS = True             # True: 2016--2025; False: SELECT_YEARS
SELECT_YEARS = [2018]
MONTHS = list(range(1, 13))   # e.g. [1] for January only
CHUNK_SIZE = 50_000
MAX_WORKERS = 1              # Start with 1 on an HDD; 2 may help on SSD. Avoid many workers.
COMPUTE_MISSING_SHAPE = True # Reuse existing shape columns; calculate missing COLUMNS only.
RECOMPUTE_ALL_SHAPE = False  # Do not enable just to append four external features.
CHECK_GEOMETRY_VALIDITY = False  # Extra topology check; no geometry repair/modification.
RASTER_CACHE_MB = 64         # Per raster, per worker; 4 rasters ~256 MB maximum LRU storage.

# Fill explicit paths if discovery finds more than one valid candidate.
# Reuse the same corrected 30 m rasters used by the existing model.
# Do not choose the masks (0/1) as distance rasters.
RASTERS = {
    'elev_m': None,          # e.g. Path(r'E:\phd_l3\terrain_data\QTP_elev_30m_....tif')
    'slope_deg': None,       # known pattern QTP_slope_deg_30m_*.tif
    'dist_riv_m': None,      # existing river distance/proximity TIFF, in metres
    'dist_lak_m': None,      # existing lake distance/proximity TIFF, in metres
}
# If a GPKG has multiple polygon layers, specify the layer for that month here.
LAYER_OVERRIDES = {}         # e.g. {'2018_01': 'waterbodies'}
# ===============================================================


# Set PREPARE_RASTERS once to build the shared inputs; subsequent runs can
# reuse existing rasters. Set RUN_EXTRACTION to False for preparation only.
PREPARE_RASTERS = False
RUN_EXTRACTION = True
PREP_DEM_SOURCE = Path(r'E:\phd_l3\terrain_data\QTP_elev.tif')
PREP_BASE = BASE
PREP_CACHE = REFERENCE_ROOT / 'reference_cache'
PREP_RIVER_MASK = None
PREP_LAKE_MASK = None
PREP_RIVER_FILENAME = 'Combined_river_mask_10m.tif'
PREP_LAKE_FILENAME = 'HMA_lake_support_10m.tif'
# If this existing January label layer is unavailable, use a monthly GPKG.
PREP_CRS_REFERENCE = BASE / 'monthly_lake_river_edit_shp' / '2018' / '01' / 'labeled.shp'
PREP_CRS_GPKG = None  # Set a monthly GPKG path to override automatic discovery.
PREP_CRS_MONTH = '2018_01'  # For LAYER_OVERRIDES if PREP_CRS_GPKG is set.
PREP_RESOLUTION = 30.0
PREP_GDAL_CACHE_MB = 512
PREP_WARP_MEMORY_MB = 256
PREP_NUM_THREADS = 4
PREP_REFERENCE_TARGET_VALUE = 1
PREP_REFERENCE_NODATA = 255
PREP_DEM_VERTICAL_UNIT = 'metre'
PREP_VERSION = 'external_features_v1'
PREP_REFERENCE_VERSION = 'reference_nodata255_v2'
PREP_TIFF_OPTIONS = ['TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512',
                     'COMPRESS=DEFLATE', 'BIGTIFF=YES', 'NUM_THREADS=4']

VERSION = 'qtp_features_20260929_v1'
MORPH = ['area_km2', 'perim_m', 'outer_m', 'elong', 'compact', 'rect_fill',
         'solidity', 'width_m', 'hole_free_c', 'hole_frac', 'n_holes']
EXTERNAL = ['elev_m', 'slope_deg', 'dist_riv_m', 'dist_lak_m']
np = pd = pa = pq = sh = rasterio = CRS = None


def dependencies():
    global np, pd, pa, pq, sh, rasterio, CRS
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import shapely as sh
    import rasterio
    from pyproj import CRS
    if int(sh.__version__.split('.')[0]) < 2:
        raise RuntimeError('Shapely >= 2 is required for vectorized geometry operations.')


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def jsave(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def statkey(path):
    p = Path(path).resolve()
    st = p.stat()
    return {'path': str(p), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def connect(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=60)


def source_info(path, ym):
    # Refuse an active SQLite WAL: a changing input invalidates repeatable extraction.
    wal = Path(str(path) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError(f'Close the editing program and commit/checkpoint this GPKG first: {path}')
    with closing(connect(path)) as con:
        layers = con.execute('SELECT table_name, column_name, geometry_type_name, srs_id FROM gpkg_geometry_columns').fetchall()
        override = LAYER_OVERRIDES.get(ym)
        if override:
            layers = [r for r in layers if r[0] == override]
        else:
            layers = [r for r in layers if r[2].upper() in ('POLYGON', 'MULTIPOLYGON', 'GEOMETRY')]
        if len(layers) != 1:
            raise RuntimeError(f'Specify LAYER_OVERRIDES[{ym!r}]; candidate layers: {layers}')
        layer, geom, _, srsid = layers[0]
        fields = con.execute(f'PRAGMA table_info({quote(layer)})').fetchall()
        pks = [r[1] for r in fields if r[5]]
        if len(pks) != 1 or next(r[2] for r in fields if r[1] == pks[0]).upper() != 'INTEGER':
            raise RuntimeError(f'Expected one INTEGER feature primary key: {path}')
        definition = con.execute('SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id=?', (srsid,)).fetchone()[0]
        crs = CRS.from_user_input(definition)
        if not crs.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-9 for a in crs.axis_info[:2]):
            raise RuntimeError(f'Input must already use projected metre coordinates: {path}')
        count = con.execute(f'SELECT COUNT(*) FROM {quote(layer)}').fetchone()[0]
        fields = [r[1] for r in fields]
        lower = {n.lower(): n for n in fields}
        # Ensure existing object IDs support unambiguous joins. This scans one column once.
        if 'obj_id' in lower:
            oid = quote(lower['obj_id'])
            bad = con.execute(f'SELECT {oid} FROM {quote(layer)} GROUP BY {oid} '
                              f'HAVING {oid} IS NULL OR COUNT(*)>1 LIMIT 1').fetchone()
            if bad is not None:
                raise RuntimeError(f'Existing obj_id is null or duplicated ({bad[0]}): {path}')
    return dict(source_path=str(path), source_layer=layer, geometry_column=geom,
                pk=pks[0], fields=fields, crs=crs.to_wkt(), count=count,
                obj_id_rule='existing obj_id' if 'obj_id' in lower else 'source_fid (local to this source layer)')


def raster_ok(path):
    with rasterio.open(path) as src:
        crs = CRS.from_user_input(src.crs) if src.crs else None
        return (src.count == 1 and crs is not None and crs.is_projected
                and all(abs(a.unit_conversion_factor - 1) < 1e-9 for a in crs.axis_info[:2])
                and abs(src.res[0] - 30) < .01 and abs(src.res[1] - 30) < .01)


def resolve_rasters():
    terrain = list(TERRAIN_ROOT.rglob('*.tif')) if TERRAIN_ROOT.exists() else []
    refs = list(REFERENCE_ROOT.rglob('*.tif')) if REFERENCE_ROOT.exists() else []
    result = {}
    for key in EXTERNAL:
        explicit = RASTERS[key]
        if explicit:
            choices = [Path(explicit)]
        else:
            choices = []
            for p in (terrain if key in ('elev_m', 'slope_deg') else refs):
                n = p.stem.lower()
                if key == 'elev_m':
                    match = ('elev' in n or 'dem' in n) and 'slope' not in n
                elif key == 'slope_deg':
                    match = 'slope' in n
                else:
                    match = ('dist' in n or 'prox' in n) and 'mask' not in n
                    match = match and (('riv' in n) if key == 'dist_riv_m' else ('lak' in n))
                if match and raster_ok(p):
                    choices.append(p)
        if len(choices) != 1 or not choices[0].is_file():
            inventory = '\n'.join(str(p) for p in (terrain if key in ('elev_m','slope_deg') else refs))
            raise RuntimeError(f'Cannot uniquely select {key}. Candidates: {choices}\n'
                               f'Set RASTERS[{key!r}] to the corrected existing TIFF.\nInventory:\n{inventory}')
        p = choices[0]
        if not raster_ok(p):
            raise RuntimeError(f'{key}: expected single-band projected 30 m raster: {p}')
        with rasterio.open(p) as src:
            if key.startswith('dist') and src.dtypes[0] in ('uint8', 'int8'):
                raise RuntimeError(f'{key}: byte raster looks like a mask, not a distance raster: {p}')
        result[key] = str(p.resolve())
        log(f'[RASTER] {key}: {p}')
    return result


class Sampler:
    """Nearest-pixel point sampling, block grouped with bounded LRU cache."""
    def __init__(self, path, vector_crs):
        self.ds = rasterio.open(path)
        if not CRS.from_user_input(self.ds.crs).equals(CRS.from_user_input(vector_crs), ignore_axis_order=True):
            self.ds.close()
            raise RuntimeError(f'Raster/vector CRS mismatch. Reuse matching January rasters: {path}')
        self.inv = ~self.ds.transform
        # Use bounded virtual blocks even if a TIFF has enormous strips.
        self.bh, self.bw = (min(512, x) for x in self.ds.block_shapes[0])
        self.cache = OrderedDict()
        self.used = 0
        self.budget = RASTER_CACHE_MB * 1024 * 1024

    def close(self):
        self.ds.close()
        self.cache.clear()

    def block(self, br, bc):
        k = (br, bc)
        if k in self.cache:
            self.cache.move_to_end(k)
            return self.cache[k]
        from rasterio.windows import Window
        y, x = br * self.bh, bc * self.bw
        win = Window(x, y, min(self.bw, self.ds.width-x), min(self.bh, self.ds.height-y))
        a = self.ds.read(1, window=win, masked=True, out_dtype='float32').filled(np.nan)
        a = a * self.ds.scales[0] + self.ds.offsets[0]
        a[~np.isfinite(a)] = np.nan
        while self.cache and self.used + a.nbytes > self.budget:
            _, old = self.cache.popitem(last=False)
            self.used -= old.nbytes
        if a.nbytes <= self.budget:
            self.cache[k] = a
            self.used += a.nbytes
        return a

    def sample(self, x, y):
        out = np.full(len(x), np.nan, dtype=np.float32)
        finite = np.isfinite(x) & np.isfinite(y)
        indices = np.flatnonzero(finite)
        if not len(indices):
            return out
        xx, yy = x[indices], y[indices]
        cc = np.floor(self.inv.a*xx + self.inv.b*yy + self.inv.c).astype(np.int64)
        rr = np.floor(self.inv.d*xx + self.inv.e*yy + self.inv.f).astype(np.int64)
        ok = (cc >= 0) & (cc < self.ds.width) & (rr >= 0) & (rr < self.ds.height)
        indices, rr, cc = indices[ok], rr[ok], cc[ok]
        nbc = (self.ds.width + self.bw-1)//self.bw
        keys = (rr//self.bh)*nbc + cc//self.bw
        order = np.argsort(keys, kind='stable')
        indices, rr, cc, keys = indices[order], rr[order], cc[order], keys[order]
        cuts = np.r_[0, np.flatnonzero(np.diff(keys))+1, len(keys)]
        for start, end in zip(cuts[:-1], cuts[1:]):
            if start == end:
                continue
            br, bc = divmod(int(keys[start]), nbc)
            a = self.block(br, bc)
            out[indices[start:end]] = a[rr[start:end]-br*self.bh, cc[start:end]-bc*self.bw]
        return out


def decode_gpkg(blob):
    if blob is None:
        return None
    b = bytes(blob)
    if b[:2] != b'GP' or len(b) < 8:
        raise ValueError('Invalid GeoPackage geometry header')
    if b[3] & 0x20:
        raise ValueError('Extended GeoPackage geometry is not supported')
    envelope = (b[3] >> 1) & 7
    sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    if envelope not in sizes:
        raise ValueError('Invalid GeoPackage geometry envelope code')
    return b[8 + sizes[envelope]:]


def divide(a, b):
    return np.divide(a, b, out=np.full(np.shape(a), np.nan), where=np.asarray(b)>0)


def shape_features(g, wanted):
    """Net area, perimeter, exterior measures; no smoothing or geometry repair."""
    result = {}
    a = sh.area(g)
    p = sh.length(g)
    if 'area_km2' in wanted: result['area_km2'] = a/1e6
    if 'perim_m' in wanted: result['perim_m'] = p
    if 'compact' in wanted: result['compact'] = divide(4*np.pi*a, p*p)
    if set(wanted) & {'outer_m','width_m','hole_free_c','hole_frac','n_holes'}:
        parts, idx = sh.get_parts(g, return_index=True)
        rings = sh.get_exterior_ring(parts)
        p0 = np.bincount(idx, weights=sh.length(rings), minlength=len(g))
        a0 = np.bincount(idx, weights=sh.area(sh.polygons(rings)), minlength=len(g))
        values = {'outer_m': p0, 'width_m': divide(2*a, p0),
                  'hole_free_c': divide(4*np.pi*a0, p0*p0),
                  'hole_frac': divide(a0-a, a0),
                  'n_holes': np.bincount(idx, weights=sh.get_num_interior_rings(parts), minlength=len(g))}
        result.update({k:v for k,v in values.items() if k in wanted})
    if 'solidity' in wanted: result['solidity'] = divide(a, sh.area(sh.convex_hull(g)))
    if set(wanted) & {'elong','rect_fill'}:
        rect = sh.minimum_rotated_rectangle(g)
        rings = sh.get_exterior_ring(rect)
        xy0 = sh.get_point(rings, 0)
        xy1 = sh.get_point(rings, 1)
        xy2 = sh.get_point(rings, 2)
        u, v = sh.distance(xy0, xy1), sh.distance(xy1, xy2)
        if 'elong' in wanted: result['elong'] = divide(np.maximum(u,v), np.minimum(u,v))
        if 'rect_fill' in wanted: result['rect_fill'] = divide(a, u*v)
    return result


def transform_chunk(rows, selected, mapping, samplers, year, month):
    df = pd.DataFrame.from_records(rows, columns=selected)
    fid = df.pop('__source_fid').to_numpy(dtype=np.int64)
    blobs = df.pop('__geometry').tolist()
    wkb = [decode_gpkg(b) for b in blobs]
    g = sh.from_wkb(wkb, on_invalid='raise')
    good = np.isin(sh.get_type_id(g), [3, 6]) & ~sh.is_empty(g) & ~sh.is_missing(g)
    if CHECK_GEOMETRY_VALIDITY: good &= sh.is_valid(g)
    points = sh.point_on_surface(g)
    x, y = sh.get_x(points), sh.get_y(points)
    x[~good], y[~good] = np.nan, np.nan
    out = pd.DataFrame({'source_fid': fid})
    out['obj_id'] = (df[mapping['obj_id']].astype(str) if 'obj_id' in mapping else pd.Series(fid).astype(str))
    out['year'], out['month'] = np.int16(year), np.int8(month)
    out['rep_x'], out['rep_y'] = x, y
    out['geom_ok'] = good.astype(np.uint8)  # Type/empty check; topology only when switch enabled.
    need = [k for k in MORPH if RECOMPUTE_ALL_SHAPE or k not in mapping]
    if need and not COMPUTE_MISSING_SHAPE:
        raise RuntimeError(f'Missing shape fields {need}; enable COMPUTE_MISSING_SHAPE.')
    if need:
        calculated = shape_features(g, need)
    else:
        calculated = {}
    for key in MORPH:
        v = (calculated[key] if key in need else pd.to_numeric(df[mapping[key]], errors='coerce').to_numpy(dtype=float))
        v = np.array(v, dtype=np.float64, copy=True)
        v[~np.isfinite(v)] = np.nan
        if key in need: v[~good] = np.nan
        out[key] = v
    # These are source labels only. Do not mistake them for separately edited manual labels.
    if 'label' in mapping:
        v = pd.to_numeric(df[mapping['label']], errors='coerce').fillna(0)
        out['source_label'] = v.where(v.isin([1,2]),0).astype('int8')
    else: out['source_label'] = np.int8(0)
    for key, sampler in samplers.items():
        v = sampler.sample(x, y)
        if key != 'elev_m': v[v < 0] = np.nan
        if key == 'slope_deg': v[v > 90] = np.nan
        out[key] = v
    return out


def process_month(task, rasters):
    dependencies()
    path, year, month = Path(task[0]), task[1], task[2]
    ym = f'{year}_{month:02d}'
    log(f'[OPEN {ym}] {path}')
    # All run identities include source metadata and the exact external rasters.
    identity = {'version':VERSION, 'source':statkey(path),
                'rasters':{k:statkey(v) for k,v in rasters.items()},
                'chunk_size':CHUNK_SIZE, 'recompute_shape':RECOMPUTE_ALL_SHAPE,
                'check_validity':CHECK_GEOMETRY_VALIDITY, 'layer':LAYER_OVERRIDES.get(ym)}
    digest = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    outdir = OUT_ROOT / str(year) / f'WB_{ym}' / digest
    outdir.mkdir(parents=True, exist_ok=True)
    # Prevent two simultaneous processes writing the same month; clean stale lock manually after a crash.
    lock = outdir/'RUNNING.lock'
    try:
        fd = os.open(str(lock), os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Lock exists: {lock}. If the earlier process has exited, delete only this lock and rerun.')
    os.write(fd, f'pid={os.getpid()}\n'.encode()); os.close(fd)
    try:
        return process_locked(path, year, month, ym, identity, outdir, rasters)
    finally:
        lock.unlink(missing_ok=True)


def process_locked(path, year, month, ym, identity, outdir, rasters):
    data_dir = outdir/'features'; data_dir.mkdir(exist_ok=True)
    done = outdir/'DONE.json'
    if done.exists():
        report = json.loads(done.read_text(encoding='utf-8'))
        valid = True
        for rec in report['parts']:
            p = data_dir/rec['file']
            if not p.is_file() or p.stat().st_size != rec['bytes']:
                valid = False; break
        if valid:
            log(f'[SKIP {ym}] complete: {report["rows"]:,} objects')
            return {'month':ym,'status':'skipped','run_dir':str(outdir),'rows':report['rows']}
        raise RuntimeError(f'Completed output has missing/changed parts: {outdir}; do not silently overwrite.')
    manifest = outdir/'manifest.json'
    if manifest.exists():
        info = json.loads(manifest.read_text(encoding='utf-8'))
        if info['identity'] != identity: raise RuntimeError('Input fingerprint mismatch')
    else:
        info = source_info(path, ym)
        info['identity'] = identity
        info['features'] = MORPH + EXTERNAL
        info['notes'] = 'obj_id is a string; source_fid links this exact source layer; source_label is not a manual-label merge.'
        jsave(manifest, info)
    lower = {n.lower():n for n in info['fields']}
    mapping = {k:lower[k] for k in MORPH + ['obj_id','label'] if k in lower}
    # Area_km2 matches case-insensitively. Do not reuse Shape_Area/Shape_Length under ambiguous units.
    need = [k for k in MORPH if RECOMPUTE_ALL_SHAPE or k not in mapping]
    log(f'[PLAN {ym}] {info["count"]:,} objects; reuse {len(MORPH)-len(need)} shape columns; compute {need}')
    columns = list(dict.fromkeys(mapping.values()))
    expr = [f'{quote(info["pk"])} AS __source_fid', f'{quote(info["geometry_column"])} AS __geometry'] + [quote(c) for c in columns]
    names = ['__source_fid','__geometry'] + columns
    parts=[]; last=None; processed=0; partno=0
    # Contiguous committed parts only. Parquet metadata allows resuming without rereading source geometries.
    while (data_dir/f'part-{partno:06d}.parquet').exists():
        fp = data_dir/f'part-{partno:06d}.parquet'
        meta = pq.read_metadata(fp)
        custom = meta.metadata or {}
        rec = json.loads(custom[b'qtp_chunk'].decode())
        if rec['part'] != partno or rec['previous_fid'] != last or rec['identity'] != identity:
            raise RuntimeError(f'Inconsistent checkpoint: {fp}')
        rec['file'], rec['bytes'] = fp.name, fp.stat().st_size
        processed += meta.num_rows; last=rec['last_fid'];parts.append(rec);partno+=1
    if len(list(data_dir.glob('part-*.parquet'))) != partno:
        raise RuntimeError(f'Non-contiguous partition files: {data_dir}')
    start=time.perf_counter(); new_count=0
    with ExitStack() as stack:
        con = connect(path)
        stack.callback(con.close)
        samplers={}
        for k,p in rasters.items():
            s=Sampler(p, info['crs']); stack.callback(s.close); samplers[k]=s
        while True:
            where='' if last is None else f' WHERE {quote(info["pk"])} > ?'
            sql=f'SELECT {", ".join(expr)} FROM {quote(info["source_layer"])}{where} ORDER BY {quote(info["pk"])} LIMIT ?'
            params=(CHUNK_SIZE,) if last is None else (last,CHUNK_SIZE)
            rows=con.execute(sql,params).fetchall()
            if not rows: break
            df=transform_chunk(rows,names,mapping,samplers,year,month)
            firstfid,lastfid=int(df.source_fid.iloc[0]),int(df.source_fid.iloc[-1])
            if not df.source_fid.is_monotonic_increasing or (last is not None and firstfid<=last):
                raise RuntimeError('Non-increasing source feature IDs')
            rec={'identity':identity,'part':partno,'previous_fid':last,'first_fid':firstfid,'last_fid':lastfid,
                 'rows':len(df),'missing':{k:int(df[k].isna().sum()) for k in MORPH+EXTERNAL},
                 'geometry_not_ok':int((df.geom_ok==0).sum())}
            fp=data_dir/f'part-{partno:06d}.parquet'
            tmp=fp.with_suffix('.tmp')
            tab=pa.Table.from_pandas(df,preserve_index=False)
            md=dict(tab.schema.metadata or {});md[b'qtp_chunk']=json.dumps(rec).encode()
            pq.write_table(tab.replace_schema_metadata(md),tmp,compression='zstd',compression_level=3)
            os.replace(tmp,fp)
            rec['file'],rec['bytes']=fp.name,fp.stat().st_size;parts.append(rec)
            processed+=len(df);new_count+=len(df);last=lastfid;partno+=1
            elapsed=time.perf_counter()-start;rate=new_count/max(elapsed,.001)
            eta=max(0,info['count']-processed)/max(rate,.001)/60
            log(f'[{ym}] {processed:,}/{info["count"]:,} | {rate:,.0f} objects/s | ETA {eta:.1f} min')
            del rows,df,tab
    if processed!=info['count']: raise RuntimeError(f'Row count mismatch: {processed} vs {info["count"]}')
    if statkey(path)!=identity['source'] or any(statkey(rasters[k])!=v for k,v in identity['rasters'].items()):
        raise RuntimeError('Input changed while running. Outputs remain uncommitted; rerun into new fingerprint directory.')
    report={'month':ym,'rows':processed,'identity':identity,'parts':parts,
            'missing':{k:sum(r['missing'][k] for r in parts) for k in MORPH+EXTERNAL},
            'geometry_not_ok':sum(r['geometry_not_ok'] for r in parts),
            'source_path':str(path),'source_layer':info['source_layer'],
            'new_objects_this_run':new_count,'seconds_this_run':round(time.perf_counter()-start,2)}
    jsave(done,report)
    log(f'[DONE {ym}] {outdir}; external missing: '+str({k:report['missing'][k] for k in EXTERNAL}))
    return {'month':ym,'status':'complete','run_dir':str(outdir),'rows':processed}


def discover():
    tasks=[]; missing=[]
    for y in (range(2016,2026) if ALL_YEARS else SELECT_YEARS):
        for m in MONTHS:
            stem=f'WB_{y}_{m:02d}'
            attr=ATTR_ROOT/str(y)/(stem+'_attributes.gpkg')
            raw=RAW_ROOT/str(y)/(stem+'.gpkg')
            path=attr if attr.is_file() else raw
            if path.is_file(): tasks.append((str(path),y,m))
            else: missing.append({'month':f'{y}_{m:02d}','attribute_path':str(attr),'raw_path':str(raw)})
    return tasks,missing


README = '''Monthly lake and river features

Inputs: monthly reservoir-excluded water objects and aligned elevation, slope,
river-distance and lake-distance rasters. Outputs are chunked Parquet tables,
with source identity and raster fingerprints in each monthly manifest.
Features are sampled at representative points. Geometry is not modified.
Existing shape fields are reused unless recomputation is requested.
The source object's feature ID is valid only for the exact input layer.
Run feature extraction before classification. Predictions modify source label
fields; do not rerun this older extractor blindly after label writeback.
'''


def extract_monthly_features(prepared_rasters=None):
    print('[START] QTP monthly feature extraction', flush=True)
    print('[SCRIPT]',Path(__file__).resolve(), flush=True)
    print('[PYTHON]',sys.executable, flush=True)
    dependencies()
    if CHUNK_SIZE<1 or MAX_WORKERS<1: raise ValueError('CHUNK_SIZE/MAX_WORKERS must be positive')
    OUT_ROOT.mkdir(parents=True,exist_ok=True)
    (OUT_ROOT/'数据说明.txt').write_text(README,encoding='utf-8-sig')
    rasters=prepared_rasters if prepared_rasters is not None else resolve_rasters()
    tasks,missing=discover()
    jsave(OUT_ROOT/'missing_months.json',missing)
    jsave(OUT_ROOT/'raster_sources.json',{k:statkey(p) for k,p in rasters.items()})
    log(f'[PLAN] {len(tasks)} months found; {len(missing)} missing; workers={MAX_WORKERS}')
    if not tasks: raise FileNotFoundError('No monthly GPKGs at configured ATTR_ROOT or RAW_ROOT.')
    good=[];errors=[]
    def record_error(t,e):
        errors.append({'input':t[0],'year':t[1],'month':t[2],'error':str(e)})
        log(f'[FAILED {t[1]}_{t[2]:02d}] {e}')
    if MAX_WORKERS==1:
        for t in tasks:
            try: good.append(process_month(t,rasters))
            except Exception as e:
                traceback.print_exc();record_error(t,e)
            jsave(OUT_ROOT/'completed_months.json',good)
            jsave(OUT_ROOT/'failed_months.json',errors)
    else:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            jobs={pool.submit(process_month,t,rasters):t for t in tasks}
            for future in as_completed(jobs):
                try: good.append(future.result())
                except Exception as e: record_error(jobs[future],e)
                jsave(OUT_ROOT/'completed_months.json',good)
                jsave(OUT_ROOT/'failed_months.json',errors)
    log(f'[FINISH] success/skip={len(good)}; failed={len(errors)}; missing={len(missing)}; output={OUT_ROOT}')
    if errors or missing:
        raise SystemExit('Incomplete requested coverage; see failed_months.json and missing_months.json.')



def prep_log(message):
    print(message, flush=True)

def prep_signature(path):
    path = Path(path)
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }

def prep_digest(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def prep_save_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)

def prep_progress(name):
    state = [-1]

    def callback(fraction, message, data):
        percent = int(fraction * 100)
        group = percent // 5
        if group > state[0]:
            state[0] = group
            prep_log(f"[{name}] {percent}%")
        return 1

    return callback

def prep_find_reference(explicit, filename):
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    candidates = sorted(PREP_BASE.rglob(filename))
    if len(candidates) != 1:
        listing = "\n".join(str(p) for p in candidates)
        raise RuntimeError(
            f"Cannot identify a unique {filename}; found {len(candidates)} candidates:\n"
            f"{listing}\n"
            "Specify the correct file in RIVER_MASK or LAKE_MASK."
        )
    return candidates[0]

def prep_cached_raster(path, metadata, builder):
    path = Path(path)
    marker = path.with_suffix(path.suffix + ".done.json")

    if path.exists() and marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous == metadata:
            check = gdal.Open(str(path))
            if check is None:
                raise RuntimeError(f"Cannot read cached raster: {path}")
            check = None
            prep_log(f"[CACHE] {path.name}")
            return path

    temporary = path.with_name(path.stem + ".partial.tif")
    if temporary.exists():
        temporary.unlink()

    prep_log(f"[BUILD] {path}")
    builder(temporary)

    check = gdal.Open(str(temporary))
    if check is None:
        raise RuntimeError(f"Cannot read generated raster: {temporary}")
    check = None

    temporary.replace(path)
    prep_save_json(marker, metadata)
    return path

def prep_prepare_terrain(target_crs):
    metadata = {
        "version": PREP_VERSION,
        "source": prep_signature(PREP_DEM_SOURCE),
        "target_crs": target_crs.to_wkt(),
        "resolution": PREP_RESOLUTION,
        "resampling": "bilinear",
        "vertical_unit": PREP_DEM_VERTICAL_UNIT,
    }
    key = prep_digest(metadata)
    dem_path = PREP_DEM_SOURCE.parent / f"QTP_elev_Albers_30m_{key}.tif"

    def build_dem(destination):
        result = gdal.Warp(
            str(destination),
            str(PREP_DEM_SOURCE),
            format="GTiff",
            dstSRS=target_crs.to_wkt(),
            xRes=PREP_RESOLUTION,
            yRes=PREP_RESOLUTION,
            targetAlignedPixels=True,
            resampleAlg="bilinear",
            outputType=gdal.GDT_Float32,
            dstNodata=-9999,
            multithread=True,
            warpMemoryLimit=PREP_WARP_MEMORY_MB,
            warpOptions=[f"NUM_THREADS={PREP_NUM_THREADS}"],
            creationOptions=PREP_TIFF_OPTIONS + ["PREDICTOR=3"],
            callback=prep_progress("DEM reprojection"),
        )
        if result is None:
            raise RuntimeError("DEM reprojection failed.")
        result = None

    prep_cached_raster(dem_path, metadata, build_dem)

    slope_path = PREP_DEM_SOURCE.parent / f"QTP_slope_deg_30m_{key}.tif"
    slope_metadata = {
        "version": PREP_VERSION,
        "dem": prep_signature(dem_path),
        "algorithm": "Horn",
        "scale": 1.0,
        "unit": "degree",
        "compute_edges": False,
    }

    def build_slope(destination):
        result = gdal.DEMProcessing(
            str(destination),
            str(dem_path),
            "slope",
            format="GTiff",
            alg="Horn",
            scale=1.0,
            slopeFormat="degree",
            computeEdges=False,
            creationOptions=PREP_TIFF_OPTIONS + ["PREDICTOR=3"],
            callback=prep_progress("slope"),
        )
        if result is None:
            raise RuntimeError("Slope calculation failed.")
        result = None

    prep_cached_raster(slope_path, slope_metadata, build_slope)
    return dem_path, slope_path

def prep_prepare_distance(source, kind, target_crs):
    metadata = {
        "version": PREP_VERSION,
        "reference_version": PREP_REFERENCE_VERSION,
        "source": prep_signature(source),
        "target_crs": target_crs.to_wkt(),
        "resolution": PREP_RESOLUTION,
        "resampling": "max",
        "target_value": PREP_REFERENCE_TARGET_VALUE,
        "src_nodata": PREP_REFERENCE_NODATA,
        "dst_nodata": 255,
        "mask_encoding": "0_background_1_water_255_nodata",
    }
    key = prep_digest(metadata)
    mask_path = PREP_CACHE / f"{kind}_mask_30m_{key}.tif"

    def build_mask(destination):
        src = gdal.Open(str(source))
        if src is None:
            raise RuntimeError(f"Cannot read reference: {source}")
        source_nodata = src.GetRasterBand(1).GetNoDataValue()
        src = None

        prep_log(f"[{kind}] Source NoData={source_nodata}, output NoData=255")

        result = gdal.Warp(
            str(destination),
            str(source),
            format="GTiff",
            dstSRS=target_crs.to_wkt(),
            xRes=PREP_RESOLUTION,
            yRes=PREP_RESOLUTION,
            targetAlignedPixels=True,
            resampleAlg="max",
            outputType=gdal.GDT_Byte,
            srcNodata=PREP_REFERENCE_NODATA,
            dstNodata=255,
            multithread=True,
            warpMemoryLimit=PREP_WARP_MEMORY_MB,
            warpOptions=[f"NUM_THREADS={PREP_NUM_THREADS}"],
            creationOptions=PREP_TIFF_OPTIONS,
            callback=prep_progress(kind + " reference 30 m"),
        )
        if result is None:
            raise RuntimeError(f"{kind} reference reprojection failed.")
        result = None

        counts = {0: 0, 1: 0, 255: 0}
        with rasterio.open(destination) as raster:
            for _, window in raster.block_windows(1):
                array = raster.read(1, window=window)
                values, numbers = np.unique(array, return_counts=True)

                for value, number in zip(values, numbers):
                    value = int(value)
                    if value not in counts:
                        raise RuntimeError(
                            f"{kind} mask has invalid value {value}, "
                            "expected 0, 1 or 255."
                        )
                    counts[value] += int(number)

        if counts[1] == 0:
            raise RuntimeError(f"{kind} reference has no target water pixels.")

        prep_log(
            f"[{kind} value counts] background={counts[0]:,}；"
            f"target={counts[1]:,}; NoData={counts[255]:,}"
        )
        prep_save_json(
            PREP_CACHE / f"{kind}_mask_counts_{key}.json",
            counts,
        )

    prep_cached_raster(mask_path, metadata, build_mask)

    distance_path = PREP_CACHE / f"dist_{kind}_m_30m_{key}.tif"
    distance_metadata = {
        "version": PREP_VERSION,
        "reference_version": PREP_REFERENCE_VERSION,
        "mask": prep_signature(mask_path),
        "units": "metre",
        "target": 1,
        "method": "GDAL_ComputeProximity",
        "use_input_nodata": False,
    }

    def build_distance(destination):
        source_ds = gdal.Open(str(mask_path))
        output_ds = None

        try:
            output_ds = gdal.GetDriverByName("GTiff").Create(
                str(destination),
                source_ds.RasterXSize,
                source_ds.RasterYSize,
                1,
                gdal.GDT_Float32,
                options=PREP_TIFF_OPTIONS + ["PREDICTOR=3"],
            )
            if output_ds is None:
                raise RuntimeError("Cannot create distance raster.")

            output_ds.SetGeoTransform(source_ds.GetGeoTransform())
            output_ds.SetProjection(source_ds.GetProjection())
            output_ds.GetRasterBand(1).SetNoDataValue(-9999)

            status = gdal.ComputeProximity(
                source_ds.GetRasterBand(1),
                output_ds.GetRasterBand(1),
                options=[
                    "VALUES=1",
                    "DISTUNITS=GEO",
                    "USE_INPUT_NODATA=NO",
                    "NODATA=-9999",
                ],
                callback=prep_progress(kind + " distance"),
            )
            if status != 0:
                raise RuntimeError(f"{kind} distance calculation failed: {status}")

            output_ds.FlushCache()

        finally:
            output_ds = None
            source_ds = None

    prep_cached_raster(distance_path, distance_metadata, build_distance)
    return distance_path


def preparation_crs():
    """Use a reference label layer or a configured monthly water-object layer."""
    if PREP_CRS_GPKG is not None:
        source, ym = Path(PREP_CRS_GPKG), PREP_CRS_MONTH
        info = source_info(source, ym)
        crs = CRS.from_user_input(info['crs'])
        log(f'[CRS] Reference: {source}')
    elif PREP_CRS_REFERENCE is not None and Path(PREP_CRS_REFERENCE).is_file():
        import pyogrio
        info = pyogrio.read_info(PREP_CRS_REFERENCE)
        if not info.get('crs'):
            raise RuntimeError(f'Missing CRS: {PREP_CRS_REFERENCE}')
        crs = CRS.from_user_input(info['crs'])
        log(f'[CRS] Reference: {PREP_CRS_REFERENCE}')
    else:
        tasks, _ = discover()
        if not tasks:
            raise FileNotFoundError('No monthly GPKG found for the reference CRS. Set PREP_CRS_GPKG.')
        source, y, m = tasks[0]
        source, ym = Path(source), f'{y}_{m:02d}'
        info = source_info(source, ym)
        crs = CRS.from_user_input(info['crs'])
        log(f'[CRS] Reference: {source}')
    if not crs.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-9 for a in crs.axis_info[:2]):
        raise RuntimeError('Reference CRS must be projected in metres.')
    return crs


def prepare_shared_rasters():
    global gdal, np, rasterio, CRS
    from osgeo import gdal
    import numpy as np
    import rasterio
    from pyproj import CRS
    gdal.UseExceptions()
    gdal.SetCacheMax(PREP_GDAL_CACHE_MB * 1024 * 1024)
    if not PREP_DEM_SOURCE.is_file():
        raise FileNotFoundError(PREP_DEM_SOURCE)
    PREP_CACHE.mkdir(parents=True, exist_ok=True)
    gdal.SetConfigOption('CPL_TMPDIR', str(PREP_CACHE))
    crs = preparation_crs()
    river_source = prep_find_reference(PREP_RIVER_MASK, PREP_RIVER_FILENAME)
    lake_source = prep_find_reference(PREP_LAKE_MASK, PREP_LAKE_FILENAME)
    dem, slope = prep_prepare_terrain(crs)
    river_distance = prep_prepare_distance(river_source, 'river', crs)
    lake_distance = prep_prepare_distance(lake_source, 'lake', crs)
    rasters = {'elev_m': dem, 'slope_deg': slope, 'dist_riv_m': river_distance, 'dist_lak_m': lake_distance}
    return {key: str(Path(value).resolve()) for key, value in rasters.items()}


def main():
    if not PREPARE_RASTERS and not RUN_EXTRACTION:
        raise ValueError('Enable PREPARE_RASTERS, RUN_EXTRACTION, or both.')
    prepared = prepare_shared_rasters() if PREPARE_RASTERS else None
    if RUN_EXTRACTION:
        extract_monthly_features(prepared)

if __name__=='__main__':
    from multiprocessing import freeze_support
    freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
