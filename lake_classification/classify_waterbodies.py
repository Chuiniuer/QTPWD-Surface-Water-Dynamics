# -*- coding: utf-8 -*-
"""2018 pooled-month RF / LightGBM comparison. Run directly; edit SETTINGS below.
Requires numpy pandas pyarrow shapely>=2 rasterio pyproj pyogrio scikit-learn joblib matplotlib;
lightgbm>=4 only when RUN_LIGHTGBM_TEST=True. No source geometry is copied.
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

# ====================== FEATURE CACHE SETTINGS ======================
BASE = Path(r'K:\Monthly surface water dynamics TIB\supervised classification\step5_lake_classification')
ATTR_ROOT = BASE / 'monthly_lake_river_attributes'
RAW_ROOT = Path(r'E:\phd_l3\QTP_reservoir_separation\water_without_reservoirs')
OUT_ROOT = Path(r'E:\phd_l3\QTP_monthly_lake_river_features')
TERRAIN_ROOT = Path(r'E:\phd_l3\terrain_data')
REFERENCE_ROOT = BASE / 'external_features_2018_01'  # Recursively find the existing distance caches

CHUNK_SIZE = 50_000
COMPUTE_MISSING_SHAPE = True # Reuse existing shape columns; calculate missing COLUMNS only.
RECOMPUTE_ALL_SHAPE = False  # Do not enable just to append four external features.
CHECK_GEOMETRY_VALIDITY = False  # Extra topology check; no geometry repair/modification.
RASTER_CACHE_MB = 64         # Per raster, per worker; 4 rasters ~256 MB maximum LRU storage.

# Fill explicit paths if discovery finds more than one valid candidate.
# Reuse the SAME corrected 30 m rasters used in January's successful feature extraction.
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

# ================== UNIFIED MODEL SETTINGS ==================
EDIT_ROOT = BASE / 'monthly_lake_river_edit_shp'
MODEL_ROOT = Path(r'E:\phd_l3\QTP_lake_river_RF_comparison')
TRAIN_YEARS = [2018]
TRAIN_MONTHS = list(range(1, 13))
PREDICT_YEARS = list(range(2016, 2026))
PREDICT_MONTHS = list(range(1, 13))
# Expected edits: EDIT_ROOT/2018/01/labeled.shp and unlabeled.shp (also supports YYYY_MM folder).
# Override nonstandard locations with {'2018_01': {'rule': [Path(...)], 'manual': [Path(...)]}}.
EDIT_OVERRIDES = {}
# If several completed feature versions exist for a month, explicitly choose one:
FEATURE_RUN_OVERRIDES = {}  # {'2018_01': Path(r'E:\...\WB_2018_01\fingerprint')}
ID_FIELD = 'obj_id'
LABEL_FIELD = 'label'
ALL_LABELED_REVIEWED = False  # True ONLY if every retained label in labeled.shp was manually checked.
SPATIAL_BLOCK_M = 25_000.0    # Same block across all months => same train/validation/test role.
RANDOM_SEED = 42
AUTO_CAP_PER_CLASS_MONTH = 50_000  # None uses all available rule samples; more RAM/time.
AUTO_MAX_MANUAL_RATIO = 5
AUTO_WEIGHT = 0.25
MIN_TEST_ACCURACY = 0.80

VERSION = 'qtp_features_20260929_v1'
MORPH = ['area_km2', 'perim_m', 'outer_m', 'elong', 'compact', 'rect_fill',
         'solidity', 'width_m', 'hole_free_c', 'hole_frac', 'n_holes']
EXTERNAL = ['elev_m', 'slope_deg', 'dist_riv_m', 'dist_lak_m']
np = pd = pa = pq = sh = rasterio = CRS = None


# ======================== USER SETTINGS ========================
RUN_RF_TEST = False
RUN_LIGHTGBM_TEST = False
RUN_PREDICTION = True

LOAD_RUN = Path(
    r"E:\phd_l3\QTP_lake_river_RF_comparison"
    r"\run_8a6d695e43bd9cd2acc8"
)
MAX_TRAIN_SAMPLES = 1_000_000  # Pooled 12 months, including selected automatic samples.
MAX_VALID_SAMPLES = 200_000
MAX_TEST_SAMPLES = 200_000
N_JOBS = min(4, max(1, (os.cpu_count() or 2)-1))
RF_TREES = 240
RF_PILOT_TREES = 24
RF_MAX_SAMPLES = 100_000      # Bootstrap draws per tree; bounds training cost.
RF_PARAMS = dict(max_depth=22, min_samples_leaf=10, max_features='sqrt',
                 max_leaf_nodes=8192, bootstrap=True, criterion='gini')
LGB_TREES = 1800
STRATEGIES = ['unweighted', 'class_weight', 'undersample']
IMPORTANCE_SAMPLE = 20_000
IMPORTANCE_REPEATS = 3
# Common metric for all algorithms: decrease in balanced accuracy, percentage points.
IMPORTANCE_GROUPS = {
    'Size': ['area_km2','perim_m','outer_m','width_m'],
    'Shape': ['elong','compact','rect_fill','solidity','hole_free_c','hole_frac','n_holes'],
    'Terrain': ['elev_m','slope_deg'],
    'Reference proximity': ['dist_riv_m','dist_lak_m']}
PREDICTION_MIN_BALANCED_ACCURACY = .80
PIPELINE_VERSION_RF = 'qtp_rf_compare_20261001_v1'
# Only one output attribute is added/updated in the source vector.
OUTPUT_LABEL_FIELD = 'label'
# ===============================================================



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


# ================== LABEL MERGE / TRAIN / PREDICT ==================
import re
import gc
import platform


def dump_parquet(df, path, metadata=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    tab=pa.Table.from_pandas(df,preserve_index=False)
    if metadata is not None:
        md=dict(tab.schema.metadata or {})
        md[b'qtp_pipeline']=json.dumps(metadata,ensure_ascii=False).encode()
        tab=tab.replace_schema_metadata(md)
    tmp=path.with_suffix('.tmp')
    pq.write_table(tab,tmp,compression='zstd',compression_level=3)
    os.replace(tmp,path)


def jload(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest_obj(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]


def canonical_ids(s):
    if s.isna().any(): raise ValueError('Null obj_id in input; cannot safely join.')
    if pd.api.types.is_float_dtype(s):
        v=s.to_numpy()
        if (abs(v)>=2**53).any() or (v!=np.floor(v)).any():
            raise ValueError('Non-integer or precision-unsafe floating object IDs.')
        return s.astype('int64').astype(str)
    return s.astype(str).str.strip().str.replace(r'^([+-]?\d+)\.0+$',r'\1',regex=True)


def shape_signature(path):
    files=[]
    for ext in ['.shp','.dbf','.shx','.prj','.cpg']:
        p=path.with_suffix(ext)
        if p.exists(): files.append(statkey(p))
    if not path.with_suffix('.dbf').exists(): raise FileNotFoundError(path.with_suffix('.dbf'))
    return files


def find_edits():
    result={}
    for year in TRAIN_YEARS:
        for month in TRAIN_MONTHS:
            ym=f'{year}_{month:02d}'
            if ym in EDIT_OVERRIDES:
                pair={k:[Path(p) for p in EDIT_OVERRIDES[ym].get(k,[])] for k in ('rule','manual')}
            else:
                candidates=[EDIT_ROOT/str(year)/f'{month:02d}', EDIT_ROOT/str(year)/str(month),
                            EDIT_ROOT/ym, EDIT_ROOT/str(year)/f'WB_{ym}']
                folders=[]
                for p in candidates:
                    if p.is_dir() and p not in folders: folders.append(p)
                if len(folders)!=1:
                    raise RuntimeError(f'{ym}: need one edit folder, found {folders}. Set EDIT_OVERRIDES.')
                files=sorted(folders[0].glob('*.shp'))
                pair={'rule':[p for p in files if re.fullmatch(r'labeled(?:_\d+)?',p.stem,re.I)],
                      'manual':[p for p in files if re.fullmatch(r'unlabeled(?:_\d+)?',p.stem,re.I)]}
            if not pair['rule'] or not pair['manual']:
                raise FileNotFoundError(f'{ym}: need labeled.shp AND unlabeled.shp; discovered {pair}')
            result[ym]=pair
    return result


def completed_feature_run(ym):
    y,m=map(int,ym.split('_'))
    parent=OUT_ROOT/str(y)/f'WB_{ym}'
    options=[Path(FEATURE_RUN_OVERRIDES[ym])] if ym in FEATURE_RUN_OVERRIDES else sorted(parent.glob('*'))
    candidates=[]
    for p in options:
        if not (p/'DONE.json').is_file() or not (p/'manifest.json').is_file(): continue
        info=jload(p/'manifest.json'); report=jload(p/'DONE.json')
        source=Path(info['source_path'])
        if not source.is_file(): continue
        if statkey(source)!=info['identity']['source']: continue
        # Reject active edits which may not yet have changed the main SQLite file.
        wal=Path(str(source)+'-wal')
        if wal.exists() and wal.stat().st_size: continue
        if any(not (p/'features'/r['file']).is_file() or (p/'features'/r['file']).stat().st_size!=r['bytes']
               for r in report['parts']): continue
        candidates.append(p)
    if len(candidates)>1:
        raise RuntimeError(f'{ym}: multiple completed feature versions: {candidates}. Set FEATURE_RUN_OVERRIDES.')
    return candidates[0] if candidates else None


def source_for_month(ym):
    y,m=map(int,ym.split('_'))
    a=ATTR_ROOT/str(y)/f'WB_{ym}_attributes.gpkg'
    b=RAW_ROOT/str(y)/f'WB_{ym}.gpkg'
    p=a if a.is_file() else b
    if not p.is_file(): raise FileNotFoundError(f'Missing both {a} and {b}')
    return (str(p),y,m)


def reference_paths_from_runs(runs):
    # Reuse the exact raster paths from successful completed feature extraction.
    if all(RASTERS[k] for k in EXTERNAL): return {k:str(Path(RASTERS[k])) for k in EXTERNAL}
    refs=[]
    for run in runs.values():
        if run is not None:
            refs.append(jload(run/'manifest.json')['identity']['rasters'])
    if refs:
        if any(r!=refs[0] for r in refs[1:]):
            raise RuntimeError('Training months used different raster cache signatures. Do not mix them silently.')
        out={k:str(RASTERS[k] or refs[0][k]['path']) for k in EXTERNAL}
        for k,p in out.items():
            if not RASTERS[k] and Path(p).exists() and statkey(p)!=refs[0][k]:
                raise RuntimeError(f'Cached raster changed since feature extraction: {p}')
        return out
    return resolve_rasters()


def feature_signature(run):
    return {'run':str(run.resolve()),'manifest':statkey(run/'manifest.json'),
            'done':statkey(run/'DONE.json')}


def read_edit_labels(pair):
    import pyogrio
    frames=[]
    for kind,paths in pair.items():
        for path in paths:
            fields=list(pyogrio.read_info(path)['fields'])
            lookup={f.lower():f for f in fields}
            if ID_FIELD.lower() not in lookup or LABEL_FIELD.lower() not in lookup:
                raise ValueError(f'{path}: required {ID_FIELD}/{LABEL_FIELD}, available {fields}')
            idf,labf=lookup[ID_FIELD.lower()],lookup[LABEL_FIELD.lower()]
            # OGR filters DBF attributes and does not load the many unlabelled geometries.
            df=pyogrio.read_dataframe(path,columns=[idf,labf],read_geometry=False,
                where=f'{quote(labf)} IN (1,2)',use_arrow=True)
            df=df.rename(columns={idf:'obj_id',labf:'label'})
            df['obj_id']=canonical_ids(df.obj_id)
            df['label']=pd.to_numeric(df.label,errors='raise').astype('int8')
            df['edit_file_kind']=np.int8(kind=='manual')
            frames.append(df)
            log(f'[LABEL] {path.name}: {len(df):,} nonzero labels')
    result=pd.concat(frames,ignore_index=True)
    if result.empty: raise ValueError('No labels 1/2 found.')
    dup=result[result.obj_id.duplicated(keep=False)]
    if not dup.empty:
        conflicts=dup.groupby('obj_id').label.nunique()
        bad=conflicts[conflicts>1]
        if len(bad): raise ValueError(f'Conflicting edited labels for the same obj_id, examples: {list(bad.index[:10])}')
    # Same-value duplicate: explicitly edited unlabeled.shp wins provenance.
    result=result.sort_values('edit_file_kind').drop_duplicates('obj_id',keep='last')
    return result.set_index('obj_id',verify_integrity=True)


def spatial_roles(df):
    good=np.isfinite(df.rep_x.to_numpy()) & np.isfinite(df.rep_y.to_numpy())
    out=np.full(len(df),-1,dtype='int8')
    bx=np.zeros(len(df),dtype='int64');by=bx.copy()
    bx[good]=np.floor(df.rep_x.to_numpy()[good]/SPATIAL_BLOCK_M).astype('int64')
    by[good]=np.floor(df.rep_y.to_numpy()[good]/SPATIAL_BLOCK_M).astype('int64')
    # Seed is included in hashing; all months of the same projected spatial block agree.
    h=pd.util.hash_pandas_object(pd.DataFrame({'bx':bx[good],'by':by[good],'seed':RANDOM_SEED}),index=False).to_numpy(dtype='uint64')
    u=h % np.uint64(10000)
    out[good]=np.where(u<6000,0,np.where(u<8000,1,2)).astype('int8')
    return out,bx,by


def ensure_external(df, feature_info, rasters, stack, samplers):
    missing=[k for k in EXTERNAL if k not in df.columns]
    if not missing: return df
    if 'rep_x' not in df or 'rep_y' not in df:
        raise RuntimeError('Feature table has no representative coordinates. Rebuild this month with the supplied feature engine.')
    for k in missing:
        if k not in samplers:
            if not Path(rasters[k]).is_file(): raise FileNotFoundError(rasters[k])
            samplers[k]=Sampler(rasters[k],feature_info['crs']);stack.callback(samplers[k].close)
        df[k]=samplers[k].sample(df.rep_x.to_numpy(),df.rep_y.to_numpy())
        if k!='elev_m': df.loc[df[k]<0,k]=np.nan
        if k=='slope_deg': df.loc[df[k]>90,k]=np.nan
    return df


def merge_month(run_root, ym, edits, feature_run, rasters):
    dest=run_root/'merged_labels'/ym;dest.mkdir(parents=True,exist_ok=True)
    finish=dest/'DONE.json'
    if finish.exists():
        meta=jload(finish)
        if all((dest/r['file']).is_file() and (dest/r['file']).stat().st_size==r['bytes'] for r in meta['parts']):
            log(f'[LABEL CACHE {ym}] {meta["matched_labels"]:,} objects');return meta
        raise RuntimeError(f'Incomplete labelled cache: {dest}')
    labels=read_edit_labels(edits)
    info=jload(feature_run/'manifest.json');finfo=jload(feature_run/'DONE.json')
    if info.get('obj_id_rule')!='existing obj_id':
        raise RuntimeError(f'{ym}: training join requires existing obj_id from the attributes GPKG, not newly assigned source FID.')
    found=0;parts=[];counts=[];missing_counts={k:0 for k in MORPH+EXTERNAL}
    with ExitStack() as stack:
        samplers={}
        for j,rec in enumerate(finfo['parts']):
            src=feature_run/'features'/rec['file']
            fp=dest/f'part-{j:06d}.parquet'
            # Part-level resume; these files are keyed by the immutable pipeline run fingerprint.
            if fp.exists():
                md=pq.read_metadata(fp).metadata
                cached=json.loads(md[b'qtp_pipeline'])
                found+=cached['rows'];counts.extend(cached['counts'])
                for k,v in cached['missing'].items():missing_counts[k]+=v
                parts.append({'file':fp.name,'bytes':fp.stat().st_size});continue
            fields=pq.read_schema(src).names
            required=['obj_id','source_fid','source_label','rep_x','rep_y']+MORPH
            if any(k not in fields for k in required): raise ValueError(f'{src}: missing {[k for k in required if k not in fields]}')
            df=pd.read_parquet(src)
            ids=canonical_ids(df.obj_id)
            loc=labels.index.get_indexer(ids)
            keep=loc>=0
            df=df.loc[keep].copy();df['obj_id']=ids[keep].to_numpy()
            ed=labels.iloc[loc[keep]]
            df['label']=ed.label.to_numpy(dtype='int8')
            baseline=pd.to_numeric(df.source_label,errors='coerce').fillna(0).to_numpy(dtype='int8')
            ismanual=(ed.edit_file_kind.to_numpy()==1) | (np.isin(baseline,[1,2]) & (df.label.to_numpy()!=baseline))
            if ALL_LABELED_REVIEWED: ismanual[:]=True
            df['is_manual']=ismanual.astype('int8')
            df=ensure_external(df,info,rasters,stack,samplers)
            for k in MORPH+EXTERNAL:
                df[k]=pd.to_numeric(df[k],errors='coerce').replace([np.inf,-np.inf],np.nan).astype('float32')
            roles,bx,by=spatial_roles(df);df['split']=roles;df['block_x']=bx;df['block_y']=by
            y,m=map(int,ym.split('_'));df['year']=np.int16(y);df['month']=np.int8(m)
            cols=['obj_id','source_fid','year','month','label','is_manual','split','block_x','block_y','rep_x','rep_y']+MORPH+EXTERNAL
            df=df[cols]
            group=df.groupby(['label','is_manual','split'],observed=True).size().reset_index(name='n')
            records=group.to_dict('records');counts.extend(records)
            miss={k:int(df[k].isna().sum()) for k in MORPH+EXTERNAL}
            for k,v in miss.items(): missing_counts[k]+=v
            found+=len(df)
            dump_parquet(df,fp,{'rows':len(df),'counts':records,'missing':miss})
            parts.append({'file':fp.name,'bytes':fp.stat().st_size})
            log(f'[MERGE {ym}] {j+1}/{len(finfo["parts"])} parts; matched {found:,}/{len(labels):,}')
    if found!=len(labels):
        raise RuntimeError(f'{ym}: label join matched {found:,}/{len(labels):,}; do not train with silently lost labels. Check obj_id and source version.')
    summary=pd.DataFrame(counts).groupby(['label','is_manual','split'],as_index=False)['n'].sum().to_dict('records')
    meta={'month':ym,'matched_labels':found,'parts':parts,'counts':summary,'missing':missing_counts,
          'feature_run':str(feature_run),'label_id_join':'obj_id; scoped to year/month'}
    jsave(finish,meta)
    return meta


def cap_candidates(pool, frame, limit):
    if len(frame)==0 or limit==0: return pool
    out=frame if pool is None else pd.concat([pool,frame],ignore_index=True)
    if limit is not None and len(out)>limit: out=out.nsmallest(limit,'sample_hash')
    return out


def count_summary(meta, cls, split=None):
    return sum(r['n'] for r in meta['counts'] if r['label']==cls and r['is_manual']==1 and (split is None or r['split']==split) and r['split']>=0)


def cap_for(n):
    limit=AUTO_MAX_MANUAL_RATIO*n
    return limit if AUTO_CAP_PER_CLASS_MONTH is None else min(limit,AUTO_CAP_PER_CLASS_MONTH)


def metrics_for(y,p):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix
    pred=np.where(p>=.5,2,1)
    pr,re,fs,support=precision_recall_fscore_support(y,pred,labels=[1,2],zero_division=0)
    both=len(np.unique(y))==2
    return {'n':int(len(y)),'accuracy':float(accuracy_score(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)) if both else None,
            'macro_f1':float(f1_score(y,pred,labels=[1,2],average='macro',zero_division=0)),
            'river_roc_auc':float(roc_auc_score(np.array(y)==2,p)) if both else None,
            'river_average_precision':float(average_precision_score(np.array(y)==2,p)) if both else None,
            'lake_precision':float(pr[0]),'lake_recall':float(re[0]),
            'river_precision':float(pr[1]),'river_recall':float(re[1]),
            'lake_n':int(support[0]),'river_n':int(support[1]),
            'confusion_matrix':confusion_matrix(y,pred,labels=[1,2]).tolist()}


def savefig(fig,path):
    for ext in ('png','svg'): fig.savefig(path.with_suffix('.'+ext),dpi=300,bbox_inches='tight',pad_inches=.06)
    import matplotlib.pyplot as plt
    plt.close(fig)


def manual_lookup(root,ym,summaries):
    if ym not in summaries:return pd.Series(dtype='int8')
    arrays=[]
    for r in summaries[ym]['parts']:
        d=pd.read_parquet(root/'merged_labels'/ym/r['file'],columns=['obj_id','label','is_manual'])
        arrays.append(d.loc[d.is_manual==1,['obj_id','label']])
    d=pd.concat(arrays,ignore_index=True)
    if d.obj_id.duplicated().any():raise RuntimeError(f'{ym}: duplicate manual keys')
    return d.set_index('obj_id').label



def quotas(counts, limit):
    """Proportional allocation; preserve class/month/source proportions up to rounding."""
    keys=list(counts); a=np.array([counts[k] for k in keys],dtype=np.int64)
    if not len(a) or a.sum()==0: return {k:0 for k in keys}
    limit=min(int(limit),int(a.sum()))
    raw=a.astype(float)*limit/a.sum(); n=np.floor(raw).astype(int)
    for j in np.argsort(-(raw-n),kind='stable')[:limit-int(n.sum())]: n[j]+=1
    return dict(zip(keys,map(int,n)))


def sample_pools(root,summaries):
    counts={}
    rows=[]
    for ym,meta in summaries.items():
        for r in meta['counts']:
            rows.append(dict(month=ym,**r))
            role=int(r['split']); manual=int(r['is_manual']); cls=int(r['label'])
            if role<0 or (not manual and role!=0): continue
            n=int(r['n'])
            if not manual:
                n=min(n,cap_for(count_summary(meta,cls,0)))
            counts[(ym,role,manual,cls)]=n
    pd.DataFrame(rows).to_csv(root/'all_label_counts.csv',index=False,encoding='utf-8-sig')
    limits={}
    for role,cap in enumerate([MAX_TRAIN_SAMPLES,MAX_VALID_SAMPLES,MAX_TEST_SAMPLES]):
        limits.update(quotas({k:n for k,n in counts.items() if k[1]==role},cap))
    pools={k:None for k,n in limits.items() if n>0}
    for ym,meta in summaries.items():
        log(f'[SAMPLE {ym}] streaming cached labels; no geometry read')
        for rec in meta['parts']:
            d=pd.read_parquet(root/'merged_labels'/ym/rec['file'])
            d=d[d.split>=0].copy()
            d['sample_hash']=pd.util.hash_pandas_object(d[['obj_id','year','month']],index=False).to_numpy(dtype='uint64') ^ np.uint64(RANDOM_SEED)
            for (role,manual,cls),g in d.groupby(['split','is_manual','label'],sort=False):
                k=(ym,int(role),int(manual),int(cls))
                if k in pools:pools[k]=cap_candidates(pools[k],g,limits[k])
    out=[]
    for role in range(3):
        sub=[v for k,v in pools.items() if k[1]==role and v is not None]
        if not sub:raise RuntimeError(f'No samples for split {role}')
        d=pd.concat(sub,ignore_index=True)
        if set(d.label.unique())!={1,2}:raise RuntimeError(f'Split {role} lacks one class')
        out.append(d)
    pd.DataFrame([dict(month=k[0],split=k[1],is_manual=k[2],label=k[3],available=counts[k],selected=limits[k]) for k in limits]).to_csv(root/'sampling_counts.csv',index=False,encoding='utf-8-sig')
    return out


def medians_and_features(train):
    keep=[];drop={};med={}
    for k in MORPH+EXTERNAL:
        x=pd.to_numeric(train[k],errors='coerce').replace([np.inf,-np.inf],np.nan)
        if x.notna().sum()==0:drop[k]='all missing in training'
        elif x.nunique(dropna=True)<2:drop[k]='constant in training'
        else:keep.append(k);med[k]=float(x.median())
    if not keep:raise RuntimeError('No usable features')
    return keep,med,drop


def matrix(d,features,medians):
    x=d[features].to_numpy(dtype='float32',copy=True)
    for j,k in enumerate(features):x[~np.isfinite(x[:,j]),j]=medians[k]
    return x


def training_variant(d,strategy):
    if strategy=='undersample':
        n=int(d.label.value_counts().min())
        idx=[]
        for c in (1,2):idx.extend(d.loc[d.label==c].nsmallest(n,'sample_hash').index.tolist())
        selected=d.loc[idx].copy().reset_index(drop=True)
    else:selected=d.copy().reset_index(drop=True)
    weights=np.where(selected.is_manual.to_numpy()==1,1.,AUTO_WEIGHT)
    # Match the total effective class weight (including reduced rule-sample weight).
    # Applied once only; model class_weight=None avoids double weighting.
    if strategy=='class_weight':
        total=weights.sum()
        for c in (1,2):
            mask=selected.label.to_numpy()==c
            weights[mask]*=total/(2*weights[mask].sum())
    return selected,weights


def p_river(model,x):
    j=list(model.classes_).index(2)
    return model.predict_proba(x)[:,j]


def train_one(algorithm,strategy,d,valid,features,medians):
    import joblib
    sub,w=training_variant(d,strategy)
    x=matrix(sub,features,medians); y=sub.label.to_numpy()
    start=time.monotonic()
    if algorithm=='RF':
        from sklearn.ensemble import RandomForestClassifier
        n=min(RF_PILOT_TREES,RF_TREES)
        model=RandomForestClassifier(n_estimators=n,**RF_PARAMS,
            max_samples=min(RF_MAX_SAMPLES,len(sub)),n_jobs=N_JOBS,
            class_weight=None,random_state=RANDOM_SEED,warm_start=True)
        log(f'[FIT RF/{strategy}] {len(y):,} samples, {n} pilot trees, {N_JOBS} threads')
        model.fit(x,y,sample_weight=w)
        elapsed=time.monotonic()-start
        log(f'[RF PILOT] {elapsed/60:.1f} min; rough full-fit estimate {elapsed*RF_TREES/n/60:.1f} min (not a guarantee)')
        if RF_TREES>n:
            model.set_params(n_estimators=RF_TREES);model.fit(x,y,sample_weight=w)
        model.set_params(warm_start=False)
    else:
        import lightgbm as lgb
        model=lgb.LGBMClassifier(n_estimators=LGB_TREES,learning_rate=.03,num_leaves=31,
            min_child_samples=40,colsample_bytree=.9,subsample=.85,subsample_freq=1,
            reg_alpha=.2,reg_lambda=2,n_jobs=N_JOBS,random_state=RANDOM_SEED,
            class_weight=None,verbosity=-1,deterministic=True,force_col_wise=True)
        model.fit(x,y,sample_weight=w,eval_set=[(matrix(valid,features,medians),valid.label)],
            eval_metric='binary_logloss',callbacks=[lgb.early_stopping(100,verbose=False)])
    info={'algorithm':algorithm,'strategy':strategy,'train_n':len(sub),
          'train_lake':int((y==1).sum()),'train_river':int((y==2).sum()),
          'lake_weight_sum':float(w[y==1].sum()),'river_weight_sum':float(w[y==2].sum()),
          'seconds':time.monotonic()-start}
    return model,info


def figure_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':11,'axes.labelsize':12,'axes.titlesize':12,'xtick.labelsize':10,
        'ytick.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'svg.fonttype':'none','savefig.facecolor':'white'})
    return plt


def figure_comparison(root,records):
    plt=figure_style();d=pd.DataFrame(records)
    fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained',sharey=True,
                         gridspec_kw={'width_ratios':[1.6,1]})
    y=np.arange(len(d));titles={'unweighted':'Unweighted','class_weight':'Class weighting','undersample':'Undersampling'}
    labels=[f'{a}: {titles[t]}' for a,t in zip(d.algorithm,d.strategy)]
    colors=['#559CAF','#89BEA2','#D397A3']
    for j,(key,name) in enumerate([('balanced_accuracy','Balanced accuracy'),('macro_f1','Macro F1'),('accuracy','Accuracy')]):
        axs[0].scatter(100*d[key],y+(j-1)*.19,s=38,color=colors[j],label=name,zorder=3)
    axs[0].set(yticks=y,yticklabels=labels,xlabel='Validation score (%)',xlim=(0,103))
    axs[0].invert_yaxis();axs[0].legend(frameon=False,fontsize=9,loc='lower left')
    axs[0].grid(axis='x',alpha=.15);axs[0].set_axisbelow(True)
    axs[1].barh(y,d.seconds/60,height=.5,color=['#559CAF' if a=='RF' else '#89BEA2' for a in d.algorithm])
    axs[1].set_xlabel('Training time (min)');axs[1].tick_params(axis='y',left=False)
    for ax,letter in zip(axs,'ab'):ax.text(-.05,1.03,letter,transform=ax.transAxes,weight='bold',fontsize=14)
    savefig(fig,root/'01_validation_comparison')


def figure_test(root,y,p):
    from sklearn.metrics import confusion_matrix,roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
    plt=figure_style();pred=np.where(p>=.5,2,1);cm=confusion_matrix(y,pred,labels=[1,2])
    frac=cm/cm.sum(axis=1,keepdims=True)
    fig,axs=plt.subplots(1,3,figsize=(12.5,3.8),layout='constrained')
    axs[0].imshow(frac,cmap='Blues',vmin=0,vmax=1)
    for i in range(2):
        for j in range(2):axs[0].text(j,i,f'{cm[i,j]:,}\n{frac[i,j]:.1%}',ha='center',va='center',color='white' if frac[i,j]>.55 else '#283C45',fontsize=12)
    axs[0].set(xticks=[0,1],xticklabels=['Lake','River'],yticks=[0,1],yticklabels=['Lake','River'],xlabel='Predicted class',ylabel='Manual class')
    fpr,tpr,_=roc_curve(np.asarray(y)==2,p);axs[1].plot(fpr,tpr,color='#559CAF',lw=2,label=f'AUC = {roc_auc_score(np.asarray(y)==2,p):.4f}')
    axs[1].plot([0,1],[0,1],'--',color='.7',lw=.8);axs[1].set(xlabel='False positive rate',ylabel='True positive rate',xlim=(0,1),ylim=(0,1.02));axs[1].legend(frameon=False,loc='lower right')
    precision,recall,_=precision_recall_curve(np.asarray(y)==2,p)
    axs[2].plot(recall,precision,color='#89BEA2',lw=2,label=f'AP = {average_precision_score(np.asarray(y)==2,p):.4f}')
    axs[2].axhline(np.mean(np.asarray(y)==2),color='.7',ls='--',lw=.8)
    axs[2].set(xlabel='River recall',ylabel='River precision',xlim=(0,1),ylim=(0,1.02));axs[2].legend(frameon=False,loc='lower left')
    for ax,letter in zip(axs,'abc'):ax.text(-.14,1.04,letter,transform=ax.transAxes,weight='bold',fontsize=14)
    savefig(fig,root/'02_selected_model_test')


def permutation_scores(model,d,features,medians,groups):
    from sklearn.metrics import balanced_accuracy_score
    x=matrix(d,features,medians);y=d.label.to_numpy()
    base=balanced_accuracy_score(y,model.predict(x));rng=np.random.default_rng(RANDOM_SEED)
    output=[]
    for name,cols in groups.items():
        js=[features.index(k) for k in cols if k in features]
        if not js:continue
        values=[]
        for rep in range(IMPORTANCE_REPEATS):
            perm=rng.permutation(len(x));z=x.copy();z[:,js]=x[perm][:,js]
            values.append(100*(base-balanced_accuracy_score(y,model.predict(z))))
        output.append({'feature':name,'mean_drop_pp':float(np.mean(values)),
                       'sd_drop_pp':float(np.std(values,ddof=1)) if len(values)>1 else 0.,
                       'baseline_balanced_accuracy':float(base),'sample_n':len(d)})
    return output


def importance_analysis(root,candidates,chosen,test,features,medians):
    import joblib
    plt=figure_style()
    # Same naturally proportioned manual test subset for all algorithms.
    sample=test.nsmallest(min(IMPORTANCE_SAMPLE,len(test)),'sample_hash')
    best_each={}
    for r in candidates:
        key=r['algorithm']
        if key not in best_each or (r['balanced_accuracy'],r['macro_f1'])>(best_each[key]['balanced_accuracy'],best_each[key]['macro_f1']):best_each[key]=r
    singles=[];grouped=[]
    for alg,r in best_each.items():
        model=joblib.load(root/'models'/(r['name']+'.joblib'))['model']
        log(f'[IMPORTANCE {alg}] {len(sample):,} manual test objects, {IMPORTANCE_REPEATS} permutations')
        singles.extend(dict(algorithm=alg,**v) for v in permutation_scores(model,sample,features,medians,{k:[k] for k in features}))
        grouped.extend(dict(algorithm=alg,**v) for v in permutation_scores(model,sample,features,medians,IMPORTANCE_GROUPS))
    single=pd.DataFrame(singles);group=pd.DataFrame(grouped)
    single.to_csv(root/'permutation_importance.csv',index=False,encoding='utf-8-sig')
    group.to_csv(root/'group_permutation_importance.csv',index=False,encoding='utf-8-sig')
    fig,axs=plt.subplots(1,2,figsize=(12,7),layout='constrained',gridspec_kw={'width_ratios':[1.35,1]})
    for ax,table in zip(axs,[single,group]):
        order=table.groupby('feature').mean_drop_pp.mean().sort_values().index.tolist()
        algs=list(best_each);dy=.72/len(algs)
        for j,alg in enumerate(algs):
            t=table[table.algorithm==alg].set_index('feature').reindex(order)
            ax.barh(np.arange(len(order))+(j-(len(algs)-1)/2)*dy,t.mean_drop_pp,height=dy*.88,
                xerr=t.sd_drop_pp,color=['#559CAF','#89BEA2'][j],label=alg,error_kw={'elinewidth':.8,'capsize':2})
        ax.set(yticks=np.arange(len(order)),yticklabels=order,xlabel='Balanced accuracy decrease (percentage points)')
        ax.axvline(0,color='.5',lw=.7);ax.legend(frameon=False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axs[0].set_title('Individual features');axs[1].set_title('Jointly permuted feature groups')
    savefig(fig,root/'03_permutation_importance')
    model=joblib.load(root/'models'/(chosen['name']+'.joblib'))['model']
    v=model.feature_importances_;v=v/v.sum() if v.sum()>0 else v
    pd.DataFrame({'feature':features,'normalized_model_importance':v}).to_csv(root/'model_native_importance.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.6,6),layout='constrained');ix=np.argsort(v)
    ax.barh(np.array(features)[ix],100*v[ix],color='#89BEA2');ax.set_xlabel('Model-native importance (%)')
    savefig(fig,root/'04_model_native_importance')
    corr=sample[features].corr(method='spearman')
    corr.to_csv(root/'feature_spearman.csv')
    fig,ax=plt.subplots(figsize=(9,8),layout='constrained');im=ax.imshow(corr,cmap='RdBu_r',vmin=-1,vmax=1)
    ax.set(xticks=range(len(features)),xticklabels=features,yticks=range(len(features)),yticklabels=features)
    plt.setp(ax.get_xticklabels(),rotation=60,ha='right');fig.colorbar(im,ax=ax,shrink=.7,label='Spearman correlation')
    savefig(fig,root/'05_feature_correlation')


def sample_and_month_figures(root,train,valid,test,p):
    from sklearn.metrics import balanced_accuracy_score,recall_score
    plt=figure_style();fig,axs=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for j,c in enumerate([1,2]):
        axs[0].bar(np.arange(3)+(j-.5)*.3,[int((d.label==c).sum()) for d in (train,valid,test)],width=.29,color=['#89BEA2','#559CAF'][j],label=['Lake','River'][j])
    axs[0].set(xticks=range(3),xticklabels=['Train pool','Validation','Test'],ylabel='Sample count');axs[0].legend(frameon=False)
    tt=test.copy();tt['p']=p;rows=[]
    for m,g in tt.groupby('month'):
        met=metrics_for(g.label,g.p);rows.append({'month':int(m),**met})
    jsave(root/'monthly_test_metrics.json',rows)
    for key,name,color in [('balanced_accuracy','Balanced accuracy','#559CAF'),('lake_recall','Lake recall','#89BEA2'),('river_recall','River recall','#D397A3')]:
        axs[1].plot([r['month'] for r in rows],[100*r[key] if r[key] is not None else np.nan for r in rows],'-o',ms=4,lw=1.4,label=name,color=color)
    axs[1].set(xticks=range(1,13),xticklabels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],ylabel='Manual test score (%)',ylim=(0,103));axs[1].tick_params(axis='x',labelsize=8);axs[1].legend(frameon=False,fontsize=9)
    savefig(fig,root/'06_samples_and_monthly_accuracy')


def fit_evaluate(root,summaries):
    import joblib
    marker=root/'EVALUATION_DONE.json'
    if marker.exists():return jload(marker)
    train,valid,test=sample_pools(root,summaries)
    features,medians,dropped=medians_and_features(train)
    jsave(root/'feature_processing.json',{'features':features,'training_medians':medians,'dropped':dropped})
    for name,d in zip(['train','validation','test'],[train,valid,test]):
        dump_parquet(d[['obj_id','year','month','label','is_manual','block_x','block_y']],root/(name+'_membership.parquet'))
    algorithms=(['RF'] if RUN_RF_TEST else [])+(['LightGBM'] if RUN_LIGHTGBM_TEST else [])
    records=[];(root/'models').mkdir(exist_ok=True)
    vx=matrix(valid,features,medians)
    for alg in algorithms:
        for strategy in STRATEGIES:
            name=alg+'_'+strategy;path=root/'models'/(name+'.joblib')
            if path.exists():
                bundle=joblib.load(path);model=bundle['model'];info=bundle['fit_info']
                log(f'[MODEL CACHE] {name}')
            else:
                model,info=train_one(alg,strategy,train,valid,features,medians)
                bundle={'model':model,'features':features,'medians':medians,'fit_info':info}
                tmp=path.with_suffix('.tmp');joblib.dump(bundle,tmp,compress=3);os.replace(tmp,path)
            met=metrics_for(valid.label,p_river(model,vx));records.append({'name':name,**info,**met})
            log(f'[VALID {name}] balanced accuracy={met["balanced_accuracy"]:.4f}; macro F1={met["macro_f1"]:.4f}')
    pd.DataFrame([{k:v for k,v in r.items() if k!='confusion_matrix'} for r in records]).to_csv(root/'validation_comparison.csv',index=False,encoding='utf-8-sig')
    chosen=max(records,key=lambda r:(r['balanced_accuracy'],r['macro_f1'],-r['seconds']))
    bundle=joblib.load(root/'models'/(chosen['name']+'.joblib'));model=bundle['model']
    tx=matrix(test,features,medians)
    algorithm_tests=[]
    for alg in algorithms:
        champion=max([r for r in records if r['algorithm']==alg],key=lambda r:(r['balanced_accuracy'],r['macro_f1'],-r['seconds']))
        tested=joblib.load(root/'models'/(champion['name']+'.joblib'))['model']
        prob=p_river(tested,tx)
        algorithm_tests.append({'algorithm':alg,'model':champion['name'],**metrics_for(test.label,prob)})
        figure_dir=root/(alg+'_test');figure_dir.mkdir(exist_ok=True)
        figure_test(figure_dir,test.label,prob)
    jsave(root/'algorithm_test_metrics.json',algorithm_tests)
    pd.DataFrame([{k:v for k,v in r.items() if k!='confusion_matrix'} for r in algorithm_tests]).to_csv(root/'algorithm_test_metrics.csv',index=False,encoding='utf-8-sig')
    p=p_river(model,tx);testmetrics=metrics_for(test.label,p)
    figure_comparison(root,records);figure_test(root,test.label,p)
    importance_analysis(root,records,chosen,test,features,medians)
    sample_and_month_figures(root,train,valid,test,p)
    # Deploy the exact evaluated model; no refit on validation/test labels.
    joblib.dump(bundle,root/'selected_model.joblib',compress=3)
    if chosen['algorithm']=='LightGBM':model.booster_.save_model(str(root/'selected_lightgbm.txt'))
    jsave(root/'selected_parameters.json',model.get_params())
    result={'chosen':chosen['name'],'selection':'validation balanced_accuracy, then macro_f1; test not used for selection',
        'test':testmetrics,'passed':testmetrics['accuracy']>MIN_TEST_ACCURACY and testmetrics['balanced_accuracy']>=PREDICTION_MIN_BALANCED_ACCURACY,
        'features':features,'dropped':dropped,'train_n':len(train),'valid_n':len(valid),'test_n':len(test)}
    jsave(root/'evaluation_report.json',result)
    allcounts=pd.read_csv(root/'all_label_counts.csv')
    total_lake=int(allcounts.loc[allcounts.label==1,'n'].sum());total_river=int(allcounts.loc[allcounts.label==2,'n'].sum())
    manual_n=int(allcounts.loc[allcounts.is_manual==1,'n'].sum());rule_n=int(allcounts.loc[allcounts.is_manual==0,'n'].sum())
    report=f'''Lake and river classification
Output: {root}
Selected model: {chosen['name']}
All labelled object-month records: {total_lake:,} lakes, {total_river:,} rivers; {manual_n:,} manual, {rule_n:,} reference-based.
Training pool / manual validation / manual test: {len(train):,} / {len(valid):,} / {len(test):,}
Selected model training sample: {chosen['train_n']:,} (lakes {chosen['train_lake']:,}; rivers {chosen['train_river']:,})
Test accuracy: {testmetrics['accuracy']:.4%}
Balanced accuracy: {testmetrics['balanced_accuracy']:.4%}
Macro F1: {testmetrics['macro_f1']:.4f}
Lake precision / recall: {testmetrics['lake_precision']:.4f} / {testmetrics['lake_recall']:.4f}
River precision / recall: {testmetrics['river_precision']:.4f} / {testmetrics['river_recall']:.4f}
River ROC AUC: {testmetrics['river_roc_auc']:.4f}
Features: {', '.join(features)}
Dropped features: {dropped}

2018 records were assigned to spatial blocks across months. Reference labels were
used only for training and downweighted. Manual validation selected the model;
test labels did not influence selection. The evaluated model was saved without
refitting to validation or test labels. Predicted labels preserve manual labels.
The spatial split has no buffer or cross-month object tracking.
'''
    (root/'report.txt').write_text(report,encoding='utf-8-sig')
    jsave(marker,result);log(f'[TEST] {testmetrics}');return result


def verified_source(run):
    info=jload(run/'manifest.json');source=Path(info['source_path'])
    wal=Path(str(source)+'-wal')
    if wal.exists() and wal.stat().st_size:raise RuntimeError(f'Close GIS editing / active WAL: {source}')
    stamp=statkey(source)
    if stamp==info['identity']['source']:return info
    receipt=run/'LABEL_WRITEBACK.json'
    if receipt.exists():
        r=jload(receipt)
        if r['original_source']==info['identity']['source'] and r['after_source']==stamp:return info
    raise RuntimeError(f'Source changed outside verified label writeback: {source}. Not recomputing features automatically.')


def locate_feature(ym):
    year=ym.split('_')[0];parent=OUT_ROOT/year/f'WB_{ym}'
    options=[Path(FEATURE_RUN_OVERRIDES[ym])] if ym in FEATURE_RUN_OVERRIDES else sorted(parent.glob('*'))
    ok=[];issues=[]
    for run in options:
        if not (run/'DONE.json').exists() or not (run/'manifest.json').exists():continue
        try:
            verified_source(run);meta=jload(run/'DONE.json')
            for r in meta['parts']:
                fp=run/'features'/r['file']
                if not fp.exists() or fp.stat().st_size!=r['bytes']:raise RuntimeError(f'Changed feature part: {fp}')
            ok.append(run)
        except Exception as e:issues.append(str(e))
    if len(ok)!=1:raise RuntimeError(f'{ym}: expected ONE completed feature run, found {len(ok)}. Set FEATURE_RUN_OVERRIDES if needed. Details: {issues}')
    return ok[0]


def predict_labels(root,ym,run,bundle,summaries,model_hash):
    info=verified_source(run);meta=jload(run/'DONE.json')
    identity={'model_sha256':model_hash,'features':feature_signature(run),'field':OUTPUT_LABEL_FIELD}
    dest=root/'predictions'/ym/digest_obj(identity);dest.mkdir(parents=True,exist_ok=True)
    jsave(dest/'identity.json',identity)
    manual=manual_lookup(root,ym,summaries)
    features=bundle['features'];med=bundle['medians'];model=bundle['model']
    count=0;counts={0:0,1:0,2:0};areas={0:0.,1:0.,2:0.}
    for r in meta['parts']:
        src=run/'features'/r['file'];out=dest/r['file']
        if not out.exists():
            d=pd.read_parquet(src)
            x=d[features].to_numpy(dtype=float)
            good=np.isfinite(x).any(axis=1)&(d.geom_ok.to_numpy()!=0)
            lab=np.zeros(len(d),dtype='int8')
            if good.any():lab[good]=model.predict(matrix(d.loc[good],features,med)).astype('int8')
            if len(manual):
                ids=canonical_ids(d.obj_id);loc=manual.index.get_indexer(ids);has=loc>=0
                lab[has]=manual.iloc[loc[has]].to_numpy(dtype='int8')
            # IDs are retained only in small prediction cache; vector gets ONE field.
            dump_parquet(pd.DataFrame({'source_fid':d.source_fid.to_numpy(),'label':lab}),out)
        z=pd.read_parquet(out);a=pd.read_parquet(src,columns=['area_km2']).area_km2.to_numpy(dtype=float)
        if len(z)!=len(a):raise RuntimeError('Prediction/feature row mismatch')
        count+=len(z)
        for c in counts:
            ix=z.label.to_numpy()==c;counts[c]+=int(ix.sum());areas[c]+=float(np.nansum(a[ix]))
        log(f'[PREDICT {ym}] {count:,} objects; cached labels only')
    record={'month':ym,'n':count,'counts':counts,'areas_km2':areas,'source':info['source_path'],'prediction_dir':str(dest)}
    jsave(dest/'DONE.json',record)
    write_label_only(run,dest,identity)
    return record


def write_label_only(run,dest,identity):
    """One transaction per month; geometry and spatial index are not rebuilt."""
    info=verified_source(run);source=Path(info['source_path']);before=statkey(source)
    receiptpath=run/'LABEL_WRITEBACK.json'
    if receiptpath.exists():
        old=jload(receiptpath)
        if old['prediction_identity']==identity and old['after_source']==before:
            log(f'[WRITEBACK SKIP] {source}');return
    lock=source.with_suffix(source.suffix+'.qtp_label.lock')
    try:fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'Writeback lock exists: {lock}. Remove only if previous process stopped.')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    con=None
    try:
        con=sqlite3.connect(str(source),timeout=60)
        # GPKG spatial triggers can refer to these functions even on attribute-only UPDATE.
        # Preserve all triggers; functions decode real GPKG geometry if invoked.
        def geom(blob):return sh.from_wkb(decode_gpkg(blob)) if blob is not None else None
        con.create_function('ST_IsEmpty',1,lambda b:1 if b is None else int(sh.is_empty(geom(b))))
        for name,j in [('ST_MinX',0),('ST_MinY',1),('ST_MaxX',2),('ST_MaxY',3)]:
            con.create_function(name,1,lambda b,j=j:None if b is None else float(sh.bounds(geom(b))[j]))
        con.execute('PRAGMA busy_timeout=60000');con.execute('PRAGMA temp_store=FILE')
        con.execute('PRAGMA cache_size=-65536');con.execute('PRAGMA synchronous=FULL')
        table=quote(info['source_layer']);pk=quote(info['pk']);field=quote(OUTPUT_LABEL_FIELD)
        existing={r[1].lower():r[1] for r in con.execute(f'PRAGMA table_info({table})')}
        if OUTPUT_LABEL_FIELD.lower() in existing:field=quote(existing[OUTPUT_LABEL_FIELD.lower()])
        con.execute('BEGIN IMMEDIATE')
        if statkey(source)!=before:raise RuntimeError('Source changed before write lock')
        if OUTPUT_LABEL_FIELD.lower() not in existing:con.execute(f'ALTER TABLE {table} ADD COLUMN {field} INTEGER')
        con.execute('CREATE TEMP TABLE qtp_labels (id INTEGER PRIMARY KEY, value INTEGER NOT NULL CHECK(value IN (0,1,2)))')
        total=0
        backup=dest/'previous_labels';backup.mkdir(exist_ok=True)
        for part in sorted(dest.glob('part-*.parquet')):
            d=pd.read_parquet(part)
            con.execute('DELETE FROM qtp_labels')
            con.executemany('INSERT INTO qtp_labels VALUES (?,?)',((int(a),int(b)) for a,b in zip(d.source_fid,d.label)))
            rows=con.execute(f'SELECT s.{pk},s.{field} FROM {table} AS s JOIN qtp_labels AS p ON s.{pk}=p.id').fetchall()
            if len(rows)!=len(d):raise RuntimeError('Missing source FIDs; transaction will roll back')
            old=pd.DataFrame(rows,columns=['source_fid','previous_label'])
            # Preserve first backup for this prediction version across retries.
            if not (backup/part.name).exists():dump_parquet(old,backup/part.name)
            con.execute(f'UPDATE {table} SET {field}=(SELECT value FROM qtp_labels WHERE id={table}.{pk}) WHERE {pk} IN (SELECT id FROM qtp_labels)')
            total+=len(d);log(f'[WRITE LABEL] {total:,}; {source.name}')
        expected=int(con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0])
        if total!=expected:raise RuntimeError(f'Count mismatch {total} != {expected}; rollback')
        con.execute("UPDATE gpkg_contents SET last_change=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE table_name=?",(info['source_layer'],))
        con.commit();con.close();con=None
        jsave(receiptpath,{'original_source':info['identity']['source'],'before_source':before,'after_source':statkey(source),
            'prediction_identity':identity,'rows':total,'field':OUTPUT_LABEL_FIELD,'prediction_dir':str(dest)})
        log(f'[LABEL DONE] {source}; one attribute only, geometry retained')
    except Exception:
        if con is not None:con.rollback()
        raise
    finally:
        if con is not None:con.close()
        lock.unlink(missing_ok=True)


def rf_main():
    import joblib,sklearn,pyogrio
    dependencies();MODEL_ROOT.mkdir(parents=True,exist_ok=True)
    log('[START] RF/LightGBM balance comparison; prediction='+str(RUN_PREDICTION))
    if RUN_RF_TEST or RUN_LIGHTGBM_TEST:
        edits=find_edits();runs={ym:locate_feature(ym) for ym in edits}
        rasters=reference_paths_from_runs(runs)
        crs=[CRS.from_user_input(jload(r/'manifest.json')['crs']) for r in runs.values()]
        if any(not c.equals(crs[0],ignore_axis_order=True) for c in crs):raise RuntimeError('Training CRS differs across months')
        settings={'version':PIPELINE_VERSION_RF,'rf':RUN_RF_TEST,'lightgbm':RUN_LIGHTGBM_TEST,
            'strategies':STRATEGIES,'seed':RANDOM_SEED,'block_m':SPATIAL_BLOCK_M,
            'sample_caps':[MAX_TRAIN_SAMPLES,MAX_VALID_SAMPLES,MAX_TEST_SAMPLES],
            'rf_params':RF_PARAMS,'rf_trees':RF_TREES,'rf_max_samples':RF_MAX_SAMPLES,
            'lgb_trees':LGB_TREES,'auto_weight':AUTO_WEIGHT,'auto_cap':AUTO_CAP_PER_CLASS_MONTH,
            'auto_ratio':AUTO_MAX_MANUAL_RATIO,'all_reviewed':ALL_LABELED_REVIEWED,
            'importance_n':IMPORTANCE_SAMPLE,'importance_repeats':IMPORTANCE_REPEATS,
            'gate':[MIN_TEST_ACCURACY,PREDICTION_MIN_BALANCED_ACCURACY],
            'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__}
        if RUN_LIGHTGBM_TEST:
            import lightgbm;settings['lightgbm_version']=lightgbm.__version__
        stamps={ym:{k:[shape_signature(p) for p in ps] for k,ps in pair.items()} for ym,pair in edits.items()}
        identity={'settings':settings,'edits':stamps,'features':{ym:feature_signature(r) for ym,r in runs.items()}}
        root=MODEL_ROOT/('run_'+digest_obj(identity));root.mkdir(exist_ok=True)
    else:
        if LOAD_RUN is None:raise ValueError('Both training switches are False: set LOAD_RUN to the completed model folder')
        root=Path(LOAD_RUN);identity=jload(root/'run_inputs.json')
    lock=root/'RUNNING.lock'
    try:fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'Run already locked: {lock}; do not run twice')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        log(f'[OUTPUT] {root}')
        if RUN_RF_TEST or RUN_LIGHTGBM_TEST:
            jsave(root/'run_inputs.json',identity)
            summaries={ym:merge_month(root,ym,edits[ym],runs[ym],rasters) for ym in edits}
            now={ym:{k:[shape_signature(p) for p in ps] for k,ps in pair.items()} for ym,pair in edits.items()}
            if now!=stamps:raise RuntimeError('Manual labels changed during processing; rerun')
            result=fit_evaluate(root,summaries)
        else:
            summaries={p.parent.name:jload(p) for p in (root/'merged_labels').glob('*/DONE.json')}
            result=jload(root/'EVALUATION_DONE.json')
        if not RUN_PREDICTION:log('[DONE] Training/report/figures completed. Prediction is OFF.');return
        if not result['passed']:log('[STOP] Test accuracy or balanced accuracy below prediction gate');return
        bundle=joblib.load(root/'selected_model.joblib');bundle['model'].set_params(n_jobs=N_JOBS)
        sha=hashlib.sha256((root/'selected_model.joblib').read_bytes()).hexdigest()
        records=[];failures=[]
        for y in PREDICT_YEARS:
            for m in PREDICT_MONTHS:
                ym=f'{y}_{m:02d}'
                try:
                    run=locate_feature(ym)
                    training_manifests=identity['features']
                    first=Path(next(iter(training_manifests.values()))['run'])
                    ref=jload(first/'manifest.json')['identity']['rasters']
                    if jload(run/'manifest.json')['identity']['rasters']!=ref:raise RuntimeError('Prediction raster signatures differ from training')
                    records.append(predict_labels(root,ym,run,bundle,summaries,sha))
                except Exception as e:
                    traceback.print_exc();failures.append({'month':ym,'error':str(e)})
                jsave(root/'prediction_summary.json',records);jsave(root/'prediction_failures.json',failures)
        if failures:raise RuntimeError(f'{len(failures)} prediction months failed; inspect prediction_failures.json')
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    from multiprocessing import freeze_support
    freeze_support()
    try:rf_main()
    except Exception:
        traceback.print_exc();sys.exit(1)
