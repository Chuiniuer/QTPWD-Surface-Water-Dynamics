# -*- coding: utf-8 -*-
"""Split predicted rivers, restore reservoirs and assign lake attributes.
"""
from pathlib import Path
from datetime import datetime
from collections import Counter
import hashlib
import json
import re
import sqlite3
import struct
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import shapely
from shapely import STRtree
from pyproj import CRS
import fiona

BASE = Path(r'K:\Monthly surface water dynamics TIB\supervised classification\step5_lake_classification')
SEPARATION = Path(r'E:\phd_l3\QTP_reservoir_separation')
WATER_ROOT = SEPARATION / 'water_without_reservoirs'  
FEATURE_ROOT = Path(r'E:\phd_l3\QTP_monthly_lake_river_features')
PREDICTION_RUN = Path(r'E:\phd_l3\QTP_lake_river_RF_comparison\run_8a6d695e43bd9cd2acc8')

FEATURE_RUN_OVERRIDES = {}  # {'2018_01': Path(r'E:\...\WB_2018_01\fingerprint')}
SOURCE_LAYERS = {}

RESERVOIR_ROOT = None

RESERVOIR_FILES = {}
REF = BASE / 'lake_type_references_QTP_Albers'
GLACIER = REF / 'RGI7_glaciers_QTP_Albers.gpkg'
PERMAFROST = REF / 'Obu_permafrost_zones_QTP_Albers.gpkg'
BOUNDARY = Path(r'D:\phdl2\QTP surface water dynamics\regions\QTP.shp')
OUTPUT_ROOT = Path(r'E:\phd_l3\QTP_monthly_water_types')
PROCESS_ALL_YEARS = True  
TEST_YEAR = 2018
TEST_MONTHS = [1]
YEARS = range(2016, 2026)
USE_GLACIAL = True       
GLACIER_DISTANCE_M = 10000.0
THERMO_MAX_AREA_KM2 = 3.0
INCLUDE_ISOLATED = False
BATCH_SIZE = 50000
REBUILD = False
SPATIAL_INDEX = True  


LABEL_FIELD = 'label'
VERSION = '2.0-medium-large'
MED_LARGE_MIN_AREA_KM2 = 1.0
UPGRADE_COMPLETED = True  
TYPE_NAMES = {
    0: 'River', 1: 'Other lake', 2: 'Medium and large lake',
    3: 'Glacier-associated lake', 4: 'Thermokarst candidate',
    5: 'Medium and large + glacier-associated lake',
    6: 'Medium and large + thermokarst candidate',
    7: 'Glacier-associated + thermokarst candidate',
    8: 'Medium and large + glacier-associated + thermokarst candidate',
    9: 'Reservoir', 99: 'Lake with unresolved reference/geometry'}

COMBINATION_CODES = np.array([1, 2, 3, 5, 4, 6, 7, 8], dtype='int16')

def log(message):
    print(f'{datetime.now():%H:%M:%S} {message}', flush=True)

def file_stamp(path):
    st = path.stat()
    return [str(path), st.st_size, st.st_mtime_ns]

def layer_info(path):
    layers = pyogrio.list_layers(path)
    spatial = [str(row[0]) for row in layers if row[1] is not None]
    preferred = SOURCE_LAYERS.get(str(Path(path).resolve()))
    if preferred is not None:
        if preferred not in spatial:
            raise ValueError(f'Recorded prediction layer not found: {path}；{preferred}')
        name = preferred
    elif 'waterbodies' in spatial:
        name = 'waterbodies'
    elif len(spatial) == 1:
        name = spatial[0]
    else:
        raise ValueError(f'Cannot uniquely identify layer: {path}; layers={spatial}')
    info = pyogrio.read_info(path, layer=name)
    if not info['crs']:
        raise ValueError(f'Missing CRS: {path}')
    return name, info

def quoted(value):
    return '"' + value.replace('"', '""') + '"'

def actual_statkey(path):
    path = Path(path).resolve()
    st = path.stat()
    return {'path': str(path), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def resolve_prediction_sources(months):
    """Resolve sources from the prediction manifest and writeback records."""
    model = PREDICTION_RUN / 'selected_model.joblib'
    if not model.is_file():
        raise FileNotFoundError(f'Model file not found: {model}; set PREDICTION_RUN.')
    digest = hashlib.sha256()
    with model.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    expected_model = digest.hexdigest()
    result = {}
    audit = []
    for year, month in months:
        ym = f'{year}_{month:02d}'
        parent = FEATURE_ROOT / str(year) / f'WB_{ym}'
        options = [Path(FEATURE_RUN_OVERRIDES[ym])] if ym in FEATURE_RUN_OVERRIDES else sorted(parent.glob('*'))
        candidates, issues = [], []
        for run in options:
            if not run.is_dir():
                continue
            manifest, receipt, done = run/'manifest.json', run/'LABEL_WRITEBACK.json', run/'DONE.json'
            if not all(p.is_file() for p in (manifest, receipt, done)):
                continue
            try:
                info = json.loads(manifest.read_text(encoding='utf-8'))
                record = json.loads(receipt.read_text(encoding='utf-8'))
                source = Path(info['source_path'])
                if not source.is_file():
                    raise ValueError(f'Prediction source not found: {source}')
                wal = Path(str(source) + '-wal')
                if wal.exists() and wal.stat().st_size:
                    raise ValueError(f'Close GIS editing and commit changes before retrying: {source}')
                if source.with_suffix(source.suffix+'.qtp_label.lock').exists():
                    raise ValueError(f'Prediction writeback lock exists: {source}')
                if record['original_source'] != info['identity']['source']:
                    raise ValueError('Writeback source does not match manifest')
                if record['after_source'] != actual_statkey(source):
                    raise ValueError('Source was modified after prediction')
                if record['prediction_identity']['model_sha256'] != expected_model:
                    raise ValueError('Prediction model differs from PREDICTION_RUN')
                field = record.get('field', record['prediction_identity']['field'])
                if field.lower() != LABEL_FIELD.lower():
                    raise ValueError(f'Prediction field {field} differs from LABEL_FIELD')
                layer = info['source_layer']
                SOURCE_LAYERS[str(source.resolve())] = layer
                _, metadata = layer_info(source)
                actual_field = next((f for f in metadata['fields'] if f.lower() == LABEL_FIELD.lower()), None)
                if actual_field is None:
                    raise ValueError(f'Prediction layer lacks label field: {source} / {layer}')
                if int(record['rows']) != int(metadata['features']):
                    raise ValueError('Prediction row count does not match source layer')
                candidates.append((source, run, layer, actual_field))
            except Exception as exc:
                issues.append(f'{run}: {exc}')
        if len(candidates) != 1:
            raise RuntimeError(f'{ym} does not have a unique valid prediction run: {len(candidates)}.\n'
                               f'Check FEATURE_ROOT/PREDICTION_RUN and set FEATURE_RUN_OVERRIDES if needed.\n'
                               + '\n'.join(issues))
        source, run, layer, field = candidates[0]
        result[(year, month)] = source
        audit.append({'month': ym, 'source_path': str(source), 'source_layer': layer,
                      'label_field': field, 'feature_run': str(run)})
        log(f'[PREDICTED SOURCE {ym}] {source} | layer={layer} | field={field}')
    return result, audit


def find_reservoirs(months):
    root = Path(RESERVOIR_ROOT) if RESERVOIR_ROOT else SEPARATION
    files = list(root.rglob('*.gpkg'))
    result = {}
    for year, month in months:
        if (year, month) in RESERVOIR_FILES:
            candidates = [Path(RESERVOIR_FILES[(year, month)])]
        else:
            pattern = re.compile(rf'(?<!\d){year}[_-]0?{month}(?!\d)')
            candidates = []
            for path in files:
                parts = [p.lower() for p in path.parts]
                if any('without' in p or 'working' in p or 'backup' in p for p in parts):
                    continue
                if not RESERVOIR_ROOT and not any('reservoir' in p for p in parts[len(SEPARATION.parts):]):
                    continue
                if pattern.search(path.stem):
                    candidates.append(path)
        if len(candidates) != 1 or not candidates[0].is_file():
            raise RuntimeError(f'{year}-{month:02d} reservoir file not uniquely found: {candidates}\n'
                               'Set RESERVOIR_ROOT or RESERVOIR_FILES; a missing file is not an empty month.')
        result[(year, month)] = candidates[0]
    return result

def same_crs(info, target, path):
    if not CRS.from_user_input(info['crs']).equals(target, ignore_axis_order=True):
        raise ValueError(f'CRS mismatch: {path}; use previously projected reference data.')

def load_references(target):
    crop = None
    if BOUNDARY.is_file():
        outline = gpd.read_file(BOUNDARY).to_crs(target)
        x0, y0, x1, y1 = outline.total_bounds
        crop = (x0-20000, y0-20000, x1+20000, y1+20000)
    refs = {}
    for name, path in [('permafrost', PERMAFROST)] + ([('glacier', GLACIER)] if USE_GLACIAL else []):
        layer, info = layer_info(path)
        same_crs(info, target, path)
        columns = None if name == 'permafrost' else []
        log(f'[REFERENCE] {path.name}')
        frame = pyogrio.read_dataframe(path, layer=layer, bbox=crop, columns=columns)
        frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
        
        invalid = ~frame.geometry.is_valid
        if invalid.any():
            frame.loc[invalid, frame.geometry.name] = shapely.make_valid(frame.geometry[invalid].to_numpy())
        if name == 'permafrost':
            extent = next((c for c in frame.columns if c.lower() == 'extent'), None)
            code = next((c for c in frame.columns if c.lower() == 'gridcode'), None)
            if extent is None:
                raise ValueError('Permafrost data lacks EXTENT.')
            normalized = frame[extent].fillna('').astype(str).str.lower().str.replace(r'[^a-z]', '', regex=True)
            mapping = {'cont': 4, 'continuous': 4, 'discon': 3, 'discontinuous': 3,
                       'spora': 2, 'sporadic': 2, 'isol': 1, 'isolated': 1, 'isolatedpatches': 1}
            zone = normalized.map(mapping)
            if zone.isna().any():
                raise ValueError(f'Unrecognised EXTENT values: {frame.loc[zone.isna(), extent].unique()}; check the input.')
            frame['_zone'] = zone.astype('int8')
            log('[PF CATEGORY] ' + str(frame.groupby([extent] + ([code] if code else [])).size().to_dict()))
        frame = frame.explode(index_parts=False, ignore_index=True)
        if crop is not None:
            frame = frame.loc[frame.geometry.intersects(shapely.box(*crop))].reset_index(drop=True)
        if not len(frame):
            raise ValueError(f'Empty reference data: {path}')
        refs[name] = STRtree(frame.geometry.to_numpy())
        if name == 'permafrost':
            refs['zones'] = frame['_zone'].to_numpy()
    return refs

def area_field(fields):
    
    for wanted in ('q_area', 'area_km2', 'Area_km2'):
        if wanted in fields:
            return wanted, 1.0
    if 'Shape_Area' in fields:
        return 'Shape_Area', 1e-6
    raise ValueError(f'No recognised area field: {fields}; add the correct field name.')

def classify(frame, refs, reservoir, year, month, area_name, area_factor):
    n = len(frame)
    geometry = frame.geometry.to_numpy()
    valid = ~shapely.is_missing(geometry) & ~shapely.is_empty(geometry) & shapely.is_valid(geometry)
    types = shapely.get_type_id(geometry)
    if np.any(~np.isin(types[valid], [3, 6])):
        raise ValueError('Non-polygon geometry encountered.')
    areas = pd.to_numeric(frame[area_name], errors='coerce').to_numpy(dtype=float) * area_factor
    if np.any(~np.isfinite(areas) | (areas < 0)):
        raise ValueError('Area values are missing or negative.')
    labels = np.ones(n, dtype='int8') if reservoir else pd.to_numeric(frame[LABEL_FIELD], errors='coerce').to_numpy()
    if not np.isin(labels, [1, 2]).all():
        raise ValueError('Invalid or missing lake/river labels; inspect predictions.')
    is_lake = labels == 1
    glacier = np.full(n, -1, dtype='int8')
    thermo = np.full(n, -1, dtype='int8')
    zones = np.full(n, -1, dtype='int8')
    final = np.zeros(n, dtype='int8')
    med_large = np.full(n, -1, dtype='int8')
    if not reservoir:
        med_large[is_lake] = (areas[is_lake] >= MED_LARGE_MIN_AREA_KM2).astype('int8')
    final[is_lake] = 99
    if reservoir:
        final[:] = 9
    else:
        idx = np.flatnonzero(is_lake & valid)
        if len(idx):
            points = shapely.point_on_surface(geometry[idx])
            pairs = refs['permafrost'].query(points, predicate='intersects')
            z = np.full(len(idx), -1, dtype='int8')
            if pairs.shape[1]:
                
                np.maximum.at(z, pairs[0], refs['zones'][pairs[1]])
            zones[idx] = z
            t = np.full(len(idx), -1, dtype='int8')
            t[areas[idx] >= THERMO_MAX_AREA_KM2] = 0
            known = z >= 1
            t[known] = ((areas[idx][known] < THERMO_MAX_AREA_KM2) &
                        (z[known] >= (1 if INCLUDE_ISOLATED else 2))).astype('int8')
            thermo[idx] = t
            if USE_GLACIAL:
                
                g = np.zeros(len(idx), dtype='int8')
                pair = refs['glacier'].query_nearest(geometry[idx], max_distance=GLACIER_DISTANCE_M,
                                                   all_matches=False, return_distance=True)
                pair, distances = pair
                if pair.shape[1]:
                    g[np.unique(pair[0][distances <= GLACIER_DISTANCE_M])] = 1
                glacier[idx] = g
                known_type = t >= 0
                final[idx[known_type]] = COMBINATION_CODES[med_large[idx[known_type]] + 2*g[known_type] + 4*t[known_type]]
            else:
                known_type = t >= 0
                final[idx[known_type]] = COMBINATION_CODES[med_large[idx[known_type]] + 4*t[known_type]]
    
    obj = frame['obj_id'].fillna('').astype(str).to_numpy() if 'obj_id' in frame else np.full(n, '', dtype=object)
    out = gpd.GeoDataFrame({'src_fid': frame.index.to_numpy(dtype='int64'), 'obj_id': obj,
        'year': np.full(n, year), 'month': np.full(n, month), 'area_km2': areas,
        'label': labels.astype('int8'), 'is_res': np.full(n, int(reservoir), dtype='int8'),
        'glacial': glacier, 'thermo': thermo, 'pf_zone': zones, 'med_large': med_large, 'final_type': final},
        geometry=geometry, crs=frame.crs)
    return out

def gpkg_wkb(blob):
    if blob is None:
        return None
    if blob[:2] != b'GP' or len(blob) < 8:
        raise ValueError('Invalid GeoPackage geometry.')
    envelope = (blob[3] >> 1) & 7
    sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    if envelope not in sizes:
        raise ValueError('Invalid GeoPackage envelope type.')
    return blob[8 + sizes[envelope]:]


def normalize_srs(blob, srs_id):
    if blob is None:
        return None
    if blob[:2] != b'GP':
        raise ValueError('Invalid GeoPackage geometry.')
    endian = '<' if blob[3] & 1 else '>'
    old = struct.unpack(endian+'i', blob[4:8])[0]
    if old == srs_id:
        return blob
    
    return blob[:4] + struct.pack(endian+'i', srs_id) + blob[8:]


def gpkg_bounds(blob):
    if blob is None:
        return None
    if blob[:2] != b'GP':
        raise ValueError('Invalid GeoPackage geometry.')
    if blob[3] & 16:
        return None
    if (blob[3] >> 1) & 7:
        vals = struct.unpack(('<' if blob[3] & 1 else '>')+'dddd', blob[8:40])
        if all(np.isfinite(vals)):
            return vals  # minx,maxx,miny,maxy
    
    g = shapely.from_wkb(gpkg_wkb(blob))
    if shapely.is_empty(g):
        return None
    b = shapely.bounds(g)
    return (float(b[0]), float(b[2]), float(b[1]), float(b[3]))


def source_sql_metadata(path, target):
    layer, info = layer_info(path)
    same_crs(info, target, path)
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True) as con:
        row = con.execute('SELECT column_name,srs_id,z,m FROM gpkg_geometry_columns WHERE table_name=?',(layer,)).fetchone()
        cols = con.execute(f'PRAGMA table_info({quoted(layer)})').fetchall()
        primary = [r[1] for r in cols if r[5]]
        if len(primary) != 1:
            raise ValueError(f'Non-unique primary key: {path}')
    fields = [r[1] for r in cols]
    area, factor = area_field(fields)
    label = next((f for f in fields if f.lower()==LABEL_FIELD.lower()), None)
    obj = next((f for f in fields if f.lower()=='obj_id'), None)
    return {'layer':layer,'geom':row[0],'srs':row[1],'z':row[2],'m':row[3],
            'pk':primary[0],'area':area,'factor':factor,'label':label,'obj':obj,'n':info['features']}


class BlobWriter:
    """Copy source geometry using SQLite and defer index updates."""
    def __init__(self, path, layer, target):
        self.path, self.layer = path, layer
        schema = {'geometry':'Unknown','properties':{
            'src_fid':'int64','obj_id':'str:80','year':'int','month':'int','area_km2':'float',
            'label':'int','is_res':'int','glacial':'int','thermo':'int','pf_zone':'int','med_large':'int','final_type':'int'}}
        with fiona.open(path,'w',driver='GPKG',layer=layer,schema=schema,crs_wkt=target.to_wkt(),
                        SPATIAL_INDEX='YES' if SPATIAL_INDEX else 'NO'):
            pass
        self.con = sqlite3.connect(path, timeout=60, uri=True)
        self.con.execute('PRAGMA synchronous=NORMAL')
        self.con.execute('PRAGMA cache_size=-65536')
        self.con.execute('PRAGMA temp_store=FILE')
        self.geom, self.srs = self.con.execute('SELECT column_name,srs_id FROM gpkg_geometry_columns WHERE table_name=?',(layer,)).fetchone()
        self.con.execute('UPDATE gpkg_geometry_columns SET z=2,m=2 WHERE table_name=?',(layer,))
        self.triggers = self.con.execute('SELECT name,sql FROM sqlite_master WHERE type=\'trigger\' AND tbl_name=?',(layer,)).fetchall()
        for name, _ in self.triggers:
            self.con.execute(f'DROP TRIGGER {quoted(name)}')
        self.con.commit()
        self.sql = f'INSERT INTO {quoted(layer)} (fid,{quoted(self.geom)},src_fid,obj_id,year,month,area_km2,label,is_res,glacial,thermo,pf_zone,med_large,final_type) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)'
        self.attached = []
        self.pending_index = []
        self.copy_seconds = 0.0

    def attach(self, path, alias):
        self.con.execute(f'ATTACH DATABASE ? AS {quoted(alias)}',(Path(path).resolve().as_uri()+'?mode=ro',))
        self.attached.append(alias)

    def copy_selected(self, path, meta, alias, predicate, year, month, reservoir=False, offset=0):
        self.attach(path, alias)
        a, t = quoted(alias), quoted(meta['layer'])
        pk, geom = quoted(meta['pk']), quoted(meta['geom'])
        area = f'{quoted(meta["area"])}*{meta["factor"]}'
        obj = f'CAST({quoted(meta["obj"])} AS TEXT)' if meta['obj'] else "''"
        if meta['srs'] == self.srs:
            geom_expr = geom
        else:
            self.con.create_function('QTP_SetSRS',1,lambda blob:normalize_srs(blob,self.srs))
            geom_expr = f'QTP_SetSRS({geom})'
        lab, code = (1,9) if reservoir else (2,0)
        sql = f'''INSERT INTO {quoted(self.layer)}
        (fid,{quoted(self.geom)},src_fid,obj_id,year,month,area_km2,label,is_res,glacial,thermo,pf_zone,med_large,final_type)
        SELECT {pk}+{offset},{geom_expr},{pk},COALESCE({obj},''),{year},{month},{area},{lab},{int(reservoir)},-1,-1,-1,-1,{code}
        FROM {a}.{t} WHERE {predicate}'''
        start = time.monotonic()
        last = [start]
        def progress():
            now = time.monotonic()
            if now-last[0]>=20:
                log(f'[COPY {self.layer}] Copying database; {(now-start)/60:.1f}min')
                last[0]=now
            return 0
        self.con.set_progress_handler(progress, 100000)
        try:
            self.con.execute(sql)
            self.con.commit()
        finally:
            self.con.set_progress_handler(None, 0)
        self.copy_seconds += time.monotonic()-start
        self.pending_index.append((alias, meta, int(reservoir)))

    def append_lakes(self, blobs, attrs):
        cols = ['src_fid','obj_id','year','month','area_km2','glacial','thermo','pf_zone','med_large','final_type']
        arrays = [attrs[c].to_numpy() for c in cols]
        rows = ((int(fid),normalize_srs(blob,self.srs),int(fid),str(obj),int(year),int(month),float(area),
                 1,0,int(glacial),int(thermo),int(zone),int(large),int(final))
                for blob,fid,obj,year,month,area,glacial,thermo,zone,large,final in zip(blobs,*arrays))
        self.con.executemany(self.sql, rows)
        self.con.commit()

    def finish(self, expected):
        start = time.monotonic()
        layer = quoted(self.layer)
        actual = self.con.execute(f'SELECT COUNT(*) FROM {layer}').fetchone()[0]
        if actual != expected:
            raise RuntimeError(f'{self.layer} count mismatch: {actual}!={expected}')
        rt = 'rtree_'+self.layer+'_'+self.geom
        if SPATIAL_INDEX:
            log(f'[INDEX {self.layer}] Copying and building spatial index')
            for alias, meta, is_res in self.pending_index:
                source_rt = 'rtree_'+meta['layer']+'_'+meta['geom']
                has = self.con.execute(f'SELECT 1 FROM {quoted(alias)}.sqlite_master WHERE type=\'table\' AND name=?',(source_rt,)).fetchone()
                if has:
                    self.con.execute(f'''INSERT OR REPLACE INTO {quoted(rt)} (id,minx,maxx,miny,maxy)
                        SELECT d.fid,r.minx,r.maxx,r.miny,r.maxy FROM {layer} AS d
                        JOIN {quoted(alias)}.{quoted(source_rt)} AS r ON r.id=d.src_fid
                        WHERE d.is_res=?''',(is_res,))
            self.con.commit()
            
            cursor = self.con.execute(f'''SELECT d.fid,d.{quoted(self.geom)} FROM {layer} AS d
                       LEFT JOIN {quoted(rt)} AS r ON d.fid=r.id
                       WHERE r.id IS NULL AND d.{quoted(self.geom)} IS NOT NULL''')
            while True:
                rows = cursor.fetchmany(BATCH_SIZE)
                if not rows:
                    break
                bounds = []
                for fid, blob in rows:
                    b = gpkg_bounds(blob)
                    if b is not None:
                        bounds.append((fid,*b))
                self.con.executemany(f'INSERT INTO {quoted(rt)} VALUES (?,?,?,?,?)', bounds)
            self.con.commit()
            extent = self.con.execute(f'SELECT min(minx),min(miny),max(maxx),max(maxy) FROM {quoted(rt)}').fetchone()
            self.con.execute('UPDATE gpkg_contents SET min_x=?,min_y=?,max_x=?,max_y=? WHERE table_name=?',(*extent,self.layer))
        else:
            
            boxes=[]
            for alias,meta,_ in self.pending_index:
                b=self.con.execute(f'SELECT min_x,min_y,max_x,max_y FROM {quoted(alias)}.gpkg_contents WHERE table_name=?',(meta['layer'],)).fetchone()
                if b and all(v is not None for v in b):boxes.append(b)
            if boxes:
                ext=(min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes))
                self.con.execute('UPDATE gpkg_contents SET min_x=?,min_y=?,max_x=?,max_y=? WHERE table_name=?',(*ext,self.layer))
        
        for _, sql in self.triggers:
            self.con.execute(sql)
        if self.con.execute("SELECT 1 FROM sqlite_master WHERE name='gpkg_ogr_contents'").fetchone():
            self.con.execute('UPDATE gpkg_ogr_contents SET feature_count=? WHERE table_name=?',(actual,self.layer))
        self.con.execute("UPDATE gpkg_contents SET last_change=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE table_name=?",(self.layer,))
        self.con.commit()
        for alias in self.attached:
            self.con.execute(f'DETACH DATABASE {quoted(alias)}')
        self.con.close()
        return time.monotonic()-start

    def close_on_error(self):
        try:
            self.con.rollback()
            self.con.close()
        except Exception:
            pass


def check_source_attributes(path, meta, reservoir=False):
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True) as con:
        table, area = quoted(meta['layer']), quoted(meta['area'])
        
        expr = f'COUNT(*),SUM(CASE WHEN {area} IS NULL OR typeof({area}) NOT IN (\'real\',\'integer\') OR {area}<0 THEN 1 ELSE 0 END)'
        if not reservoir:
            if meta['label'] is None:
                raise ValueError(f'Missing prediction labels: {path}')
            lab = quoted(meta['label'])
            expr += f',SUM(CASE WHEN {lab} IS NULL OR {lab} NOT IN (1,2) THEN 1 ELSE 0 END),SUM(CASE WHEN {lab}=1 THEN 1 ELSE 0 END),SUM(CASE WHEN {lab}=2 THEN 1 ELSE 0 END),SUM(CASE WHEN {lab}=2 THEN {area}*{meta["factor"]} ELSE 0 END)'
        else:
            expr += f',SUM({area}*{meta["factor"]})'
        values = con.execute(f'SELECT {expr} FROM {table}').fetchone()
    if values[1]:
        raise ValueError(f'{path} has {values[1]} missing, negative or non-numeric areas.')
    if not reservoir and values[2]:
        raise ValueError(f'{path} has {values[2]} invalid lake/river labels.')
    return values


def process_month(year, month, water, reservoir, target, refs, signature):
    folder = OUTPUT_ROOT / str(year) / f'{month:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    outputs = [folder/'River.gpkg',folder/'LakeAndReservior.gpkg']
    marker = folder/'completed.json'
    if marker.exists() and all(p.exists() for p in outputs) and not REBUILD:
        old = json.loads(marker.read_text(encoding='utf-8'))
        if old.get('signature')==signature and old.get('outputs')==[file_stamp(p) for p in outputs]:
            log(f'[SKIP] {year}-{month:02d}')
            return
    if any(p.exists() for p in outputs) and not REBUILD:
        raise RuntimeError(f'Existing output does not match cache: {folder}; protect any manual edits.')
    meta = source_sql_metadata(water,target)
    resmeta = source_sql_metadata(reservoir,target)
    log(f'[CHECK] {year}-{month:02d}; checking labels and areas')
    t0=time.monotonic()
    values=check_source_attributes(water,meta)
    resvalues=check_source_attributes(reservoir,resmeta,True)
    n,nlake,nriver=int(values[0]),int(values[3] or 0),int(values[4] or 0)
    nres=int(resvalues[0])
    log(f'[PLAN] copy {nriver:,} rivers and {nres:,} reservoirs; classify {nlake:,} lakes')
    times={'attribute_check_s':time.monotonic()-t0,'lake_read_s':0.,'lake_classify_s':0.,'lake_write_s':0.}
    temps=[folder/'River_working.gpkg',folder/'LakeAndReservior_working.gpkg']
    for path in temps:
        if path.exists():path.unlink()
    river=BlobWriter(temps[0],'River',target)
    lake=BlobWriter(temps[1],'LakeAndReservior',target)
    counts,totals=Counter({0:nriver}),Counter({0:float(values[5] or 0)})
    try:
        log('[COPY RIVER] Copying source geometries')
        river.copy_selected(water,meta,'water_source',f'{quoted(meta["label"])}=2',year,month)
        times['river_copy_s']=river.copy_seconds
        lake.attach(water,'water_source')
        lake.pending_index.append(('water_source',meta,0))
        con=sqlite3.connect(Path(water).resolve().as_uri()+'?mode=ro',uri=True)
        try:
            obj=quoted(meta['obj']) if meta['obj'] else "''"
            sql=f'SELECT {quoted(meta["pk"])},{quoted(meta["geom"])},{obj},{quoted(meta["area"])} FROM {quoted(meta["layer"])} WHERE {quoted(meta["label"])}=1 ORDER BY {quoted(meta["pk"])}'
            cursor=con.execute(sql)
            processed=0
            while True:
                tick=time.monotonic()
                rows=cursor.fetchmany(BATCH_SIZE)
                if not rows:break
                ids=[r[0] for r in rows];blobs=[r[1] for r in rows]
                geoms=shapely.from_wkb([gpkg_wkb(b) for b in blobs])
                frame=gpd.GeoDataFrame({'obj_id':[r[2] for r in rows],LABEL_FIELD:np.ones(len(rows),dtype='int8'),
                                        meta['area']:[r[3] for r in rows]},geometry=geoms,crs=target,index=ids)
                times['lake_read_s']+=time.monotonic()-tick
                tick=time.monotonic()
                attrs=classify(frame,refs,False,year,month,meta['area'],meta['factor'])
                times['lake_classify_s']+=time.monotonic()-tick
                tick=time.monotonic()
                lake.append_lakes(blobs,attrs)
                times['lake_write_s']+=time.monotonic()-tick
                counts.update(attrs.final_type.astype(int).tolist())
                for code,value in attrs.groupby('final_type').area_km2.sum().items():totals[int(code)]+=float(value)
                processed+=len(rows)
                if processed%100000<len(rows) or processed==nlake:
                    elapsed=time.monotonic()-t0
                    log(f'[LAKES] {processed:,}/{nlake:,} ; elapsed {elapsed/60:.1f}min; lake read {times["lake_read_s"]:.1f}s, classify {times["lake_classify_s"]:.1f}s, write {times["lake_write_s"]:.1f}s')
            if processed!=nlake:raise RuntimeError('Lake count mismatch.')
            offset=con.execute(f'SELECT COALESCE(MAX({quoted(meta["pk"])}),0)+1 FROM {quoted(meta["layer"])}').fetchone()[0]
        finally:
            con.close()
        log('[COPY RESERVOIR] Copying monthly reservoir geometries')
        lake.copy_selected(reservoir,resmeta,'reservoir_source','1=1',year,month,True,int(offset))
        counts[9]=nres;totals[9]=float(resvalues[2] or 0)
        times['reservoir_copy_s']=lake.copy_seconds
        times['river_index_finalize_s']=river.finish(nriver)
        times['lake_index_finalize_s']=lake.finish(nlake+nres)
    except Exception:
        river.close_on_error();lake.close_on_error()
        raise
    for temp,final in zip(temps,outputs):temp.replace(final)
    summary={'signature':signature,'outputs':[file_stamp(p) for p in outputs],
             'counts':dict(counts),'areas_km2':dict(totals),'seconds':time.monotonic()-t0,'timing_s':times,
             'engine':'sqlite_blob_copy_lakes_only'}
    marker.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=['final_type,type,object_count,area_km2']+[f'{c},{TYPE_NAMES[c]},{counts[c]},{totals[c]:.8f}' for c in sorted(counts)]
    (folder/'summary.csv').write_text('\n'.join(rows),encoding='utf-8-sig')
    log(f'[DONE] {folder} ; elapsed {summary["seconds"]/60:.1f}min; stage timings (s)={times}')
def data_description():
    return """Monthly river, lake and reservoir classification

Each month contains River.gpkg, LakeAndReservior.gpkg and a summary.csv.
The model predictions and waterbodies retain their original geometries.
The med_large flag denotes lake area >=1 km2. The glacial flag denotes a
lake within 10 km of glacier polygons. The thermo flag identifies lakes
<3 km2 with an interior point in continuous, discontinuous or sporadic
permafrost; isolated permafrost is excluded. Unmatched permafrost is unknown.
These are overlapping environmental attributes, not proof of lake origin.
Reservoirs are classified separately using their monthly mapped geometry.

final_type: 0 river; 1 other lake; 2 medium/large; 3 glacier-associated;
4 thermokarst candidate; 5 medium/large and glacier-associated;
6 medium/large and thermokarst; 7 glacier-associated and thermokarst;
8 all three; 9 reservoir; 99 unresolved lake attributes.
Use the individual flags to count all lakes with a given attribute.
"""



def upgrade_completed(folder, signature):
    marker = folder/'completed.json'
    outputs = [folder/'River.gpkg',folder/'LakeAndReservior.gpkg']
    if not marker.exists() or not all(p.exists() for p in outputs):
        return False
    old = json.loads(marker.read_text(encoding='utf-8'))
    previous = old.get('signature',{})
    if previous.get('version') not in ('1.1-source-manifest',):
        return False
    comparable = dict(signature)
    comparable['version'] = previous['version']
    comparable.pop('med_large_min_area',None)
    if previous != comparable:
        return False
    receipt = folder/'medium_large_upgrade.json'
    journal = json.loads(receipt.read_text(encoding='utf-8')) if receipt.exists() else {'signature':signature,'updated':{}}
    if journal.get('signature') != signature:
        raise RuntimeError(f'Upgrade receipt does not match parameters: {folder}')
    for i,path in enumerate(outputs):
        allowed = journal['updated'].get(path.name) or old['outputs'][i]
        if file_stamp(path) != allowed:
            raise RuntimeError(f'Existing result may have been edited; stopping upgrade: {path}')
    backup = folder/'completed_before_medium_large.json'
    if not backup.exists():
        backup.write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8')
    log(f'[UPGRADE] {folder}; updating medium/large attribute and final code')
    started=time.monotonic()
    for path,layer in zip(outputs,['River','LakeAndReservior']):
        if path.name in journal['updated']:
            continue
        wal=Path(str(path)+'-wal')
        if wal.exists() and wal.stat().st_size:
            raise RuntimeError(f'Close GIS editing and commit changes: {path}')
        con=sqlite3.connect(path,timeout=60)
        try:
            con.execute('BEGIN IMMEDIATE')
            triggers=con.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' AND tbl_name=?",(layer,)).fetchall()
            
            for name,_ in triggers:con.execute(f'DROP TRIGGER {quoted(name)}')
            fields=[r[1] for r in con.execute(f'PRAGMA table_info({quoted(layer)})')]
            if 'med_large' not in fields:con.execute(f'ALTER TABLE {quoted(layer)} ADD COLUMN med_large INTEGER')
            if layer=='River':
                con.execute(f'UPDATE {quoted(layer)} SET med_large=-1,final_type=0')
            else:
                con.execute(f'''UPDATE {quoted(layer)} SET med_large=CASE WHEN is_res=1 THEN -1
                          WHEN area_km2>={MED_LARGE_MIN_AREA_KM2} THEN 1 ELSE 0 END''')
                mask = 'med_large'+ ('+2*glacial' if USE_GLACIAL else '') + '+4*thermo'
                cases = ' '.join(f'WHEN {i} THEN {int(code)}' for i,code in enumerate(COMBINATION_CODES))
                unknown = 'final_type=9 OR thermo<0 OR thermo IS NULL'
                if USE_GLACIAL:unknown+=' OR glacial<0 OR glacial IS NULL'
                con.execute(f'''UPDATE {quoted(layer)} SET final_type=CASE WHEN is_res=1 THEN 9
                         WHEN {unknown} THEN 99 ELSE CASE ({mask}) {cases} ELSE 99 END END''')
            for _,sql in triggers:con.execute(sql)
            con.execute("UPDATE gpkg_contents SET last_change=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE table_name=?",(layer,))
            con.commit()
        except Exception:
            con.rollback();raise
        finally:
            con.close()
        journal['updated'][path.name]=file_stamp(path)
        receipt.write_text(json.dumps(journal,ensure_ascii=False,indent=2),encoding='utf-8')
    counts,areas=Counter(),Counter()
    for path,layer in zip(outputs,['River','LakeAndReservior']):
        with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True) as con:
            for code,n,area in con.execute(f'SELECT final_type,count(*),sum(area_km2) FROM {quoted(layer)} GROUP BY final_type'):
                counts[int(code)]+=int(n);areas[int(code)]+=float(area or 0)
    old.update({'signature':signature,'outputs':[file_stamp(p) for p in outputs],
                'counts':dict(counts),'areas_km2':dict(areas),'attribute_upgrade_seconds':time.monotonic()-started})
    marker.write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=['final_type,type,object_count,area_km2']+[f'{c},{TYPE_NAMES[c]},{counts[c]},{areas[c]:.8f}' for c in sorted(counts)]
    (folder/'summary.csv').write_text('\n'.join(rows),encoding='utf-8-sig')
    log(f'[UPGRADED] {folder}；{(time.monotonic()-started)/60:.1f}min')
    return True


def main():
    if int(shapely.__version__.split('.')[0]) < 2:
        raise RuntimeError('Shapely >=2 is required.')
    log('[START] Copying rivers and reservoirs; classifying lakes')
    months = [(y, m) for y in YEARS for m in range(1, 13)] if PROCESS_ALL_YEARS else [(TEST_YEAR, m) for m in TEST_MONTHS]
    predicted_sources, source_audit = resolve_prediction_sources(months)
    reservoirs = find_reservoirs(months)
    plans = []
    for y, m in months:
        water = predicted_sources[(y, m)]
        if not water.is_file():
            raise FileNotFoundError(water)
        _, info = layer_info(water)
        if not any(f.lower() == LABEL_FIELD.lower() for f in info['fields']):
            raise ValueError(f'{water} has no prediction label field.')
        plans.append((y, m, water, reservoirs[(y, m)]))
    _, info = layer_info(plans[0][2])
    target = CRS.from_user_input(info['crs'])
    if not target.is_projected or any(abs(a.unit_conversion_factor-1)>1e-9 for a in target.axis_info[:2]):
        raise ValueError('Waterbody CRS must be projected in metres.')
    ref_stamp = [file_stamp(PERMAFROST)] + ([file_stamp(GLACIER)] if USE_GLACIAL else [])
    if BOUNDARY.exists():
        ref_stamp += [file_stamp(BOUNDARY.with_suffix(s)) for s in ['.shp', '.shx', '.dbf', '.prj'] if BOUNDARY.with_suffix(s).exists()]
    params = {'version': VERSION, 'target': target.to_wkt(), 'references': ref_stamp,
              'glacial': USE_GLACIAL, 'distance_m': GLACIER_DISTANCE_M, 'max_area': THERMO_MAX_AREA_KM2,
              'include_isolated': INCLUDE_ISOLATED, 'spatial_index': SPATIAL_INDEX,
              'med_large_min_area': MED_LARGE_MIN_AREA_KM2}
    
    pending = []
    for y, m, water, reservoir in plans:
        sig = dict(params, inputs=[file_stamp(water), file_stamp(reservoir)])
        folder = OUTPUT_ROOT / str(y) / f'{m:02d}'
        marker = folder / 'completed.json'
        outs = [folder / 'River.gpkg', folder / 'LakeAndReservior.gpkg']
        if not REBUILD and marker.exists() and all(p.exists() for p in outs):
            old = json.loads(marker.read_text(encoding='utf-8'))
            if old.get('signature') == sig and old.get('outputs') == [file_stamp(p) for p in outs]:
                log(f'[SKIP] {y}-{m:02d}')
                continue
        if UPGRADE_COMPLETED and not REBUILD and upgrade_completed(folder, sig):
            continue
        pending.append((y, m, water, reservoir, sig))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / 'prediction_sources.json').write_text(json.dumps(source_audit, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUTPUT_ROOT / 'data_description.txt').write_text(data_description(), encoding='utf-8-sig')
    if not pending:
        log('[FINISHED] All months already complete.')
        return
    refs = load_references(target)
    for y, m, water, reservoir, sig in pending:
        process_month(y, m, water, reservoir, target, refs, sig)
    log('[FINISHED] '+str(OUTPUT_ROOT))

if __name__ == '__main__':
    try:
        main()
    except Exception:
        log('[ERROR] Failed months are not marked complete; completed months are retained.')
        raise
