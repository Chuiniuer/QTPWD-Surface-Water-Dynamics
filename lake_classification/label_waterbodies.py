# -*- coding: utf-8 -*-
"""QTP monthly candidate labels + shape attributes. Python >=3.9, Shapely >=2.
Run directly. Read-only inputs; no reprojection, no reference reconstruction.
Dependencies: numpy pandas geopandas pyogrio rasterio shapely pyproj.
"""
from pathlib import Path
import os, sys, json, time, hashlib, logging, math, traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------- Only edit this configuration ----------------
PROCESS_ALL_YEARS = False   # False: 2018 only; True: 2016--2025, completed months skipped
YEARS = [2018]
INPUT_ROOT = Path(r'E:\phd_l3\QTP_reservoir_separation\water_without_reservoirs')
BASE = Path(r'K:\Monthly surface water dynamics TIB\supervised classification\step5_lake_classification')
REFERENCE_ROOT = BASE / 'reference_data'
# Leave None to use reference_data/active_run.json from the previous preparation script.
RIVER_MASK = None
LAKE_MASK = None
OUTPUT_ROOT = BASE / 'monthly_lake_river_attributes'
WORKERS = 2                 # 1 for slow/network disks; do not start 12 workers
BATCH_SIZE = 4000
TILE_SIZE = 1024
MIN_LABEL_AREA_KM2 = 0.0    # candidate-label cutoff, NOT deletion or size-class definition
MATCH_MIN = 0.80
OTHER_MAX = 0.05
VALID_MIN = 0.99
REBUILD_CHANGED = False    # changed inputs/rules: stop unless explicitly True
# -------------------------------------------------------------
VERSION = '1.0.0'
LOG = logging.getLogger('QTP')


def setup_log(path=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if path:
        handlers.append(logging.FileHandler(path, encoding='utf-8'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(process)d %(message)s',
                        handlers=handlers, force=True)


def stamp(path):
    p = Path(path); s = p.stat()
    return {'path': str(p.resolve()), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns}


def atomic_json(path, data):
    p = Path(path); tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def references():
    if (RIVER_MASK is None) != (LAKE_MASK is None):
        raise ValueError('Set BOTH RIVER_MASK and LAKE_MASK, or leave both None.')
    if RIVER_MASK is not None:
        paths = [Path(RIVER_MASK), Path(LAKE_MASK)]
    else:
        active = REFERENCE_ROOT / 'active_run.json'
        if not active.exists():
            raise FileNotFoundError(f'{active}: set RIVER_MASK and LAKE_MASK to the prepared files.')
        run = Path(load_json(active)['run'])
        paths = [run / 'Combined_river_mask_10m.tif', run / 'HMA_lake_support_10m.tif']
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f'Prepared reference missing: {p}; no reconstruction started.')
    return paths


def validate_reference(r, l, crs):
    from pyproj import CRS
    target = CRS.from_user_input(crs)
    method = target.coordinate_operation.method_name if target.coordinate_operation else ''
    if not target.is_projected or 'albers' not in method.lower() or any(
        abs(a.unit_conversion_factor - 1) > 1e-9 for a in target.axis_info[:2]):
        raise ValueError('Monthly CRS must be Albers with metre axes.')
    for ds in (r, l):
        if not target.equals(CRS.from_user_input(ds.crs), ignore_axis_order=True):
            raise ValueError('Reference/monthly CRS mismatch; automatic reprojection disabled.')
        if ds.count != 1 or ds.nodata != 255:
            raise ValueError('Expected prepared single-band 0/1 reference with NoData=255.')
    if r.transform != l.transform or r.shape != l.shape:
        raise ValueError('River/lake references must have identical grids.')
    t = r.transform
    if abs(t.a-10)>1e-8 or abs(t.e+10)>1e-8 or t.b or t.d:
        raise ValueError('Expected north-up 10 m reference grid.')


def reference_counts(geoms, good, river, lake):
    """Shared tile burns; count object pixels once, source union already encoded.
    Overlapping object pixels trigger independent burns in that tile (no overwrite loss).
    All tiles, including outside reference extent, contribute to the denominator.
    """
    import numpy as np
    import shapely
    import rasterio
    from rasterio.features import rasterize
    from rasterio.enums import MergeAlg
    from rasterio.windows import Window
    n = len(geoms)
    counts = np.zeros((n, 5), dtype='int64') # object, common-valid, river, lake, conflict
    bins = defaultdict(list)
    t = river.transform
    for i in np.flatnonzero(good):
        x0, y0, x1, y1 = shapely.bounds(geoms[i])
        c0, c1 = math.floor((x0-t.c)/t.a), math.ceil((x1-t.c)/t.a)
        r0, r1 = math.floor((t.f-y1)/(-t.e)), math.ceil((t.f-y0)/(-t.e))
        for rr in range(r0//TILE_SIZE, max(r0, r1-1)//TILE_SIZE+1):
            for cc in range(c0//TILE_SIZE, max(c0, c1-1)//TILE_SIZE+1):
                bins[(rr, cc)].append(int(i))
    last = time.monotonic()
    for k, ((rr, cc), ids) in enumerate(sorted(bins.items()), 1):
        window = Window(cc*TILE_SIZE, rr*TILE_SIZE, TILE_SIZE, TILE_SIZE)
        tf = rasterio.windows.transform(window, t)
        shape = (TILE_SIZE, TILE_SIZE)
        labels = rasterize([(geoms[i], j+1) for j,i in enumerate(ids)], out_shape=shape,
                           transform=tf, dtype='int32', all_touched=False)
        if not labels.any():
            continue
        ar = river.read(1, window=window, boundless=True, fill_value=255)
        al = lake.read(1, window=window, boundless=True, fill_value=255)
        if not np.isin(ar, [0,1,255]).all() or not np.isin(al, [0,1,255]).all():
            raise ValueError('Reference has values other than 0, 1, 255.')
        valid = (ar != 255) & (al != 255)
        flags = [valid, (ar==1)&valid, (al==1)&valid, (ar==1)&(al==1)&valid]
        overlap = False
        if len(ids)>1:
            multiplicity = rasterize([(geoms[i],1) for i in ids], out_shape=shape,
                transform=tf, dtype='uint32', merge_alg=MergeAlg.add, all_touched=False)
            overlap = bool((multiplicity>1).any())
        if overlap:
            for i in ids:
                mask = rasterize([(geoms[i],1)], out_shape=shape, transform=tf,
                                 dtype='uint8', all_touched=False).astype(bool)
                counts[i,0] += np.count_nonzero(mask)
                for j, flag in enumerate(flags,1):
                    counts[i,j] += np.count_nonzero(mask & flag)
        else:
            counts[ids,0] += np.bincount(labels.ravel(), minlength=len(ids)+1)[1:]
            for j,flag in enumerate(flags,1):
                counts[ids,j] += np.bincount(labels[flag], minlength=len(ids)+1)[1:]
        if time.monotonic()-last > 20:
            LOG.info('[COVERAGE] batch tiles %d/%d', k, len(bins)); last=time.monotonic()
    return counts


def area_values(df, geoms, good):
    """Reuse q_area from the reservoir script; verify a deterministic sample per batch."""
    import numpy as np
    import pandas as pd
    import shapely
    area = np.full(len(df), np.nan)
    if 'q_area' in df:
        area = pd.to_numeric(df['q_area'], errors='coerce').to_numpy(dtype=float)*1e6
    use = good & np.isfinite(area) & (area>0)
    ix = np.flatnonzero(use)
    if len(ix):
        sample = ix[np.unique(np.linspace(0,len(ix)-1,min(64,len(ix))).astype(int))]
        real = shapely.area(geoms[sample])
        if not np.allclose(area[sample],real,rtol=1e-6,atol=1e-4):
            LOG.warning('[AREA] q_area sample mismatch; recomputing this batch.')
            use[:] = False
    need = good & ~use
    area[need] = shapely.area(geoms[need])
    area[~good] = np.nan
    return area, int(np.count_nonzero(use))


def label_objects(area, counts, good, reservoir_review):
    import numpy as np
    total, valid = counts[:,0], counts[:,1]
    fractions = np.full((len(area),4), np.nan)
    have = good & (total>0)
    fractions[have,0] = valid[have]/total[have]
    observed = have & (valid>0)
    fractions[observed,1:] = counts[observed,2:]/total[observed,None]
    vf, rf, lf, cf = fractions.T
    label = np.zeros(len(area), dtype='int16')
    status = np.full(len(area),'unmatched',dtype=object)
    status[(rf>0)|(lf>0)] = 'weak_match'
    status[(rf>OTHER_MAX)&(lf>OTHER_MAX)] = 'conflict'
    status[area/1e6<MIN_LABEL_AREA_KM2] = 'below_seed_area'
    eligible = good & (vf>=VALID_MIN) & (area/1e6>=MIN_LABEL_AREA_KM2) & (reservoir_review==0)
    take_lake = eligible & (lf>=MATCH_MIN) & (rf<=OTHER_MAX)
    take_river = eligible & (rf>=MATCH_MIN) & (lf<=OTHER_MAX)
    label[take_lake]=1; status[take_lake]='lake_candidate'
    label[take_river]=2; status[take_river]='river_candidate'
    status[have & (vf<VALID_MIN)]='reference_nodata'
    status[good & (total==0)]='zero_pixel'
    status[reservoir_review==1]='reservoir_review'
    status[reservoir_review==2]='reservoir_unknown'
    status[~good]='invalid_geometry'
    return fractions, label, status


def shape_metrics(geoms, good, area):
    import numpy as np
    import shapely
    names = ['perim_m','outer_m','elong','compact','rect_fill','solidity','width_m','hole_free_c','hole_frac']
    out = {k:np.full(len(geoms),np.nan) for k in names}
    out['n_parts']=np.zeros(len(geoms),dtype='int32')
    out['n_holes']=np.zeros(len(geoms),dtype='int32')
    ix=np.flatnonzero(good)
    if not len(ix): return out
    gg=geoms[ix]; a=area[ix]
    p=shapely.length(gg)
    parts, parent=shapely.get_parts(gg, return_index=True)
    rings=shapely.get_exterior_ring(parts)
    p0=np.bincount(parent,weights=shapely.length(rings),minlength=len(ix))
    a0=np.bincount(parent,weights=shapely.area(shapely.polygons(rings)),minlength=len(ix))
    hull=shapely.convex_hull(gg)
    rect=shapely.minimum_rotated_rectangle(hull) # reuse hull; no second hull construction requested
    xy=shapely.get_coordinates(rect).reshape(-1,5,2)
    edges=np.linalg.norm(np.diff(xy,axis=1),axis=2)
    length=edges.max(axis=1); width=edges.min(axis=1)
    out['perim_m'][ix]=p; out['outer_m'][ix]=p0
    out['elong'][ix]=length/width
    out['compact'][ix]=4*np.pi*a/(p*p)
    out['rect_fill'][ix]=a/(length*width)
    out['solidity'][ix]=a/shapely.area(hull)
    out['width_m'][ix]=2*a/p0
    out['hole_free_c'][ix]=4*np.pi*a0/(p0*p0)
    out['hole_frac'][ix]=(a0-a)/a0
    out['n_parts'][ix]=np.bincount(parent,minlength=len(ix))
    out['n_holes'][ix]=np.bincount(parent,weights=shapely.get_num_interior_rings(parts),minlength=len(ix)).astype('int32')
    return out


def rules():
    return dict(version=VERSION, min_area_km2=MIN_LABEL_AREA_KM2,
                match_min=MATCH_MIN,other_max=OTHER_MAX,valid_min=VALID_MIN,
                tile_size=TILE_SIZE,batch_size=BATCH_SIZE)


def run_month(job):
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import pyogrio
    import rasterio
    import shapely
    src, dst, refs, year, month = job
    src,dst=Path(src),Path(dst)
    dst.parent.mkdir(parents=True,exist_ok=True)
    setup_log(dst.with_suffix('.log'))
    params={'input':stamp(src),'references':[stamp(p) for p in refs],'rules':rules()}
    key=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()
    done=dst.with_suffix('.done.json')
    if done.exists() and dst.exists():
        old=load_json(done)
        if old.get('key')==key and old.get('output')==stamp(dst):
            LOG.info('[SKIP] %s complete',src.name); return old['summary']
        if not REBUILD_CHANGED:
            raise RuntimeError(f'{dst}: inputs/rules/output changed; set REBUILD_CHANGED=True to replace.')
    elif dst.exists() and not REBUILD_CHANGED:
        raise RuntimeError(f'{dst}: output has no completion record; inspect or set REBUILD_CHANGED=True.')
    temp=dst.with_name(dst.stem+'.working.gpkg')
    for p in [temp,Path(str(temp)+'-wal'),Path(str(temp)+'-shm')]:
        if p.exists():p.unlink()
    layers=pyogrio.list_layers(src)
    if len(layers)!=1: raise ValueError(f'{src}: expected exactly one monthly layer; got {layers.tolist()}')
    layer=str(layers[0][0]); info=pyogrio.read_info(src,layer=layer)
    fid=info.get('fid_column')
    if not fid: raise ValueError('GPKG primary FID column not exposed; cannot keyset-stream safely.')
    fields=[f for f in ['q_area','q_ovfrac','q_status'] if f in info['fields']]
    def quote(s):return '"'+s.replace('"','""')+'"'
    # Explicit columns avoid copying original unused attributes; geometry retained.
    geom_col=info.get('geometry_name')
    # GDAL metadata may omit geometry_name: read it from standard GPKG metadata.
    import sqlite3
    with sqlite3.connect(f'file:{src.as_posix()}?mode=ro',uri=True) as db:
        geom_col=db.execute('SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?',(layer,)).fetchone()[0]
    select=', '.join(quote(x) for x in [fid,geom_col]+fields)
    last_fid=None; first=True; count=0; reused=0; stats=Counter(); start=time.monotonic()
    LOG.info('[START] %s; features=%s',src.name,info['features'])
    with rasterio.Env(GDAL_CACHEMAX=128*1024*1024), rasterio.open(refs[0]) as river, rasterio.open(refs[1]) as lake:
        validate_reference(river,lake,info['crs'])
        while True:
            where='' if last_fid is None else f' WHERE {quote(fid)} > {last_fid}'
            sql=f'SELECT {select} FROM {quote(layer)}{where} ORDER BY {quote(fid)} LIMIT {BATCH_SIZE}'
            df=pyogrio.read_dataframe(src,sql=sql,sql_dialect='SQLITE',fid_as_index=True)
            if df.empty:break
            last_fid=int(df.index[-1])
            gg=df.geometry.to_numpy()
            good=shapely.is_valid(gg)&~shapely.is_empty(gg)&np.isin(shapely.get_type_id(gg),[3,6])
            a,n_reuse=area_values(df,gg,good); good &= np.isfinite(a)&(a>0); reused+=n_reuse
            reservoir=np.full(len(df),2,dtype='int16')
            if 'q_ovfrac' in df:
                ov=pd.to_numeric(df['q_ovfrac'],errors='coerce').to_numpy(float)
                reservoir[np.isfinite(ov)&(ov==0)]=0
                reservoir[np.isfinite(ov)&(ov>0)]=1
            if 'q_status' in df:
                reservoir[df['q_status'].fillna('').str.contains('partial|reservoir',case=False).to_numpy()]=1
            t0=time.monotonic()
            counts=reference_counts(gg,good,river,lake)
            fractions,labels,status=label_objects(a,counts,good,reservoir)
            coverage_s=time.monotonic()-t0
            # Labels assigned BEFORE morphology; shape does not determine training labels.
            t0=time.monotonic(); metrics=shape_metrics(gg,good,a); metric_s=time.monotonic()-t0
            values={'obj_id':df.index.to_numpy(dtype='int64'),'year':np.full(len(df),year,dtype='int16'),
                    'month':np.full(len(df),month,dtype='int16'),'area_km2':a/1e6,
                    'label':labels,'label_state':status,'res_review':reservoir,
                    'ref_valid':fractions[:,0],'river_frac':fractions[:,1],
                    'lake_frac':fractions[:,2],'conflict_frac':fractions[:,3],**metrics}
            out=gpd.GeoDataFrame(values,geometry=gg,crs=df.crs)
            pyogrio.write_dataframe(out,temp,layer='waterbodies',driver='GPKG',append=not first,
                                     layer_options={'SPATIAL_INDEX':'YES'} if first else None)
            first=False; count+=len(df); stats.update(status.tolist())
            LOG.info('[BATCH] %s %s/%s; reference %.1fs, metrics %.1fs, elapsed %.1fmin',
                     src.name,count,info['features'],coverage_s,metric_s,(time.monotonic()-start)/60)
    if first:
        raise RuntimeError(f'{src}: no records; no completion marker written.')
    summary={'year':year,'month':month,'features':count,'q_area_reused':reused,
             'states':dict(stats),'elapsed_seconds':time.monotonic()-start}
    # Verify row count without loading output geometry. No large-file hash pass.
    oi=pyogrio.read_info(temp,layer='waterbodies')
    if oi['features']!=count:raise RuntimeError('Output row count mismatch.')
    os.replace(temp,dst)
    atomic_json(done,{'key':key,'parameters':params,'output':stamp(dst),'summary':summary,
                      'versions':{'shapely':shapely.__version__,'gdal':pyogrio.__gdal_version_string__,
                                  'rasterio':rasterio.__version__}})
    LOG.info('[DONE] %s',dst)
    return summary


README = r'''# Monthly lake and river candidate labels

Inputs are monthly reservoir-excluded water polygons and aligned 10 m river and
lake reference masks. The candidate label is 0 for unlabelled, 1 for lake and
2 for river. Thresholds are recorded with the output. The polygons are retained
and geometric attributes are computed for subsequent classification.

Reference masks represent reference evidence, not definitive ground truth.
Per-month completion markers and input signatures control reruns.
'''


def main():
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    setup_log(OUTPUT_ROOT/'run.log')
    LOG.info('[START] Monthly candidate labels and shape metrics; script=%s',Path(__file__).resolve())
    import shapely
    if int(shapely.__version__.split('.')[0])<2:raise RuntimeError('Shapely >=2 required.')
    if not (0<=OTHER_MAX<MATCH_MIN<=1 and 0<VALID_MIN<=1 and MIN_LABEL_AREA_KM2>=0):
        raise ValueError('Invalid classification thresholds.')
    refs=references()
    LOG.info('[REFERENCES] river=%s; lake=%s',*refs)
    (OUTPUT_ROOT/'README_数据说明.md').write_text(README+'\n\nRun parameters:\n```json\n'+json.dumps(rules(),ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8-sig')
    years=list(range(2016,2026)) if PROCESS_ALL_YEARS else YEARS
    jobs=[]; missing=[]
    for year in years:
        for month in range(1,13):
            src=INPUT_ROOT/str(year)/f'WB_{year}_{month:02d}.gpkg'
            dst=OUTPUT_ROOT/str(year)/f'WB_{year}_{month:02d}_attributes.gpkg'
            if src.is_file():jobs.append((str(src),str(dst),[str(p) for p in refs],year,month))
            else:missing.append(str(src))
    for p in missing:LOG.warning('[MISSING] %s',p)
    if not jobs:raise FileNotFoundError('No monthly inputs found.')
    results=[]; errors=[]
    if WORKERS<=1:
        for job in jobs:
            try:results.append(run_month(job))
            except Exception as exc:
                LOG.exception('[FAILED] %s',job[0]); errors.append({'file':job[0],'error':str(exc)})
    else:
        with ProcessPoolExecutor(max_workers=min(WORKERS,len(jobs))) as pool:
            futures={pool.submit(run_month,job):job for job in jobs}
            for future in as_completed(futures):
                try:results.append(future.result())
                except Exception as exc:
                    LOG.exception('[FAILED] %s',futures[future][0])
                    errors.append({'file':futures[future][0],'error':str(exc)})
    atomic_json(OUTPUT_ROOT/'run_summary.json',{'months':sorted(results,key=lambda x:(x['year'],x['month'])),
                'missing':missing,'errors':errors})
    if errors or missing:raise RuntimeError(f'Incomplete: {len(errors)} failed, {len(missing)} missing; see logs. Completed months retained.')
    LOG.info('[FINISHED] %s months complete/skipped. Output: %s',len(results),OUTPUT_ROOT)


if __name__=='__main__':
    from multiprocessing import freeze_support
    freeze_support()
    try:main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
