"""Calculate surface water area trends, seasonality and extremes."""
from pathlib import Path
import argparse
import csv
import json
import platform
from datetime import datetime
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

BASE = Path(r'K:\Monthly surface water dynamics TIB\supervised classification\step4_analysis')
DATES = pd.date_range('2016-01-01', '2025-12-01', freq='MS')
COLS = dict(Total='Total_surface_water_area_km2', Small='Small_waterbody_area_km2',
            Medium='Medium_waterbody_area_km2', Large='Large_waterbody_area_km2')
CONFIG = dict(period=12, seasonal=7, trend=23, low_pass=13, robust=True)
FIELDS = ['mean_km2','beta_yr','chg10_km2','sig_km2','delta_km2','cseason',
          's_sd_km2','r_sd_km2','low_mag','high_mag','low_dev','high_dev',
          'i_low','i_high','low_n','high_n','low_date','high_date']

def analyse(y):
    y = np.asarray(y, dtype=float)
    if y.shape != (120,) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Analysis requires 120 finite nonnegative monthly areas')
    fit = STL(y, **CONFIG).fit(inner_iter=2, outer_iter=15)
    if not np.isfinite(np.array([fit.trend,fit.seasonal,fit.resid])).all():
        raise ValueError('Non-finite STL components')
    if not np.allclose(y, fit.trend+fit.seasonal+fit.resid, atol=1e-8, rtol=1e-10):
        raise ValueError('STL reconstruction failed')
    beta = float(np.polyfit(np.arange(120)/12, fit.trend, 1)[0])
    annual = y.reshape(10,12)
    sigma = float(annual.std(axis=1,ddof=0).mean())
    delta = float(annual.mean(axis=1).std(ddof=0))
    dev = annual-annual.mean(axis=1,keepdims=True)
    qlo,qhi = np.quantile(dev,[.1,.9],axis=0,method='linear')
    lo,hi = (dev<qlo).ravel(),(dev>qhi).ravel()
    d = dev.ravel()
    result = dict(mean_km2=float(y.mean()),beta_yr=beta,chg10_km2=10*beta,
                  sig_km2=sigma,delta_km2=delta,
                  cseason=100*sigma/(sigma+delta) if sigma+delta>0 else None,
                  s_sd_km2=float(fit.seasonal.std()),r_sd_km2=float(fit.resid.std()))
    for label,mask,sign in [('low',lo,-1),('high',hi,1)]:
        ids = np.flatnonzero(mask)
        result[label+'_n'] = len(ids)
        k = int(ids[np.argmax(np.abs(d[ids]))]) if len(ids) else None
        mag = float(abs(d[k])) if k is not None else None
        denominator = mag+10*abs(beta)+sigma if mag is not None else 0
        result[label+'_mag'] = mag
        result[label+'_dev'] = float(d[k]) if k is not None else None
        result[label+'_date'] = DATES[k].strftime('%Y-%m') if k is not None else ''
        result['i_'+label] = sign*100*mag/denominator if denominator>0 else None
    components = pd.DataFrame(dict(Date=DATES.strftime('%Y-%m'),Area_km2=y,
        Trend_km2=fit.trend,Seasonal_km2=fit.seasonal,Residual_km2=fit.resid,
        AnnualDeviation_km2=d,LowThreshold_km2=np.tile(qlo,10),
        HighThreshold_km2=np.tile(qhi,10),LowEvent=lo,HighEvent=hi))
    return result,components

def read_table(path, watershed=False):
    df = pd.read_csv(path,encoding='utf-8-sig')
    needed = ['Year','Month']+list(COLS.values())+(['Name'] if watershed else [])
    if not set(needed).issubset(df):
        raise ValueError(f'{path}: missing columns {set(needed)-set(df)}')
    for c in ['Year','Month']:
        v = pd.to_numeric(df[c],errors='raise')
        if v.isna().any() or (v!=np.floor(v)).any():
            raise ValueError(f'{path}: invalid {c}')
        df[c] = v.astype(int)
    if watershed and (df.Name.isna().any() or df.Name.astype(str).str.strip().eq('').any()):
        raise ValueError('Missing watershed name')
    df['Date'] = pd.to_datetime(dict(year=df.Year,month=df.Month,day=1))
    for c in COLS.values():
        df[c] = pd.to_numeric(df[c],errors='raise')
    groups = list(df.groupby('Name',sort=True)) if watershed else [('QTP',df)]
    if not groups:
        raise ValueError(f'Empty table: {path}')
    checked=[]
    for name,g in groups:
        g=g.sort_values('Date')
        if not pd.DatetimeIndex(g.Date).equals(DATES):
            raise ValueError(f'{name}: expected exactly 2016-01 through 2025-12')
        vals=g[list(COLS.values())].to_numpy(float)
        if not np.isfinite(vals).all() or (vals<0).any():
            raise ValueError(f'{name}: invalid areas')
        if not np.allclose(vals[:,0],vals[:,1:].sum(axis=1),atol=1e-7,rtol=1e-10):
            raise ValueError(f'{name}: size-class areas do not sum to total')
        checked.append((str(name),g))
    return checked

def grids(args,out):
    import rasterio
    import shapefile
    paths=[args.grid_dir/str(d.year)/f'QTP_water_area_25km_{d.year}_{d.month:02d}.tif' for d in DATES]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing rasters:\n'+'\n'.join(missing))
    with rasterio.open(paths[0]) as src:
        crs,transform,shape=src.crs,src.transform,src.shape
        if crs is None or not crs.is_projected:
            raise ValueError('Projected CRS required')
        if crs.linear_units.lower() not in ('metre','meter','metres','meters'):
            raise ValueError('Expected Albers CRS in metres')
        if not (np.isclose(np.hypot(transform.a,transform.d),25000) and
                np.isclose(np.hypot(transform.b,transform.e),25000)):
            raise ValueError('Raster pixels must be 25000 metres')
    values=[]; footprint=np.zeros(shape,bool); valid_count=np.zeros(shape,np.int16)
    metadata=[]
    for p in paths:
        with rasterio.open(p) as src:
            if src.crs!=crs or src.transform!=transform or src.shape!=shape:
                raise ValueError(f'Misaligned raster: {p}')
            if src.count<2:
                raise ValueError(f'{p}: requires Band 1 area km2 and Band 2 valid fraction; upload sample to adapt single-band data')
            area=src.read(1,masked=True).astype(float).filled(np.nan)
            vf=src.read(2,masked=True).astype(float).filled(np.nan)
            if np.any(np.isfinite(vf)&((vf<0)|(vf>1))):
                raise ValueError(f'{p}: Band 2 must be a fraction in [0,1]')
            if np.any(np.isfinite(area)&((area<0)|(area>625.001))):
                raise ValueError(f'{p}: Band 1 area outside [0,625] km2; verify units')
            footprint |= np.isfinite(area)|np.isfinite(vf)
            good=np.isfinite(area)&np.isfinite(vf)&(vf>=args.min_valid)
            valid_count+=good
            values.append(np.where(good,area,np.nan))
            metadata.append(dict(path=str(p),size=p.stat().st_size,nodata=src.nodata,
                                 descriptions=src.descriptions))
    cube=np.stack(values)
    rows,cols=np.where(footprint)
    if not len(rows):
        raise ValueError('No covered raster cells')
    shp=out/'QTP_25km_extremes_centres.shp'
    schema=[('grid_id','N',12,0),('row','N',8,0),('col','N',8,0),('n_valid','N',5,0),
            ('status','C',16,0),('map_ok','N',1,0)]
    schema += [(f,'C',7,0) if f.endswith('_date') else
               (f,'N',5,0) if f.endswith('_n') else (f,'F',24,10) for f in FIELDS]
    counts={}
    trends=[]; seasons=[]; residuals=[]; complete_ids=[]
    with shapefile.Writer(str(shp),shapeType=shapefile.POINT,encoding='utf-8') as writer, \
         (out/'Grid_summary.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        for f in schema: writer.field(*f)
        names=[f[0] for f in schema]
        csvwriter=csv.DictWriter(handle,fieldnames=names);csvwriter.writeheader()
        for n,(r,c) in enumerate(zip(rows,cols)):
            gid=int(r*shape[1]+c+1)
            record=dict.fromkeys(names)
            record.update(grid_id=gid,row=int(r),col=int(c),n_valid=int(valid_count[r,c]),map_ok=0)
            y=cube[:,r,c]
            if np.isfinite(y).all():
                result,comp=analyse(y)
                record.update(result)
                record['status']='ZERO_WATER' if np.all(y==0) else 'OK'
                record['map_ok']=int(y.mean()>=args.map_min_area)
                complete_ids.append(gid)
                trends.append(comp.Trend_km2.to_numpy());seasons.append(comp.Seasonal_km2.to_numpy())
                residuals.append(comp.Residual_km2.to_numpy())
            else:
                record['status']='INCOMPLETE'
            counts[record['status']]=counts.get(record['status'],0)+1
            x,ycoord=transform*(float(c)+.5,float(r)+.5)
            writer.point(x,ycoord);writer.record(*[record[k] for k in names]);csvwriter.writerow(record)
            if (n+1)%500==0: print(f'Grid {n+1}/{len(rows)}',flush=True)
    shp.with_suffix('.prj').write_text(crs.to_wkt(version='WKT1_ESRI'),encoding='utf-8')
    shp.with_suffix('.cpg').write_text('UTF-8',encoding='ascii')
    np.savez_compressed(out/'Grid_STL_components.npz',grid_id=np.array(complete_ids,dtype=np.int64),
        dates=DATES.strftime('%Y-%m').to_numpy(dtype='U7'),
        trend=np.array(trends).reshape(-1,120),seasonal=np.array(seasons).reshape(-1,120),
        residual=np.array(residuals).reshape(-1,120))
    return dict(counts=counts,rasters=metadata,crs=crs.to_wkt(),transform=list(transform))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--grid-dir',type=Path)
    parser.add_argument('--stats-dir',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--min-valid',type=float,default=.90)
    parser.add_argument('--map-min-area',type=float,default=.1)
    args=parser.parse_args()
    if not 0<=args.min_valid<=1 or args.map_min_area<0:
        parser.error('Invalid quality or mapping threshold')
    args.grid_dir=args.grid_dir or args.base/'QTP_water_area_25km_QTP_Albers'/'monthly'
    args.stats_dir=args.stats_dir or args.base/'QTP_waterbody_statistics_0p1'
    out=args.output_dir or args.base/('QTP_trend_extremes_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    q=read_table(args.stats_dir/'QTP_monthly_waterbody_statistics.csv')
    w=read_table(args.stats_dir/'Watershed_monthly_waterbody_statistics.csv',True)
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='RUNNING',python=platform.python_version(),stl=CONFIG,
                  inner_iter=2,outer_iter=15,min_valid=args.min_valid,
                  map_min_area=args.map_min_area,period='2016-01/2025-12',
                  grid_dir=str(args.grid_dir),stats_dir=str(args.stats_dir),
                  missing_policy='complete 120 months only; no interpolation',
                  index_trend_magnitude='10 * abs(beta_km2_per_year)')
    mpath=out/'run_manifest.json'
    try:
        results=[];components=[]
        for scale,groups in [('QTP',q),('Watershed',w)]:
            for region,frame in groups:
                for label,column in COLS.items():
                    result,comp=analyse(frame[column].to_numpy(float))
                    results.append(dict(Scale=scale,Region=region,SizeClass=label,**result))
                    comp.insert(0,'SizeClass',label);comp.insert(0,'Region',region);comp.insert(0,'Scale',scale)
                    components.append(comp)
        pd.DataFrame(results).to_csv(out/'Regional_summary.csv',index=False,encoding='utf-8-sig')
        pd.concat(components).to_csv(out/'Regional_monthly_components.csv',index=False,encoding='utf-8-sig')
        manifest['grid']=grids(args,out)
        import statsmodels, rasterio, shapefile
        manifest['versions']=dict(numpy=np.__version__,pandas=pd.__version__,
            statsmodels=statsmodels.__version__,rasterio=rasterio.__version__,pyshp=shapefile.__version__)
        manifest['status']='COMPLETE'
    except Exception as exc:
        manifest.update(status='FAILED',error=str(exc))
        raise
    finally:
        mpath.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(f'Complete: QTP, {len(w)} watersheds and grids. Output: {out}')

if __name__=='__main__':
    main()
