# -*- coding: utf-8 -*-
"""Lake type distribution and seasonality, Figure 5."""
from pathlib import Path
from datetime import datetime
import json, hashlib, sqlite3, struct, os, time, calendar
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm, rankdata
from statsmodels.tsa.seasonal import STL
import rasterio
from rasterio.features import rasterize, geometry_mask
from rasterio.transform import from_bounds
import fiona
import geopandas as gpd
import shapely
from pyproj import CRS


INPUT_ROOT = Path(r'E:\phd_l3\QTP_monthly_water_types')
OUTPUT_ROOT = Path(r'D:\phdl2\QTP surface water dynamics\code availability\test_output')
CACHE_ROOT = Path(r'K:\Monthly surface water dynamics TIB\supervised classification\step4_analysis\QTP_lake_type_figures\cache')
BOUNDARY = Path(r'D:\phdl2\QTP surface water dynamics\regions\QTP.shp')
BASINS = Path(r'D:\phdl2\QTP surface water dynamics\regions\QTP_watershed.shp')
MAP_YEAR, MAP_MONTH = 2025, 9
YEARS = range(2016, 2026)
DISPLAY_WIDTH = 6500
READ_BATCH = 20000
REFRESH_MAP = False
REFRESH_ATTRIBUTE_STATS = False

DPI = 300

CODES = list(range(1,7))
PLOT_ORDER = [6,2,3,4,5,1]
NAMES = ['Other lakes','Medium and large lakes','Glacial lakes','Thermokarst lakes','Reservoirs','Total']
COLORS = ['#B600C6','#216FE8','#08BCEB','#FF5139','#884665','#777777']
# Attributes can overlap; the map uses reservoir > glacial > thermokarst > medium/large > other.

MAP_NAMES = ['Other lakes','Medium and large lakes','Thermokarst lakes','Glacial lakes','Reservoirs']
MAP_COLORS = [COLORS[0],COLORS[1],COLORS[3],COLORS[2],COLORS[4]]
VERSION = 'lake-types-attribute-v2'
SIZE_EDGES = np.array([0,.001,.01,.1,1,10,100,np.inf])
SIZE_NAMES = ['<0.001','0.001–<0.01','0.01–<0.1','0.1–<1','1–<10','10–<100','≥100']

def log(s):
    print(f'{datetime.now():%H:%M:%S} {s}', flush=True)


def stamp(p):
    p=Path(p);s=p.stat();return [str(p.resolve()),s.st_size,s.st_mtime_ns]


def key(paths, settings):
    return hashlib.sha256(json.dumps([list(map(stamp,paths)),settings],sort_keys=True).encode()).hexdigest()[:20]


def save_json(p, value):
    t=p.with_suffix('.partial.json');t.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)


def q(s):
    return '"'+s.replace('"','""')+'"'


def table_info(path):
    wal=Path(str(path)+'-wal')
    if wal.exists() and wal.stat().st_size:raise RuntimeError(f'Close GIS editing and commit changes before plotting: {path}')
    con=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    rows=con.execute('SELECT table_name,column_name,srs_id FROM gpkg_geometry_columns').fetchall()
    preferred=[r for r in rows if r[0]=='LakeAndReservior']
    if len(preferred)==1: row=preferred[0]
    elif len(rows)==1: row=rows[0]
    else:con.close();raise ValueError(f'Ambiguous vector layer: {path}')
    fields={r[1] for r in con.execute(f'PRAGMA table_info({q(row[0])})')}
    if not {'final_type','area_km2','med_large'}<=fields:
        con.close();raise ValueError(f'Expected latest v2 classification fields: {path}')
    crs=con.execute('SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id=?',(row[2],)).fetchone()[0]
    return con,row[0],row[1],CRS.from_user_input(crs)


def fit_monthly(y):
    """STL trend OLS; HAC lag12 fitted-mean interval; Hamed–Rao MK P."""
    y=np.asarray(y,float);t=np.arange(len(y))/12
    trend=STL(y,period=12,robust=True).fit().trend
    x=np.column_stack([np.ones(len(y)),t]);inv=np.linalg.inv(x.T@x)
    beta=inv@x.T@trend;res=trend-x@beta;xu=x*res[:,None]
    meat=xu.T@xu
    for lag in range(1,13):
        cross=xu[lag:].T@xu[:-lag];meat+=(1-lag/13)*(cross+cross.T)
    cov=inv@meat@inv
    ci=norm.ppf(.975)*np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',x,cov,x),0))
    p=hr_mk_test(trend)["mk_p"]
    return trend,x@beta,ci,float(beta[1]),float(p)


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(width=.65,length=3,labelsize=11)
    ax.set_axisbelow(True);ax.grid(axis='y',color='#ECEFF0',linewidth=.6)
    ax.yaxis.set_major_locator(MaxNLocator(3))


def save_figure(fig,stem):
    for ext in ['png','svg']:
        p=OUTPUT_ROOT/f'{stem}.{ext}'
        fig.savefig(p,dpi=DPI,bbox_inches='tight',pad_inches=.04);log(f'[SAVE] {p}')
    plt.close(fig)


def load_map_vectors(path):
    con,layer,geomcol,target=table_info(path);con.close()
    operation=target.coordinate_operation
    if not target.is_projected or operation is None or 'albers' not in operation.method_name.lower():
        raise ValueError(f'Monthly lake CRS must be Albers: {target.to_string()}')
    if len(target.axis_info)<2 or any(abs(a.unit_conversion_factor-1)>1e-9 for a in target.axis_info[:2]):
        raise ValueError('Monthly lake CRS must use metres')
    geo=gpd.read_file(BOUNDARY)
    if geo.crs is None:raise ValueError('Boundary CRS missing')
    geo=geo.to_crs(target).dissolve()
    if geo.empty or geo.geometry.is_empty.any():raise ValueError('Boundary empty')
    log(f'[CRS] Use monthly lakes: {target.name}; only reference boundaries transformed')
    return geo,target


def attribute_statistics(cache):
    """One geometry-free SQLite aggregate scan per month; individual persistent caches.
    Crucially includes known positive attributes of final_type=99 objects.
    """
    rows=[];sizes=[];unknown=[]
    for year in YEARS:
        for month in range(1,13):
            path=INPUT_ROOT/str(year)/f'{month:02d}'/'LakeAndReservior.gpkg'
            marker=path.parent/'completed.json'
            if not path.is_file():raise FileNotFoundError(path)
            if not marker.is_file():raise ValueError(f'Incomplete source month: {path.parent}')
            record=json.loads(marker.read_text(encoding='utf-8'))
            if record.get('signature',{}).get('version')!='2.0-medium-large':
                raise ValueError(f'Expected upgraded medium/large classification: {path}')
            sig=key([path],[VERSION,'independent flags and sizes',SIZE_NAMES])
            file=cache/f'attributes_{year}_{month:02d}_{sig}.json'
            if file.exists() and not REFRESH_ATTRIBUTE_STATS:
                data=json.loads(file.read_text(encoding='utf-8'));log(f'[STATS CACHE] {year}-{month:02d}')
            else:
                con,layer,geomcol,crs=table_info(path)
                try:
                    fields={r[1] for r in con.execute(f'PRAGMA table_info({q(layer)})')}
                    if not {'is_res','glacial','thermo','med_large','final_type'}<=fields:raise ValueError(f'Missing attribute flags: {path}')

                    natural='COALESCE(is_res,0)=0'
                    conditions=[f'({natural}) AND COALESCE(med_large,0)<=0 AND COALESCE(glacial,0)<=0 AND COALESCE(thermo,0)<=0',
                                f'({natural}) AND med_large=1',f'({natural}) AND glacial=1',f'({natural}) AND thermo=1','is_res=1','1=1']
                    unknown_condition=f'({natural}) AND (COALESCE(glacial,-1)<0 OR COALESCE(thermo,-1)<0 OR COALESCE(med_large,-1)<0)'
                    conditions += [unknown_condition,f'({natural}) AND area_km2<1']
                    size_conditions=[]
                    for lo,hi in zip(SIZE_EDGES[:-1],SIZE_EDGES[1:]):
                        clause=f'({natural}) AND area_km2>={lo}'
                        if np.isfinite(hi):clause+=f' AND area_km2<{hi}'
                        size_conditions.append(clause)
                    all_conditions=conditions+size_conditions
                    expressions=[]
                    for c in all_conditions:
                        expressions += [f'SUM(CASE WHEN {c} THEN 1 ELSE 0 END)',f'SUM(CASE WHEN {c} THEN area_km2 ELSE 0 END)']
                    expressions += ['COUNT(*)','SUM(CASE WHEN area_km2 IS NULL OR area_km2<0 THEN 1 ELSE 0 END)']
                    log(f'[STATS SCAN] {year}-{month:02d}; attributes only, no geometry decoding')
                    values=con.execute(f'SELECT {",".join(expressions)} FROM {q(layer)}').fetchone()
                    if values[-1]:raise ValueError(f'Invalid area field: {path}')
                    pairs=np.asarray([v or 0 for v in values[:-2]],float).reshape(-1,2)
                    data={'signature':sig,'groups':pairs[:6].tolist(),'unknown':pairs[6].tolist(),
                          'small':pairs[7].tolist(),'sizes':pairs[8:].tolist(),'source_count':int(values[-2])}
                    save_json(file,data)
                finally:con.close()
            for code,pair in zip(CODES,data['groups']):
                rows.append([year,month,code,NAMES[code-1],int(pair[0]),float(pair[1])])
            for label,pair in zip(SIZE_NAMES,data['sizes']):sizes.append([year,month,label,int(pair[0]),float(pair[1])])
            unknown.append([year,month,*data['unknown'],*data['small']])
    frame=pd.DataFrame(rows,columns=['year','month','code','type','object_count','area_km2'])
    frame['date']=pd.to_datetime(dict(year=frame.year,month=frame.month,day=1))
    sizes=pd.DataFrame(sizes,columns=['year','month','size_km2','object_count','area_km2'])
    audit=pd.DataFrame(unknown,columns=['year','month','unknown_objects','unknown_area_km2','small_objects','small_area_km2'])
    areas=frame.pivot(index='date',columns='code',values='area_km2').reindex(columns=CODES)
    counts=frame.pivot(index='date',columns='code',values='object_count').reindex(columns=CODES)
    return frame,areas,counts,sizes,audit


def gpkg_wkb(blob):
    if blob is None:return None
    b=bytes(blob)
    if b[:2]!=b'GP':raise ValueError('Expected GeoPackage geometry')
    env=(b[3]>>1)&7;offsets={0:8,1:40,2:56,3:56,4:72}
    if env not in offsets:raise ValueError('Invalid geometry envelope')
    return b[offsets[env]:]


def display_codes(res,medium,glacial,thermo):
    """Map-only priority; independent flags remain intact for all statistics."""
    codes=np.ones(len(res),dtype='uint8')
    codes[np.asarray(medium)==1]=2
    codes[np.asarray(thermo)==1]=3
    codes[np.asarray(glacial)==1]=4
    codes[np.asarray(res)==1]=5
    return codes


def map_cache(path,geo,cache):
    con,layer,geomcol,crs=table_info(path);con.close()
    if not crs.equals(CRS.from_user_input(geo.crs)):raise ValueError('Reference transformation CRS mismatch')
    bounds=np.asarray(geo.total_bounds,float);dx=bounds[2]-bounds[0];dy=bounds[3]-bounds[1]
    bounds+=np.array([-dx,-dy,dx,dy])*.015
    settings=[VERSION,DISPLAY_WIDTH,'all_touched-max-map-priority',crs.to_wkt(),geo.geometry.to_wkb(hex=True).tolist()]
    file=cache/('display_'+key([path],settings)+'.npz')
    if file.exists() and not REFRESH_MAP:
        log(f'[MAP CACHE] {file.name}')
        with np.load(file) as z:return z['overview'],z['bounds']
    nx=max(1,round(DISPLAY_WIDTH*dx/max(dx,dy)));ny=max(1,round(DISPLAY_WIDTH*dy/max(dx,dy)))
    tr=from_bounds(*bounds,nx,ny);arr=np.zeros((ny,nx),dtype='uint8')
    con,layer,geomcol,_=table_info(path)
    total=con.execute(f'SELECT count(*) FROM {q(layer)}').fetchone()[0]
    cur=con.execute(f'SELECT {q(geomcol)},is_res,med_large,glacial,thermo FROM {q(layer)}')
    done=bad=0;started=time.monotonic()
    log(f'[MAP] {total:,} objects; all_touched display {nx} x {ny}; no zoom regions')
    try:
        while True:
            records=cur.fetchmany(READ_BATCH)
            if not records:break
            geoms=shapely.from_wkb([gpkg_wkb(r[0]) for r in records],on_invalid='ignore')
            valid=~shapely.is_missing(geoms)&~shapely.is_empty(geoms)
            bad+=int((~valid).sum())
            flags=np.array([[v if v is not None else -1 for v in r[1:]] for r in records],dtype='int16')
            codes=display_codes(*flags.T)

            # temporary raster for every batch. Max keeps priority independent of row order.
            if valid.any():
                bb=shapely.bounds(geoms[valid]);xmin,ymin=np.nanmin(bb[:,:2],axis=0);xmax,ymax=np.nanmax(bb[:,2:],axis=0)
                c0=max(0,int(np.floor((xmin-bounds[0])/tr.a))-1);c1=min(nx,int(np.ceil((xmax-bounds[0])/tr.a))+1)
                r0=max(0,int(np.floor((bounds[3]-ymax)/abs(tr.e)))-1);r1=min(ny,int(np.ceil((bounds[3]-ymin)/abs(tr.e)))+1)
                if c1>c0 and r1>r0:
                    win=rasterio.windows.Window(c0,r0,c1-c0,r1-r0)
                    ix=np.flatnonzero(valid);ix=ix[np.argsort(codes[ix],kind='stable')]
                    burned=rasterize(((geoms[j],int(codes[j])) for j in ix),out_shape=(r1-r0,c1-c0),
                                      transform=rasterio.windows.transform(win,tr),all_touched=True,dtype='uint8')
                    np.maximum(arr[r0:r1,c0:c1],burned,out=arr[r0:r1,c0:c1])
            done+=len(records)
            if done%100000<len(records):log(f'[MAP] {done:,}/{total:,}; {(time.monotonic()-started)/60:.1f} min')
    finally:con.close()
    mask=geometry_mask(geo.geometry,out_shape=arr.shape,transform=tr,invert=True);arr[~mask]=6
    temp=file.with_suffix('.partial.npz');np.savez_compressed(temp,overview=arr,bounds=bounds);os.replace(temp,file)
    save_json(file.with_suffix('.json'),dict(source=stamp(path),missing_or_empty_geometry=bad,display_only=True,
              all_touched=True,map_priority='Reservoir > Glacial > Thermokarst > Medium and large > Other',
              pixel_width_m=tr.a,pixel_height_m=abs(tr.e)))
    return arr,bounds


def seasonality_metrics(areas):
    """Li et al. seasonal dominance metric, adapted to all 12 calendar months.

    sigma: average within-year population SD; delta: population SD of annual
    means. This is not an additive variance decomposition or a share of Total.
    """
    rows=[]
    for code in PLOT_ORDER:
        series=areas[code]
        if len(series)%12 or not np.isfinite(series).all():
            raise ValueError('Seasonality requires complete, finite calendar years')
        if not np.array_equal(series.index.month,np.tile(np.arange(1,13),len(series)//12)):
            raise ValueError('Seasonality requires January–December order')
        y=series.to_numpy().reshape(-1,12)
        sigma=float(y.std(axis=1,ddof=0).mean())
        delta=float(y.mean(axis=1).std(ddof=0))
        rows.append(dict(code=code,type=NAMES[code-1],mean_area_km2=float(y.mean()),
                         seasonal_sd_km2=sigma,interannual_sd_km2=delta,
                         seasonal_contribution_pct=100*sigma/(sigma+delta) if sigma+delta>0 else np.nan))
    return pd.DataFrame(rows).set_index('code')


def draw_climatology(a,series,code):
    clean(a);y=series.to_numpy().reshape(-1,12);mean=y.mean()
    a.set_title(NAMES[code-1],fontsize=12.5,loc='left',pad=5)
    if mean>0:
        v=100*(y/mean-1);avg=v.mean(axis=0);lo,hi=np.percentile(v,[25,75],axis=0)
        a.fill_between(range(1,13),lo,hi,color=COLORS[code-1],alpha=.2,lw=0)
        a.plot(range(1,13),avg,color=COLORS[code-1],lw=1.5)
    a.axhline(0,color='#AEB7BC',lw=.7)
    a.set_xlim(.8,12.2)
    a.set_xticks([1,4,7,10,12],['Jan','Apr','Jul','Oct','Dec'],fontsize=11)


def draw_main(arr,bounds,geo,areas,counts,fits,metrics):
    fw=16.;left=.85;w=14.65;mh=w*(bounds[3]-bounds[1])/(bounds[2]-bounds[0])
    bottom=.65;rowh=1.65;rgap=.65;sh=2*rowh+rgap;sy=bottom
    by=sy+sh+.9;th=1.85;tgap=.65;bh=2*th+tgap;my=by+bh+.85;fh=my+mh+.45
    fig=plt.figure(figsize=(fw,fh))
    def ax(x,y,ww,h):return fig.add_axes([x/fw,y/fh,ww/fw,h/fh])
    def letter(x,y,s):fig.text(x/fw,y/fh,s,fontsize=17,fontweight='bold',va='bottom')
    cmap=ListedColormap(['white',*MAP_COLORS,'#EBEDEE']);nc=BoundaryNorm(np.arange(-.5,7.5),7)
    a=ax(left,my,w,mh)
    a.imshow(arr,extent=[bounds[0],bounds[2],bounds[1],bounds[3]],origin='upper',cmap=cmap,norm=nc,interpolation='none')
    geo.boundary.plot(ax=a,color='#47545B',linewidth=.7)
    a.set_xlim(bounds[[0,2]]);a.set_ylim(bounds[[1,3]]);a.set_aspect('equal');a.axis('off');add_location_grid(a,geo.crs,5)
    if BASINS.is_file():
        basins=gpd.read_file(BASINS)
        if basins.crs is None:raise ValueError('Basin CRS missing')
        basins=basins.to_crs(geo.crs);basins.boundary.plot(ax=a,color='#87949B',linewidth=.45,alpha=.7,zorder=2)
        if 'Name' in basins:
            for _,r in basins.iterrows():
                p=r.geometry.representative_point();a.text(p.x,p.y,str(r['Name']),fontsize=15,ha='center',va='center',
                  color='black',path_effects=[pe.withStroke(linewidth=2.5,foreground='white')],zorder=4)
    letter(left-.3,my+mh+.07,'a')
    a.legend([Patch(facecolor=MAP_COLORS[i],edgecolor='none') for i in [1,3,2,4,0]],
             [MAP_NAMES[i] for i in [1,3,2,4,0]],loc='upper right',bbox_to_anchor=(.995,.995),
             ncol=5,frameon=False,fontsize=15,
             handlelength=1.0,handletextpad=.4,columnspacing=.65,borderpad=0)
    a.text(.99,.025,f'{calendar.month_name[MAP_MONTH]} {MAP_YEAR}',transform=a.transAxes,ha='right',fontsize=17,color='black')
    snapshot=pd.Timestamp(MAP_YEAR,MAP_MONTH,1);order=[2,3,4,5,1]


    from shapely.geometry import box
    union=geo.geometry.iloc[0]
    chosen=None
    for ww,hh in [(.18,.17),(.16,.16),(.15,.15),(.15,.13),(.14,.13)]:
        for xx in [.09,.075,.065]:
            for yy in [.055,.04]:
                x0=bounds[0]+(xx-.075)*(bounds[2]-bounds[0]);x1=bounds[0]+(xx+ww+.015)*(bounds[2]-bounds[0])
                y0=bounds[1]+(yy-.025)*(bounds[3]-bounds[1]);y1=bounds[1]+(yy+hh+.045)*(bounds[3]-bounds[1])
                if not union.intersects(box(x0,y0,x1,y1)):
                    chosen=[xx,yy,ww,hh];break
            if chosen is not None:break
        if chosen is not None:break
    if chosen is None:

        log('[INSET] No suitable exterior rectangle; area inset omitted')
        extra,ea=plt.subplots(figsize=(4,2.8));clean(ea);ea.barh(range(5),areas.loc[snapshot,order].to_numpy()/1000,color=[COLORS[c-1] for c in order])
        ea.set_yticks(range(5),['Med./large','Glacial','Thermokarst','Reservoirs','Other']);ea.invert_yaxis();ea.set_xlabel('Area (10³ km²)')
        plt.close(extra)
    else:
        inset=a.inset_axes(chosen);clean(inset);inset.grid(False);inset.patch.set_visible(False)
        inset.barh(range(5),areas.loc[snapshot,order].to_numpy()/1000,
                   color=[COLORS[c-1] for c in order],height=.58)
        inset.set_yticks(range(5),['Med./large','Glacial','Thermokarst','Reservoirs','Other'],fontsize=11.5)
        inset.invert_yaxis();inset.set_xlabel('Area (10³ km²)',fontsize=12,labelpad=2)
        inset.xaxis.set_major_locator(MaxNLocator(3));inset.tick_params(labelsize=11)
    letter(left-.3,by+bh+.12,'b');cw=(w-1.44)/3
    for k,code in enumerate(PLOT_ORDER):
        c=ax(left+(k%3)*(cw+.72),by+(1-k//3)*(th+tgap),cw,th);clean(c)
        y=areas[code].to_numpy();tr,pred,ci,slope,p=fits[code];color=COLORS[code-1]
        c.plot(areas.index,y/1000,color=color,lw=.75,alpha=.55)
        c.plot(areas.index,tr/1000,color=color,lw=1.6)
        c.plot(areas.index,pred/1000,color='#55636C',lw=1,ls='--')
        c.fill_between(areas.index,(pred-ci)/1000,(pred+ci)/1000,color=color,alpha=.17,lw=0)
        c.set_title(NAMES[code-1],fontsize=12.5,loc='left',pad=6)
        stars='**' if p<.01 else '*' if p<.05 else ''
        c.text(.025,.95,f'Slope = {slope:+.1f} km² yr⁻¹{stars}',transform=c.transAxes,fontsize=11.5,va='top')
        c.margins(y=.27);c.set_xlim(areas.index[0],areas.index[-1])
        c.set_xticks(pd.to_datetime(['2016-01-01','2019-01-01','2022-01-01','2025-01-01']),['2016','2019','2022','2025'])
        c.set_ylabel('Area (10³ km²)',fontsize=11.5)
        if k==3:alignment_ref=c.yaxis.label
    draw_seasonal_panels(fig,ax,letter,left,w,sy,sh,rowh,rgap,areas,metrics,alignment_ref)
    save_figure(fig,'Figure5')


def draw_seasonal_panels(fig,ax,letter,left,w,sy,sh,rowh,rgap,areas,metrics,alignment_ref):
    """c: seasonal dominance; d: monthly climatology and interannual spread.

    Both use existing 120-month area summaries only. c shows a normalized SD
    metric, not a variance partition or a share of plateau Total.
    """
    labels=['Total','Medium/large','Glacial','Thermokarst','Reservoirs','Other']
    letter(left-.3,sy+sh+.12,'c')

    c=ax(left+.9,sy,4.1,sh);clean(c);c.grid(False)
    fractions=metrics.loc[PLOT_ORDER,'seasonal_contribution_pct'].to_numpy()
    yy=np.arange(len(PLOT_ORDER));color=[COLORS[code-1] for code in PLOT_ORDER]
    # Pale tracks show the 0–100% scale; they are not a second data series.
    c.barh(yy,np.full(len(yy),100),height=.52,color='#E6EAEC',edgecolor='none',zorder=1)
    for k,value in enumerate(fractions):
        if np.isfinite(value):
            c.barh(k,value,left=0,height=.52,color=color[k],edgecolor='none',zorder=2)
        else:c.text(50,k,'No variability',fontsize=11.5,ha='center',va='center',zorder=4)
    c.axvline(50,color='#67777E',lw=.85,ls=(0,(3,3)),zorder=3)
    c.set_yticks(yy,labels,fontsize=12);c.set_ylim(5.6,-.95)
    c.set_xlim(0,100);c.set_xticks([0,25,50,75,100]);c.tick_params(axis='y',length=3,pad=5)
    c.text(25,-.63,'Nonseasonal dominance',ha='center',va='center',fontsize=11,color='#4B585F')
    c.text(75,-.63,'Seasonal dominance',ha='center',va='center',fontsize=11,color='#4B585F')
    c.xaxis.tick_top();c.tick_params(axis='x',pad=3)
    c.set_xlabel('Seasonal contribution (%)',fontsize=12,labelpad=6)
    c.spines['left'].set_visible(True)
    c.spines[['top','right']].set_visible(False)
    dx=left+5.8;dw=w-5.8;cw=(dw-1.0)/3
    letter(dx-.3,sy+sh+.12,'d')
    rows=[]
    for k,code in enumerate(PLOT_ORDER):
        d=ax(dx+(k%3)*(cw+.5),sy+(1-k//3)*(rowh+rgap),cw,rowh);clean(d)
        y=areas[code].to_numpy().reshape(-1,12);mean=float(y.mean())
        d.set_title(NAMES[code-1],loc='left',fontsize=12.5,pad=6)
        if mean>0:
            v=100*(y/mean-1);avg=v.mean(axis=0);lo,hi=np.percentile(v,[25,75],axis=0)
            m=np.arange(1,13);col=COLORS[code-1]


            d.vlines(m,lo,hi,color=col,lw=3.5,alpha=.35)
            d.scatter(m,avg,s=25,color=col,edgecolor='white',lw=.4,zorder=4)
            d.axhline(0,color='#9AA6AC',lw=.7)
            for month in range(12):rows.append([code,NAMES[code-1],month+1,float(y[:,month].mean()),float(avg[month]),float(lo[month]),float(hi[month])])
            d.margins(y=.25)
        else:d.text(.5,.5,'No area recorded',transform=d.transAxes,ha='center',fontsize=11)
        d.set_xlim(.4,12.6);d.set_xticks([1,4,7,10,12],['Jan','Apr','Jul','Oct','Dec'],fontsize=11)
        if k%3==0:d.set_ylabel('Area deviation (%)',fontsize=11.5)
        if k==3:bottom_d=d

    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    desired_left=alignment_ref.get_window_extent(renderer).x0
    medium_left=c.get_yticklabels()[1].get_window_extent(renderer).x0
    c.tick_params(axis='y',pad=5+(medium_left-desired_left)*72/fig.dpi)
    jan=bottom_d.get_xticklabels()[0].get_window_extent(renderer)
    label_height=c.xaxis.label.get_window_extent(renderer).height
    cy=c.transAxes.inverted().transform((c.bbox.x0,jan.y0+label_height/2))[1]
    c.xaxis.set_label_coords(.5,cy);c.xaxis.label.set_verticalalignment('center')


def main():
    started=time.monotonic();OUTPUT_ROOT.mkdir(parents=True,exist_ok=True);cache=CACHE_ROOT;cache.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':12,
                         'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.7,'savefig.facecolor':'white'})
    log('[START] Independent lake attributes; single-month five-colour map; no zoom regions')
    frame,areas,counts,sizes,audit=attribute_statistics(cache)
    path=INPUT_ROOT/str(MAP_YEAR)/f'{MAP_MONTH:02d}'/'LakeAndReservior.gpkg'
    geo,target=load_map_vectors(path);arr,bounds=map_cache(path,geo,cache)
    fits={c:fit_monthly(areas[c]) for c in CODES};metrics=seasonality_metrics(areas)
    draw_main(arr,bounds,geo,areas,counts,fits,metrics)
    log(f'[DONE] {OUTPUT_ROOT}; {(time.monotonic()-started)/60:.1f} min')

def add_location_grid(ax,crs,step=5):
    """Sparse geographic graticule transformed into the existing map CRS.

    Independent of the 25-km analysis grid. Labels use geographic degrees;
    projected x/y coordinates are never relabelled as longitude/latitude.
    """
    from pyproj import Transformer
    from shapely.geometry import LineString, box
    inverse=Transformer.from_crs(crs,"EPSG:4326",always_xy=True)
    forward=Transformer.from_crs("EPSG:4326",crs,always_xy=True)
    x0,x1=ax.get_xlim();y0,y1=ax.get_ylim()
    t=np.linspace(0,1,81)
    px=np.r_[x0+t*(x1-x0),np.full(81,x1),x1-t*(x1-x0),np.full(81,x0)]
    py=np.r_[np.full(81,y0),y0+t*(y1-y0),np.full(81,y1),y1-t*(y1-y0)]
    lon,lat=inverse.transform(px,py)
    lon=np.asarray(lon);lat=np.asarray(lat)
    good=np.isfinite(lon)&np.isfinite(lat)
    if not good.any():return
    west,east=lon[good].min(),lon[good].max()
    south,north=lat[good].min(),lat[good].max()
    frame=box(x0,y0,x1,y1)
    def segments(geom):
        if geom.is_empty:return []
        if geom.geom_type=="LineString":return [geom]
        return [g for g in getattr(geom,"geoms",[]) if g.geom_type=="LineString"]
    for kind,lo,hi in [("lon",west,east),("lat",south,north)]:
        for degree in np.arange(np.ceil(lo/step)*step,hi,step):
            if kind=="lon":
                xx,yy=forward.transform(np.full(401,degree),np.linspace(max(-89,south-2),min(89,north+2),401))
            else:
                xx,yy=forward.transform(np.linspace(west-2,east+2,401),np.full(401,degree))
            coords=np.column_stack([xx,yy]);coords=coords[np.isfinite(coords).all(axis=1)]
            if len(coords)<2:continue
            pieces=segments(LineString(coords).intersection(frame))
            ends=[]
            for piece in pieces:
                q=np.asarray(piece.coords)
                ax.plot(q[:,0],q[:,1],color="#74848B",lw=.35,alpha=.48,
                        linestyle=(0,(3,4)),zorder=2.5,clip_on=True)
                ends.extend([q[0],q[-1]])
            if not ends:continue
            if kind=="lon":
                pt=max(ends,key=lambda v:v[1])
                if abs(pt[1]-y1)>(y1-y0)*.001:continue
                text=f"{abs(degree):g}°"+("E" if degree>=0 else "W")
                ax.annotate(text,pt,xytext=(0,3),textcoords="offset points",ha="center",va="bottom",fontsize=11,color="#59666D",annotation_clip=False)
            else:
                pt=min(ends,key=lambda v:v[0])
                if abs(pt[0]-x0)>(x1-x0)*.001:continue
                text=f"{abs(degree):g}°"+("N" if degree>=0 else "S")
                ax.annotate(text,pt,xytext=(-3,0),textcoords="offset points",ha="right",va="center",fontsize=11,color="#59666D",annotation_clip=False)
    ax.set_xlim(x0,x1);ax.set_ylim(y0,y1)

def hr_mk_test(values, alpha=0.05):
    """Hamed–Rao MK on monthly STL trend; all significant ACF lags, lag=None.

    Equivalent p/z/tau/variance to pyMannKendall 1.4.3 on finite nondegenerate
    series. Constant series return p=1. Nonpositive corrected variance -> NaN.
    p is two-sided and unadjusted for multiple spatial tests, as in the Li map.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
        return dict(mk_p=np.nan, mk_z=np.nan, mk_tau=np.nan, mk_var=np.nan, mk_flag='INVALID')
    n = len(x)
    ii, jj = np.triu_indices(n, 1)
    differences = x[jj] - x[ii]
    score = float(np.sign(differences).sum())
    tau = score / (n * (n - 1) / 2)
    _, ties = np.unique(x, return_counts=True)
    variance = (n*(n-1)*(2*n+5) - np.sum(ties*(ties-1)*(2*ties+5))) / 18
    if variance == 0:
        return dict(mk_p=1., mk_z=0., mk_tau=0., mk_var=0., mk_flag='CONSTANT')
    sen = float(np.median(differences/(jj-ii)))
    ranks = rankdata(x - np.arange(1,n+1)*sen)
    centred = ranks-ranks.mean()
    denominator = float(np.dot(centred,centred))

    if denominator > 0:
        acf = np.correlate(centred,centred,'full')[n-1:]/denominator
        cutoff = norm.ppf(1-alpha/2)/np.sqrt(n)
        correction = 0.
        for lag in range(1,n):
            if abs(acf[lag]) > cutoff:
                correction += (n-lag)*(n-lag-1)*(n-lag-2)*acf[lag]
        variance *= 1 + 2*correction/(n*(n-1)*(n-2))
    if not np.isfinite(variance) or variance <= 0:
        return dict(mk_p=np.nan, mk_z=np.nan, mk_tau=tau, mk_var=variance, mk_flag='BAD_VAR')
    z = (score-np.sign(score))/np.sqrt(variance) if score else 0.
    p = float(2*(1-norm.cdf(abs(z))))
    return dict(mk_p=p, mk_z=float(z), mk_tau=tau, mk_var=float(variance), mk_flag='OK')


if __name__ == '__main__':
    main()
