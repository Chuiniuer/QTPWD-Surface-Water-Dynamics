"""Figure 3: transitions in surface water persistence."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.patheffects as pe
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window, transform as window_transform
from rasterio.transform import from_origin, from_bounds
from rasterio.features import rasterize, geometry_mask
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box, LineString
from shapely.prepared import prep

FIGURE_OUTPUT = Path(r"D:\phdl2\QTP surface water dynamics\code availability\test_output")
BASE = Path(r'K:\Monthly surface water dynamics TIB\supervised classification\step4_analysis')
SOURCE = BASE / 'QTP_FH_results/multi_year_mean_and_transition/Transition_2016_2020_to_2021_2025_revised.tif'
BOUNDARY = Path(r'D:\phdl2\QTP surface water dynamics\regions\QTP.shp')
BASINS = BOUNDARY.with_name('QTP_watershed.shp')
REGIONS = Path(r'K:\Monthly surface water dynamics TIB\QTP_shapefiles\selected visualisation region')
PROJECTED = Path(r'E:\phd_l3\QTP_FH_results\Transition_Albers_10m')
OUTPUT = BASE / 'QTP_Figure3_transitions'
PERIODS = ('2016–2020', '2021–2025')

CODES = [11, 22, 33, 13, 12, 23, 31, 21, 32, 1, 2, 3, 10, 20, 30]
LABELS = [
    'Perennial', 'Wet intermittent', 'Dry intermittent',
    'Perennial to dry intermittent', 'Perennial to wet intermittent',
    'Wet to dry intermittent', 'Dry intermittent to perennial',
    'Wet intermittent to perennial', 'Dry to wet intermittent',
    'New perennial', 'New wet intermittent', 'New dry intermittent',
    'Lost perennial', 'Lost wet intermittent', 'Lost dry intermittent']
HEX = ['#0000FE', '#3E6AC1', '#73B2FF', '#DF8F06', '#DDDC05', '#FEFF73',
       '#019E73', '#98E500', '#73FEDF', '#AA00E5', '#F67AB6', '#DEC5FF',
       '#343434', '#686868', '#9C9C9C']
CLASS_NAMES = ['Nonwater', 'Perennial', 'Wet intermittent', 'Dry intermittent']
# Keep these four legacy groups unchanged: old cached arrays use this order.
GROUP_NAMES = ['New perennial', 'Lost perennial', 'New seasonal', 'Lost seasonal']
GROUP_CODES = [(1,), (10,), (2, 3), (20, 30)]

GROUP_COLORS = ['#41966B', '#B65372', '#A4BE46', '#D8A3AE']
GROUP_STYLES = ['-', '-', '--', '--']
# Five exhaustive groups for the current figure and its geographic profiles.
SUMMARY_NAMES = ['Unchanged', 'Drying', 'Wetting', 'New water', 'Lost water']
SUMMARY_CODES = [(11,22,33), (13,12,23), (31,21,32), (1,2,3), (10,20,30)]
SUMMARY_COLORS = ['#4779AA', '#C59143', '#459F82', '#A07BB8', '#777777']
SUMMARY_STYLES = ['-'] * 5  
ORDER = ['KlB', 'QdB', 'QlB', 'NrB', 'QtB', 'YtB', 'SWB', 'YB', 'STB']
VALID = np.array([0] + CODES, dtype=np.uint8)


def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def signature(paths, settings):
    files = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(json.dumps([files, settings], sort_keys=True).encode()).hexdigest()


def vector_stamp(frame):
    return [frame.crs.to_wkt(), [g.wkb_hex for g in frame.geometry]]


def windows(width, height, size=1024):
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width-col), min(size, height-row))


def raster_grid(geo):
    x0, y0, x1, y1 = geo.total_bounds
    x0 = math.floor(x0/10)*10; y0 = math.floor(y0/10)*10
    x1 = math.ceil(x1/10)*10; y1 = math.ceil(y1/10)*10
    return from_origin(x0, y1, 10, 10), int(round((x1-x0)/10)), int(round((y1-y0)/10))


def load_vectors(args):
    reference = gpd.read_file(args.crs_reference, rows=1)
    crs = reference.crs
    if crs is None or not crs.is_projected:
        raise ValueError('CRS reference must have the same projected Albers CRS as Figure 2.')
    if not crs.coordinate_operation or 'albers' not in crs.coordinate_operation.method_name.lower():
        raise ValueError('An Albers equal-area reference is required.')
    if any(abs(a.unit_conversion_factor-1) > 1e-8 for a in crs.axis_info):
        raise ValueError('Albers coordinates must use metres.')
    geo = gpd.read_file(args.boundary)
    if geo.crs is None: raise ValueError('Boundary CRS missing')
    geo = geo.to_crs(crs).dissolve()
    if geo.empty or not geo.geometry.is_valid.all(): raise ValueError('Invalid boundary')
    basins = gpd.read_file(args.basins)
    if basins.crs is None or 'Name' not in basins: raise ValueError('Basins require Name and CRS')
    if basins.Name.isna().any(): raise ValueError('Missing basin Name')
    basins = basins.to_crs(crs)
    basins['Name'] = basins.Name.astype(str).replace({'KIB': 'KlB', 'QIB': 'QlB'})
    basins = basins.dissolve(by='Name').reset_index()
    if not basins.geometry.is_valid.all(): raise ValueError('Invalid basin geometry')
    for i in range(len(basins)):
        for j in range(i):
            if basins.geometry.iloc[i].intersection(basins.geometry.iloc[j]).area > 1.:
                raise ValueError('Basin polygons overlap; areas cannot be allocated unambiguously.')
    names = [n for n in ORDER if n in set(basins.Name)] + sorted(set(basins.Name)-set(ORDER))
    basins = basins.set_index('Name').loc[names].reset_index()
    regions = []
    for n in range(1, 7):
        p = args.regions_dir / f'r{n}.shp'
        g = gpd.read_file(p)
        if g.crs is None or len(g) != 1 or not g.geometry.is_valid.all():
            raise ValueError(f'{p}: requires one valid polygon and CRS')
        if g.geometry.iloc[0].geom_type not in ('Polygon', 'MultiPolygon'):
            raise ValueError(f'{p}: polygon required')
        regions.append(g.to_crs(crs))
    return crs, geo, basins, regions


def project_transition(source, dest, geo, rebuild=False):
    tr, nx, ny = raster_grid(geo)
    key = signature([source], [vector_stamp(geo), list(tr), nx, ny, 'transition-10m-nearest-v1'])
    meta = dest.with_suffix('.json')
    if dest.exists() and meta.exists() and not rebuild:
        record = json.loads(meta.read_text(encoding='utf-8'))
        if record.get('key') == key:
            print('REUSE 10 m projected raster:', dest, flush=True)
            return
    if dest.exists() and not rebuild:
        raise RuntimeError('Existing projected raster has different metadata. It was NOT overwritten. '
                           'Only use --rebuild-projection if a new projection is intended.')
    boundary = geo.geometry.iloc[0]; prepared = prep(boundary)
    profile = dict(driver='GTiff', width=nx, height=ny, count=1, dtype='uint8',
                   crs=geo.crs, transform=tr, nodata=255, tiled=True,
                   blockxsize=512, blockysize=512, compress='DEFLATE', zlevel=6,
                   predictor=2, BIGTIFF='YES', SPARSE_OK='TRUE')
    temp = dest.with_suffix('.partial.tif')
    print('CREATE 10 m Albers transition raster (first run only):', dest, flush=True)
    count = math.ceil(nx/1024)*math.ceil(ny/1024); last = time.monotonic()
    with rasterio.Env(GDAL_CACHEMAX=256*1024*1024):
        with rasterio.open(source) as src, rasterio.open(temp, 'w', **profile) as dst:
            if src.crs is None: raise ValueError('Transition raster CRS missing')
            src_nodata = src.nodata
            with WarpedVRT(src, crs=geo.crs, transform=tr, width=nx, height=ny,
                           nodata=255, dtype='float32', resampling=Resampling.nearest,
                           warp_mem_limit=128) as vrt:
                for i, win in enumerate(windows(nx, ny)):
                    if not prepared.intersects(box(*rasterio.windows.bounds(win, tr))): continue
                    a = vrt.read(1, window=win, masked=True).filled(255)
                    inside = geometry_mask([boundary], out_shape=a.shape,
                                           transform=window_transform(win, tr), invert=True)
                    selected = a[inside & (a != 255)]
                    if selected.size and not np.isin(selected, VALID).all():
                        bad = np.unique(selected[~np.isin(selected, VALID)])[:12]
                        raise ValueError(f'Unexpected transition codes {bad}; expected 10*start+end (0–3).')
                    a[~inside] = 255
                    dst.write(a.astype('uint8'), 1, window=win)
                    if time.monotonic()-last > 15:
                        print(f'  projection blocks {i+1:,}/{count:,}', flush=True); last=time.monotonic()
    os.replace(temp, dest)
    atomic_json(meta, dict(key=key, source=str(source), source_nodata=src_nodata,
                          code_formula='10*earlier_class+later_class', resolution_m=10))


def geographic_edges(geo, step):
    
    x0,y0,x1,y1 = geo.total_bounds
    t=np.linspace(0,1,301)
    xx=np.r_[x0+t*(x1-x0),np.full(t.size,x1),x1-t*(x1-x0),np.full(t.size,x0)]
    yy=np.r_[np.full(t.size,y0),y0+t*(y1-y0),np.full(t.size,y1),y1-t*(y1-y0)]
    lon,lat=Transformer.from_crs(geo.crs,4326,always_xy=True).transform(xx,yy)
    
    edges=[]
    for vals in (lon,lat):
        edges.append(np.arange((math.floor(min(vals)/step)-1)*step,
                               (math.ceil(max(vals)/step)+2)*step,step))
    return edges


def calculate_statistics(projected, geo, basins, cache, step, refresh=False):
    key=signature([projected],[vector_stamp(geo),vector_stamp(basins),list(basins.Name),step,'transition-stats-v1'])
    file=cache/('statistics_'+key[:20]+'.npz')
    if file.exists() and not refresh:
        print('REUSE transition statistics:',file.name,flush=True)
        return read_npz(file),file
    le,ae=geographic_edges(geo,step)
    counts=np.zeros(34,np.int64); region=np.zeros((len(basins),34),np.int64)
    lon_counts=np.zeros((4,len(le)-1),np.int64);lat_counts=np.zeros((4,len(ae)-1),np.int64)
    uncovered=np.zeros(34,np.int64)
    group_lut=np.full(256,-1,np.int8)
    for k,codes in enumerate(GROUP_CODES): group_lut[list(codes)]=k
    prepared=[prep(g) for g in basins.geometry]; boundary=prep(geo.geometry.iloc[0])
    convert=Transformer.from_crs(geo.crs,4326,always_xy=True)
    print('CALCULATE full-resolution areas, basin areas and profiles in one scan.',flush=True)
    last=time.monotonic()
    with rasterio.open(projected) as src:
        tr=src.transform;pixel_area=abs(tr.a*tr.e-tr.b*tr.d)/1e6
        if not np.isclose(pixel_area,.0001,rtol=0,atol=1e-12):raise ValueError('Expected 10 m equal-area pixels')
        for i,win in enumerate(windows(src.width,src.height,2048)):
            tile=box(*rasterio.windows.bounds(win,tr))
            if not boundary.intersects(tile):continue
            a=src.read(1,window=win);valid=a!=255
            vals=a[valid]
            if not np.isin(vals,VALID).all():raise ValueError('Unexpected projected transition value')
            counts+=np.bincount(vals,minlength=34)
            water=valid&(a!=0)
            if water.any():
                shapes=[(g,j+1) for j,g in enumerate(basins.geometry) if prepared[j].intersects(tile)]
                ids=rasterize(shapes,out_shape=a.shape,transform=window_transform(win,tr),
                              fill=0,dtype='uint16') if shapes else np.zeros(a.shape,np.uint16)
                joint=ids[water].astype(np.int64)*34+a[water]
                allocation=np.bincount(joint,minlength=(len(basins)+1)*34).reshape(len(basins)+1,34)
                uncovered+=allocation[0];region+=allocation[1:]
            gi=group_lut[a]
            rr,cc=np.nonzero(gi>=0)
            if len(rr):
                x=tr.c+(cc+win.col_off+.5)*tr.a
                y=tr.f+(rr+win.row_off+.5)*tr.e
                lon,lat=convert.transform(x,y);groups=gi[rr,cc]
                for k in range(4):
                    m=groups==k
                    lon_counts[k]+=np.histogram(np.asarray(lon)[m],le)[0]
                    lat_counts[k]+=np.histogram(np.asarray(lat)[m],ae)[0]
            if time.monotonic()-last>15:
                print('  statistics block',i+1,flush=True);last=time.monotonic()
    expected=np.array([counts[list(c)].sum() for c in GROUP_CODES])
    if not np.array_equal(lon_counts.sum(axis=1),expected) or not np.array_equal(lat_counts.sum(axis=1),expected):
        raise ValueError('Geographic band totals differ from global event areas; no incomplete statistics saved.')
    if not np.array_equal(region.sum(axis=0)[CODES]+uncovered[CODES],counts[CODES]):
        raise ValueError('Basin allocation balance failed')
    if uncovered[CODES].sum():
        warnings.warn(f'{uncovered[CODES].sum()*pixel_area:.4f} km² outside basin polygons; retained in QTP totals.')
    result=dict(counts=counts,basin_counts=region,unassigned_counts=uncovered,
                basin_names=np.array(list(basins.Name)),lon_edges=le,lat_edges=ae,
                lon_counts=lon_counts,lat_counts=lat_counts,pixel_area=np.array(pixel_area))
    atomic_npz(file,**result)
    return result,file


def group_values(counts, pixel_area=1.):
    """Exhaustive partition of the 15 water-related transitions; excludes 0->0."""
    return np.array([np.asarray(counts)[list(c)].sum() for c in SUMMARY_CODES]) * pixel_area


def validate_group_profiles(s, profiles):
    expected=group_values(s['counts']).astype(np.int64)
    for dim in ('lon','lat'):
        key='summary_'+dim+'_counts'
        arr=np.asarray(profiles[key])
        if arr.shape!=(5,len(s[dim+'_edges'])-1) or (arr<0).any():
            raise ValueError('Invalid five-group profile cache: '+key)
        if not np.array_equal(arr.sum(axis=1),expected):
            raise ValueError('Five-group profile areas do not match the original transition statistics.')
    if int(expected.sum())!=int(s['counts'][CODES].sum()):
        raise ValueError('Five-group partition does not cover all 15 transition classes.')


def five_group_profiles(projected, geo, cache, s):
    """Add three missing profiles ONCE; keep all old analysis/display caches.

    The new/lost profiles are sums of the original four profiles. Only stable,
    drying and wetting pixels need geographic coordinates in this extra pass.
    Neither reprojection nor basin statistics are repeated.
    """
    key=signature([projected],[vector_stamp(geo),SUMMARY_CODES,
        s['lon_edges'].tolist(),s['lat_edges'].tolist(),s['counts'].tolist(),
        'transition-five-profiles-v1'])
    file=cache/('five_group_profiles_'+key[:20]+'.npz')
    if file.exists():
        result=read_npz(file);validate_group_profiles(s,result)
        print('REUSE five-group geographic profiles:',file.name,flush=True)
        return result,file
    result={}
    for dim in ('lon','lat'):
        arr=np.zeros((5,len(s[dim+'_edges'])-1),np.int64)
        old=s[dim+'_counts']
        arr[3]=old[0]+old[2];arr[4]=old[1]+old[3]
        result['summary_'+dim+'_counts']=arr
    print('ADD five-group profiles: one pass over existing Albers raster; NO reprojection.',flush=True)
    lut=np.full(256,-1,np.int8)
    for k,codes in enumerate(SUMMARY_CODES[:3]):lut[list(codes)]=k
    inverse=Transformer.from_crs(geo.crs,4326,always_xy=True)
    boundary=prep(geo.geometry.iloc[0]);last=time.monotonic()
    with rasterio.open(projected) as src:
        tr=src.transform
        if src.crs!=geo.crs or not np.isclose(abs(tr.a*tr.e-tr.b*tr.d)/1e6,float(s['pixel_area'])):
            raise ValueError('Projected raster and cached area statistics do not use the same grid.')
        nblocks=math.ceil(src.width/2048)*math.ceil(src.height/2048)
        for i,win in enumerate(windows(src.width,src.height,2048)):
            if not boundary.intersects(box(*rasterio.windows.bounds(win,tr))):continue
            codes=src.read(1,window=win);groups=lut[codes]
            rr,cc=np.nonzero(groups>=0)
            
            for start in range(0,len(rr),262144):
                r=rr[start:start+262144];c=cc[start:start+262144]
                x=tr.c+(c+win.col_off+.5)*tr.a+(r+win.row_off+.5)*tr.b
                y=tr.f+(c+win.col_off+.5)*tr.d+(r+win.row_off+.5)*tr.e
                lon,lat=inverse.transform(x,y);g=groups[r,c]
                for dim,coord in [('lon',lon),('lat',lat)]:
                    for k in range(3):
                        result['summary_'+dim+'_counts'][k]+=np.histogram(
                            np.asarray(coord)[g==k],s[dim+'_edges'])[0]
            if time.monotonic()-last>15:
                print(f'  missing-profile blocks {i+1:,}/{nblocks:,}',flush=True)
                last=time.monotonic()
    validate_group_profiles(s,result)
    atomic_npz(file,**result)
    return result,file


def make_display(projected,geo,cache,width):
    key=signature([projected],[vector_stamp(geo),width,'transition-display-v1'])
    file=cache/('display_'+key[:20]+'.npz')
    if file.exists():
        print('REUSE overview display:',file.name,flush=True);return read_npz(file),file
    print('CREATE sampled overview (display only).',flush=True)
    with rasterio.open(projected) as src:
        ratio=min(1.,width/max(src.width,src.height))
        nx=max(1,round(src.width*ratio));ny=max(1,round(src.height*ratio))
        a=src.read(1,out_shape=(ny,nx),resampling=Resampling.nearest)
        b=src.bounds
        inside=geometry_mask(geo.geometry,out_shape=a.shape,transform=from_bounds(*b,nx,ny),invert=True)
        a[~inside]=254
        data=dict(data=a,extent=np.array([b.left,b.right,b.bottom,b.top]))
    atomic_npz(file,**data);return data,file


def local_views(projected,regions,cache,width):
    previews=[];files=[]
    for i,g in enumerate(regions,1):
        bounds=tuple(g.total_bounds)
        key=signature([projected],[vector_stamp(g),width,'transition-zoom-v1'])
        file=cache/('zoom_'+key[:20]+'.npz')
        if file.exists():
            print('REUSE region display:',i,flush=True);data=read_npz(file)
        else:
            print('CREATE region display:',i,flush=True)
            scale=width/max(bounds[2]-bounds[0],bounds[3]-bounds[1])
            nx=max(1,round((bounds[2]-bounds[0])*scale));ny=max(1,round((bounds[3]-bounds[1])*scale))
            tr=from_bounds(*bounds,nx,ny)
            with rasterio.open(projected) as src:
                with WarpedVRT(src,crs=g.crs,transform=tr,width=nx,height=ny,nodata=255,
                               resampling=Resampling.nearest,warp_mem_limit=128) as vrt:
                    a=vrt.read(1)
            inside=geometry_mask(g.geometry,out_shape=a.shape,transform=tr,invert=True)
            a[~inside]=255
            data=dict(data=a,bounds=np.array(bounds));atomic_npz(file,**data)
        previews.append(data);files.append(str(file.resolve()))
    return previews,files


def transition_matrix(counts,pixel_area):
    # Includes valid 0→0 only in the matrix; excludes it from water footprint.
    return np.array([[counts[10*i+j]*pixel_area for j in range(4)] for i in range(4)])


def budgets(matrix):
    before=matrix.sum(axis=1);after=matrix.sum(axis=0)
    stay=np.diag(matrix);gain=after-stay;loss=before-stay
    total_gain=matrix[0,1:].sum();total_loss=matrix[1:,0].sum()
    return before,after,gain,loss,np.r_[gain[1:],total_gain],np.r_[loss[1:],total_loss]


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=11.5,length=3,width=.65)
    ax.set_axisbelow(True)


def panel_letter(fig,x,y,letter,fw,fh):
    fig.text(x/fw,y/fh,letter,fontsize=16,fontweight='bold',ha='left',va='bottom')


def map_colours():
    colours=['#FFFFFF']*256
    for c,h in zip(CODES,HEX):colours[c]=h
    colours[254]='#D8D8D8'
    return ListedColormap(colours),BoundaryNorm(np.arange(-.5,256.5,1),256)


def add_basin_boundaries_and_labels(ax, basin_path, crs, regions):
    """Figure-1 basin styling; only the overview map is changed.

    Keep raw Name labels and representative-point anchors from Figure 1.
    Move labels inside their own basin when necessary to protect zoom boxes,
    their existing numbers, other map text and existing inset axes.
    """
    from matplotlib.transforms import Bbox
    from shapely.geometry import Point, box as geometry_box

    basin_path = Path(basin_path)
    if not basin_path.is_file():
        raise FileNotFoundError(basin_path)
    basins = gpd.read_file(basin_path)
    if basins.crs is None:
        raise ValueError('Basin CRS missing')
    if 'Name' not in basins:
        raise ValueError('Basin file requires Name')
    basins = basins.to_crs(crs)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    basins.boundary.plot(ax=ax, color='#87949B', linewidth=.45,
                         alpha=.7, zorder=2)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    
    fontsize = 15. * (ax.get_window_extent(renderer).width / fig.dpi) / 14.65
    margin = 3. * fig.dpi / 72.
    blocked = []
    for region in regions:
        x0, y0, x1, y1 = region.total_bounds
        pts = ax.transData.transform([[x0, y0], [x1, y1]])
        blocked.append(Bbox.from_extents(pts[:, 0].min(), pts[:, 1].min(),
                                        pts[:, 0].max(), pts[:, 1].max()).padded(margin))
    
    blocked.extend(t.get_window_extent(renderer).padded(margin)
                   for t in ax.texts if t.get_visible() and t.get_text())
    for child in ax.child_axes:
        blocked.append(child.get_tightbbox(renderer).padded(margin))
    frame = ax.get_window_extent(renderer)
    inverse = ax.transData.inverted()
    for _, row in basins.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        anchor = geom.representative_point()
        text = ax.text(anchor.x, anchor.y, str(row['Name']), fontsize=fontsize,
                       ha='center', va='center', color='black', zorder=4,
                       path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
        
        measured = text.get_window_extent(renderer)
        halfw, halfh = measured.width / 2., measured.height / 2.
        padding = 2. * fig.dpi / 72.
        gx0, gy0, gx1, gy1 = geom.bounds
        candidates = [(anchor.x, anchor.y)]
        grid = [(float(x), float(y)) for x in np.linspace(gx0, gx1, 49)
                for y in np.linspace(gy0, gy1, 49)
                if geom.covers(Point(float(x), float(y)))]
        grid.sort(key=lambda p: (p[0]-anchor.x)**2 + (p[1]-anchor.y)**2)
        candidates.extend(grid)
        chosen = None
        
        
        for require_inside in (True, False):
            for x, y in candidates:
                px, py = ax.transData.transform((x, y))
                bbox = Bbox.from_extents(px-halfw-padding, py-halfh-padding,
                                        px+halfw+padding, py+halfh+padding)
                if not (frame.contains(bbox.x0, bbox.y0) and
                        frame.contains(bbox.x1, bbox.y1)):
                    continue
                if any(bbox.overlaps(other) for other in blocked):
                    continue
                if require_inside:
                    corners = inverse.transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
                    if not geom.covers(geometry_box(*corners.ravel())):
                        continue
                chosen = (x, y, bbox)
                break
            if chosen is not None:
                break
        if chosen is None:
            text.remove()
            raise RuntimeError(f"No unobstructed basin-label position for {row['Name']}; "
                               'the existing zoom boxes and numbers were preserved.')
        text.set_position(chosen[:2])
        blocked.append(chosen[2].padded(margin))
    ax.set_xlim(xlim); ax.set_ylim(ylim)


def draw_figure(s,display,geo,regions,previews,out,dpi=300,basin_path=BASINS):
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':12.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.7,
        'savefig.facecolor':'white','axes.unicode_minus':True})
    extent=display['extent'];cmap,norm=map_colours()
    
    
    fw=16.;left=.85;mapw=11.45;gap=.68;sidew=2.35
    maph=mapw*(extent[3]-extent[2])/(extent[1]-extent[0])
    bottom=.64;bottomh=3.25;rowgap=.93
    zoomw=8.30;zoomgap=.12;cell=(zoomw-2*zoomgap)/3;zoomh=2*cell+zoomgap
    midy=bottom+bottomh+rowgap;legendh=1.08
    legendy=midy+zoomh+.19;midh=zoomh+.19+legendh
    mapy=midy+midh+.44
    topy=mapy+maph+.65;toph=1.35;fh=topy+toph+.21
    fig=plt.figure(figsize=(fw,fh))
    def ax(x,y,w,h):return fig.add_axes([x/fw,y/fh,w/fw,h/fh])
    main=ax(left,mapy,mapw,maph);top=ax(left,topy,mapw,toph)
    right=ax(left+mapw+gap,mapy,sidew,maph)
    llines=ax(left+mapw+gap,topy,sidew,toph);llines.axis('off')
    main.imshow(display['data'],extent=extent,origin='upper',cmap=cmap,norm=norm,
                interpolation='nearest',zorder=1)
    geo.boundary.plot(ax=main,color='#545454',linewidth=.65,zorder=4)
    main.set_xlim(extent[:2]);main.set_ylim(extent[2:]);main.set_aspect('equal');main.set_axis_off()
    add_location_grid(main,geo.crs)
    for n,g in enumerate(regions,1):
        g.boundary.plot(ax=main,color='black',linewidth=1.,zorder=5)
        x0,y0,x1,y1=g.total_bounds
        main.annotate(str(n),(x1,y1),xytext=(3,3),textcoords='offset points',fontsize=18,
                      fontweight='bold',zorder=6,path_effects=[pe.withStroke(linewidth=2.5,foreground='white')])
    main.set_xlim(extent[:2]);main.set_ylim(extent[2:])
    p=float(s['pixel_area'])
    lon=(s['lon_edges'][:-1]+s['lon_edges'][1:])/2
    lat=(s['lat_edges'][:-1]+s['lat_edges'][1:])/2
    for k in range(5):
        top.plot(lon,s['summary_lon_counts'][k]*p/1000,
                 color=SUMMARY_COLORS[k],ls=SUMMARY_STYLES[k],lw=1.6)
        right.plot(s['summary_lat_counts'][k]*p/1000,lat,
                   color=SUMMARY_COLORS[k],ls=SUMMARY_STYLES[k],lw=1.6)
    limits=geo.to_crs(4326).total_bounds
    top.set_xlim(limits[0]-.1,limits[2]+.1);top.set_ylim(bottom=0)
    right.set_ylim(limits[1]-.1,limits[3]+.1);right.set_xlim(left=0)
    top.set_xlabel('Longitude (°E)',labelpad=4);top.set_ylabel('Area (10³ km²)')
    right.set_xlabel('Area (10³ km²)');right.set_ylabel('Latitude (°N)',labelpad=4)
    top.yaxis.set_major_locator(MaxNLocator(4));right.xaxis.set_major_locator(MaxNLocator(3))
    clean(top);clean(right)
    top.grid(axis='y',color='#EAEAEA',lw=.5);right.grid(axis='y',color='#EAEAEA',lw=.5)
    llines.legend([Line2D([0],[0],color=c,ls=ls,lw=2.) for c,ls in zip(SUMMARY_COLORS,SUMMARY_STYLES)],
                  SUMMARY_NAMES,frameon=False,loc='center left',fontsize=12.5,
                  handlelength=2.3,handletextpad=.7,labelspacing=.43,borderaxespad=0)
    panel_letter(fig,left-.32,mapy+maph+.16,'a',fw,fh)
    
    legend=ax(left+.035,legendy,zoomw-.035,legendh);legend.axis('off')
    compact=['Perennial','Wet intermittent','Dry intermittent',
             'Pere. to dry inte.','Pere. to wet inte.','Wet inte. to dry inte.',
             'Dry inte. to pere.','Wet inte. to pere.','Dry inte. to wet inte.',
             'New perennial','New wet intermittent','New dry intermittent',
             'Lost perennial','Lost wet intermittent','Lost dry intermittent']
    
    
    leglabels=['Perennial','Wet intermittent','Dry intermittent',
               'Pere. to dry inte.','Pere. to wet inte.','Wet to dry inte.',
               'Dry inte. to pere.','Wet inte. to pere.','Dry to wet inte.',
               'New perennial','New wet inte.','New dry inte.',
               'Lost perennial','Lost wet inte.','Lost dry inte.']
    legend_width=zoomw-.035
    legend.set_xlim(0,legend_width);legend.set_ylim(0,legendh)
    renderer=fig.canvas.get_renderer()
    normal=FontProperties(family='serif',size=12.)
    bold=FontProperties(family='serif',size=12.5,weight='bold')
    square=9/72.;textgap=4/72.
    widths=[]
    for col,heading in enumerate(SUMMARY_NAMES):
        entry_width=max(renderer.get_text_width_height_descent(t,normal,False)[0]/fig.dpi
                        for t in leglabels[col*3:col*3+3])
        heading_width=renderer.get_text_width_height_descent(heading,bold,False)[0]/fig.dpi
        widths.append(max(square+textgap+entry_width,heading_width))
    colgap=(legend_width-sum(widths))/4
    if colgap<.02:
        raise ValueError('Legend font is wider than the available space; increase zoomw in draw_figure.')
    xx=0.
    for col,heading in enumerate(SUMMARY_NAMES):
        legend.text(xx,legendh*.97,heading,ha='left',va='top',fontproperties=bold)
        for row in range(3):
            k=col*3+row;yy=legendh*(.66-.26*row)
            legend.add_patch(Rectangle((xx,yy-square/2),square,square,
                                       facecolor=HEX[k],edgecolor='#666666',lw=.45))
            legend.text(xx+square+textgap,yy,leglabels[k],ha='left',va='center',
                        fontproperties=normal)
        xx+=widths[col]+colgap
    for i,preview in enumerate(previews):
        col=i%3;row=i//3;x=left+col*(cell+zoomgap);y=midy+(1-row)*(cell+zoomgap)
        bx=ax(x,y,cell,cell);b=preview['bounds'];a=preview['data']
        bx.imshow(a,extent=[b[0],b[2],b[1],b[3]],origin='upper',cmap=cmap,norm=norm,interpolation='nearest')
        bx.set_xlim(b[0],b[2]);bx.set_ylim(b[1],b[3]);bx.set_aspect('equal')
        bx.set_xticks([]);bx.set_yticks([])
        for sp in bx.spines.values():sp.set_color('#777777');sp.set_linewidth(.6)
        bx.text(.03,.97,str(i+1),transform=bx.transAxes,ha='left',va='top',fontsize=18,
                fontweight='bold',bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=1.2))
    
    bar=ax(11.10,midy,4.23,midh-.04)
    vals=s['counts'][CODES]*p/1000;ys=np.arange(15)
    bar.barh(ys,vals,color=HEX,height=.70,edgecolor='#666666',linewidth=.45)
    clean(bar);bar.set_yticks(ys,compact,fontsize=12.)
    bar.invert_yaxis();bar.tick_params(axis='y',length=0)
    maximum=max(float(vals.max()),1e-6);bar.set_xlim(0,maximum*1.06)
    bar.xaxis.set_major_locator(MaxNLocator(4));bar.set_xlabel('Area (10³ km²)',labelpad=4)
    for y in [2.5,5.5,8.5,11.5]:bar.axhline(y,color='#D4D4D4',lw=.7,zorder=0)
    bar.grid(axis='x',color='#ECECEC',lw=.55)
    
    budget=ax(left+.15,bottom,6.25,bottomh)
    m=transition_matrix(s['counts'],p);_,_,_,_,g,l=budgets(m)
    g=g/1000;l=l/1000;net=g-l;x=np.arange(4)
    budget.bar(x-.23,g,width=.27,color='#72B7A0',label='Gross gain')
    budget.bar(x+.07,-l,width=.27,color='#D28B9A',label='Gross loss')
    budget.bar(x+.31,net,width=.14,facecolor='#E9ECEF',edgecolor='#586875',
               linewidth=1.2,label='Net change',zorder=3)
    budget.axhline(0,color='#666666',lw=.7)
    limit=max(float(g.max()),float(l.max()),1e-6)
    budget.set_ylim(-limit*1.18,limit*1.18);budget.set_xlim(-.6,3.62)
    budget.set_xticks(x,['Perennial','Wet\nintermittent','Dry\nintermittent','Total water'])
    budget.set_ylabel('Area change (10³ km²)',labelpad=5)
    budget.yaxis.set_major_locator(MaxNLocator(5));clean(budget)
    budget.grid(axis='y',color='#EBEBEB',lw=.5)
    budget.legend(loc='lower center',bbox_to_anchor=(.5,1.015),ncol=3,frameon=False,
                  fontsize=12.,handlelength=1.0,handletextpad=.45,columnspacing=.85,borderaxespad=0)
    panel_letter(fig,left-.32,bottom+bottomh+.34,'b',fw,fh)
    # c: exhaustive within-basin composition, not the four incomplete events.
    basin=ax(8.25,bottom,7.08,bottomh)
    names=s['basin_names'];values=np.array([group_values(cnt) for cnt in s['basin_counts']],float)
    denominators=values.sum(axis=1)
    shares=np.divide(100*values,denominators[:,None],out=np.zeros_like(values),where=denominators[:,None]>0)
    y=np.arange(len(names));cumulative=np.zeros(len(names))
    for k in range(5):
        basin.barh(y,shares[:,k],left=cumulative,color=SUMMARY_COLORS[k],height=.62,
                   edgecolor='white',linewidth=.5)
        cumulative+=shares[:,k]
    for j,den in enumerate(denominators):
        if den==0:basin.text(50,j,'No water transitions',ha='center',va='center',fontsize=11.)
    basin.set_xlim(0,100);basin.set_ylim(len(names)-.5,-.65)
    clean(basin);basin.set_yticks(y,names,fontsize=12.5);basin.tick_params(axis='y',length=0)
    basin.set_xticks([0,25,50,75,100]);basin.set_xlabel('Transition area share (%)',labelpad=4)
    basin.grid(axis='x',color='#ECECEC',lw=.55)
    basin.legend([Patch(facecolor=c,edgecolor='none') for c in SUMMARY_COLORS],SUMMARY_NAMES,
                 loc='lower left',bbox_to_anchor=(0,1.015),ncol=5,frameon=False,
                 fontsize=12.,handlelength=1.0,handletextpad=.45,columnspacing=.85,borderaxespad=0)
    panel_letter(fig,7.79,bottom+bottomh+.34,'c',fw,fh)
    add_basin_boundaries_and_labels(main,basin_path,geo.crs,regions)
    for ext in ('png','svg'):
        path=out/f'Figure3_water_type_transitions.{ext}'
        fig.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=.04)
        print('Saved:',path,flush=True)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--boundary',type=Path,default=BOUNDARY)
    parser.add_argument('--basins',type=Path,default=BASINS)
    parser.add_argument('--regions-dir',type=Path,default=REGIONS)
    parser.add_argument('--crs-reference',type=Path,default=REGIONS/'r1.shp')
    parser.add_argument('--projected-dir',type=Path,default=PROJECTED)
    parser.add_argument('--output',type=Path,default=OUTPUT,help='Directory for existing cache files')
    parser.add_argument('--figure-output',type=Path,default=FIGURE_OUTPUT)
    parser.add_argument('--bin-degrees',type=float,default=.1)
    parser.add_argument('--display-width',type=int,default=5000)
    parser.add_argument('--zoom-width',type=int,default=1200)
    parser.add_argument('--dpi',type=int,default=300)
    parser.add_argument('--rebuild-projection',action='store_true')
    parser.add_argument('--refresh-statistics',action='store_true')
    parser.add_argument('--plot-only',action='store_true',help='Use saved arrays only; missing/mismatched caches stop without raster processing')
    args=parser.parse_args()
    out=args.output;manifest=out/'Figure3_run.json'
    if not .05<=args.bin_degrees<=2:parser.error('--bin-degrees must be 0.05–2')
    if not 500<=args.display_width<=12000 or not 200<=args.zoom_width<=4000:parser.error('Invalid display dimensions')
    if args.dpi<72 or args.dpi>600:parser.error('--dpi must be 72–600')
    if args.plot_only and (args.rebuild_projection or args.refresh_statistics):parser.error('--plot-only cannot rebuild caches')
    crs,geo,basins,regions=load_vectors(args)
    print('Using selected-region CRS:',crs.name,flush=True)
    vector_key=[vector_stamp(geo),vector_stamp(basins),list(basins.Name),[vector_stamp(g) for g in regions]]
    out.mkdir(parents=True,exist_ok=True);cache=out/'cache';cache.mkdir(exist_ok=True)
    args.figure_output.mkdir(parents=True,exist_ok=True)
    if args.plot_only:
        if not manifest.exists():raise FileNotFoundError('No Figure3_run.json. Run once without --plot-only first.')
        old=json.loads(manifest.read_text(encoding='utf-8'))
        if old['vectors']!=vector_key:raise RuntimeError('Vector geometry changed. No raster processing started in --plot-only mode.')
        if old['source']!=str(args.source.resolve()) or old['bin_degrees']!=args.bin_degrees:
            raise RuntimeError('Source path or geographic bin setting differs from cached run.')
        if old['display_width']!=args.display_width or old['zoom_width']!=args.zoom_width:
            raise RuntimeError('Display resolution differs from cached run; use the original settings for --plot-only.')
        s=read_npz(Path(old['statistics']))
        if not old.get('group_profiles') or not Path(old['group_profiles']).is_file():
            raise FileNotFoundError('The old cache contains only four event profiles. Run once without '
                '--plot-only to add the missing three profiles from the existing projected raster. '
                'Existing projection, area statistics and display caches will be reused.')
        profiles=read_npz(Path(old['group_profiles']));validate_group_profiles(s,profiles)
        s.update(profiles);display=read_npz(Path(old['display']))
        previews=[read_npz(Path(f)) for f in old['zooms']]
        print('PLOT ONLY: reusing the completed statistical snapshot; no source TIFF reads.',flush=True)
    else:
        if not args.source.is_file():raise FileNotFoundError(args.source)
        args.projected_dir.mkdir(parents=True,exist_ok=True)
        projected=args.projected_dir/(args.source.stem+'_Albers10m.tif')
        project_transition(args.source,projected,geo,args.rebuild_projection)
        s,sfile=calculate_statistics(projected,geo,basins,cache,args.bin_degrees,args.refresh_statistics)
        profiles,gfile=five_group_profiles(projected,geo,cache,s)
        s.update(profiles)
        display,dfile=make_display(projected,geo,cache,args.display_width)
        previews,zfiles=local_views(projected,regions,cache,args.zoom_width)
        atomic_json(manifest,dict(source=str(args.source.resolve()),projected=str(projected.resolve()),
            statistics=str(sfile.resolve()),group_profiles=str(gfile.resolve()),
            display=str(dfile.resolve()),zooms=zfiles,vectors=vector_key,
            bin_degrees=args.bin_degrees,display_width=args.display_width,zoom_width=args.zoom_width))
    draw_figure(s,display,geo,regions,previews,args.figure_output,args.dpi,basin_path=args.basins)
    print('Figures:',args.figure_output,flush=True)


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


if __name__=='__main__':
    main()

