"""Figure 2: surface water persistence."""
from pathlib import Path
from contextlib import ExitStack
import argparse, json, hashlib, os, csv, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window, transform as window_transform
from rasterio.transform import from_origin, from_bounds
from rasterio.features import geometry_mask, rasterize
import geopandas as gpd
from pyproj import CRS, Transformer
from scipy.stats import norm, rankdata
from shapely.geometry import box

FIGURE_OUTPUT = Path(r"D:\phdl2\QTP surface water dynamics\code availability\test_output")
BASE=Path(r'K:\Monthly surface water dynamics TIB\supervised classification\step4_analysis')
REGIONS=Path(r'K:\Monthly surface water dynamics TIB\QTP_shapefiles\selected visualisation region')
BOUNDARY=Path(r'D:\phdl2\QTP surface water dynamics\regions\QTP.shp')
PROJECTED=Path(r'E:\phd_l3\QTP_FH_results\Class_Albers_10m')
NAMES=['Perennial','Wet intermittent','Dry intermittent','Total']
COLORS=['#0000FF','#4169B8','#78BDF5','#888888']


def signature(paths,settings):
    source=[(str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(json.dumps([source,settings],sort_keys=True).encode()).hexdigest()


def atomic_json(path,obj):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def windows(width,height,size=512):
    for r in range(0,height,size):
        for c in range(0,width,size):
            yield Window(c,r,min(size,width-c),min(size,height-r))


def project_one(source,dest,geo,crs,trans,nx,ny,rebuild):
    meta=dest.with_suffix('.json')
    key=signature([source],[crs.to_wkt(),list(trans),nx,ny,geo.geometry.iloc[0].wkb_hex,'10m-nearest-v1'])
    if dest.exists() and meta.exists() and json.loads(meta.read_text())['key']==key and not rebuild:
        print('REUSE projected:',dest.name,flush=True);return
    if dest.exists() and not rebuild:
        raise RuntimeError(f'{dest}: metadata mismatch; not overwriting. Use --rebuild-projection deliberately.')
    temp=dest.with_suffix('.partial.tif')
    profile=dict(driver='GTiff',width=nx,height=ny,count=1,dtype='uint8',crs=crs,
        transform=trans,nodata=255,tiled=True,blockxsize=512,blockysize=512,
        compress='DEFLATE',predictor=2,zlevel=6,BIGTIFF='YES',SPARSE_OK='TRUE')
    boundary=geo.geometry.iloc[0]
    print('CREATE 10 m Albers:',source.name,flush=True)
    total=math.ceil(nx/512)*math.ceil(ny/512)
    with rasterio.Env(GDAL_CACHEMAX=256*1024*1024):
        with rasterio.open(source) as src, rasterio.open(temp,'w',**profile) as dst:
            if src.crs is None:raise ValueError(f'{source}: CRS missing')
            with WarpedVRT(src,crs=crs,transform=trans,width=nx,height=ny,
                    resampling=Resampling.nearest,nodata=255,dtype='float32',warp_mem_limit=128) as vrt:
                for k,win in enumerate(windows(nx,ny)):
                    wb=rasterio.windows.bounds(win,trans)
                    if not boundary.intersects(box(*wb)):continue
                    shape=(int(win.height),int(win.width))
                    inside=geometry_mask([boundary],out_shape=shape,
                        transform=window_transform(win,trans),invert=True)
                    arr=vrt.read(1,window=win,masked=True).filled(255)
                    valid=inside&(arr!=255)
                    if not np.isin(arr[valid],[0,1,2,3]).all():
                        raise ValueError(f'{source}: expected class codes 0–3; check NoData')
                    arr[~inside]=255
                    dst.write(arr.astype('uint8'),1,window=win)
                    if k%500==0:print(f'  blocks {k+1:,}/{total:,}',flush=True)
    os.replace(temp,dest);atomic_json(meta,{'key':key,'source':str(source),'resolution_m':10})


def statistics(paths,geo,cache,step):
    key=signature(paths,[step,geo.geometry.iloc[0].wkb_hex,'areas-profile-union-v1'])
    file=cache/('statistics_'+key[:20]+'.npz')
    if file.exists():
        print('REUSE area statistics:',file.name,flush=True)
        with np.load(file) as z:return {k:z[k] for k in z.files}
    print('CALCULATE native 10 m areas and geographic profiles (one cached scan).',flush=True)
    bounds=geo.to_crs(4326).total_bounds
    lon_edges=np.arange(np.floor(bounds[0]/step)*step,np.ceil(bounds[2]/step)*step+step*1.01,step)
    lat_edges=np.arange(np.floor(bounds[1]/step)*step,np.ceil(bounds[3]/step)*step+step*1.01,step)
    area=np.zeros((11,4));lon_hist=np.zeros((4,len(lon_edges)-1));lat_hist=np.zeros((4,len(lat_edges)-1))
    valid_counts=np.zeros(11,dtype=np.int64);union_area=0.;complete_union_area=0.
    with ExitStack() as stack:
        readers=[stack.enter_context(rasterio.open(p)) for p in paths]
        ref=readers[0];tr=ref.transform;convert=Transformer.from_crs(ref.crs,4326,always_xy=True)
        pixel_area=abs(tr.a*tr.e-tr.b*tr.d)/1e6
        boundary=geo.geometry.iloc[0]
        for k,win in enumerate(windows(ref.width,ref.height)):
            if not boundary.intersects(box(*rasterio.windows.bounds(win,tr))):continue
            blocks=[r.read(1,window=win) for r in readers]
            union=np.zeros(blocks[0].shape,bool);complete=np.ones_like(union)
            for i,a in enumerate(blocks):
                valid_counts[i]+=np.count_nonzero(a!=255)
                area[i,:3]+=np.bincount(a[a<=3],minlength=4)[1:4]*pixel_area
                if i:
                    union|=(a>=1)&(a<=3);complete&=(a!=255)
            masks=[blocks[0]==c for c in (1,2,3)]+[union]
            union_area+=union.sum()*pixel_area;complete_union_area+=(union&complete).sum()*pixel_area
            for c,mask in enumerate(masks):
                rr,cc=np.nonzero(mask)
                if not len(rr):continue
                x=tr.c+(cc+win.col_off+.5)*tr.a
                y=tr.f+(rr+win.row_off+.5)*tr.e
                lon,lat=convert.transform(x,y)
                lon_hist[c]+=np.histogram(lon,lon_edges)[0]*pixel_area
                lat_hist[c]+=np.histogram(lat,lat_edges)[0]*pixel_area
            if k%1000==0:print('  statistics block',k,flush=True)
    area[:,3]=area[:,:3].sum(axis=1)
    result=dict(area=area,lon_edges=lon_edges,lat_edges=lat_edges,lon_hist=lon_hist,
                lat_hist=lat_hist,union_area=np.array(union_area),
                union_complete_area=np.array(complete_union_area),valid_pixels=valid_counts)
    tmp=file.with_suffix('.tmp.npz');np.savez_compressed(tmp,**result);os.replace(tmp,file)
    return result


def classified_profiles(stats):
    """Derive the displayed total from cached class bands; no raster access.

    Do not change the statistics cache signature or overwrite legacy union data.
    """
    result=dict(stats)
    for dim in ('lon','lat'):
        hist=np.array(stats[dim+'_hist'],dtype=float,copy=True)
        hist[3]=hist[:3].sum(axis=0)
        if not np.allclose(hist.sum(axis=1),stats['area'][0],rtol=1e-9,atol=1e-5):
            raise ValueError(f'{dim}: class profiles do not match ten-year map areas')
        result[dim+'_hist']=hist
    return result


def display(path,geo,cache,width):
    key=signature([path],[width,'display-v1'])
    file=cache/('display_'+key[:20]+'.npz')
    if file.exists():
        print('REUSE display:',file.name,flush=True)
        with np.load(file) as z:return z['data'],z['extent']
    with rasterio.open(path) as src:
        ratio=width/max(src.width,src.height);nx=max(1,round(src.width*ratio));ny=max(1,round(src.height*ratio))
        a=src.read(1,out_shape=(ny,nx),resampling=Resampling.nearest)
        b=src.bounds;extent=np.array([b.left,b.right,b.bottom,b.top])
        inside=geometry_mask(geo.geometry,out_shape=(ny,nx),transform=from_bounds(*b,nx,ny),invert=True)
        a[~inside]=254
    tmp=file.with_suffix('.tmp.npz');np.savez_compressed(tmp,data=a,extent=extent);os.replace(tmp,file)
    return a,extent


def fit_annual(y):
    x=np.arange(len(y),dtype=float);X=np.column_stack([np.ones(len(y)),x])
    inv=np.linalg.inv(X.T@X);coef=inv@X.T@y;res=y-X@coef;u=X*res[:,None]
    meat=u.T@u
    # Bartlett HAC lag 1 for annual data. NOT lag 12 from monthly series.
    cross=u[1:].T@u[:-1];meat+=.5*(cross+cross.T)
    cov=inv@meat@inv
    fitted=X@coef;se=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',X,cov,X),0))
    return coef,fitted,1.96*se,hr_mk_test(y)['mk_p']


def clean(ax):
    ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=11.5,length=3,width=.65)
    ax.grid(axis='y',color='#ECEFF0',linewidth=.5);ax.set_axisbelow(True)


def local_views(path,folder,crs,cache,width=1200):
    regions=[];previews=[]
    for n in range(1,7):
        file=folder/f'r{n}.shp'
        frame=gpd.read_file(file)
        if len(frame)!=1 or frame.crs is None or not frame.geometry.is_valid.all():
            raise ValueError(f'{file}: requires one valid polygon with CRS')
        frame=frame.to_crs(crs);bounds=tuple(frame.total_bounds)
        key=signature([path],[bounds,crs.to_wkt(),width,'local-class-v1'])
        cached=cache/('zoom_'+key[:20]+'.npz')
        if cached.exists():
            print('REUSE local view:',n,flush=True)
            with np.load(cached) as z:arr=z['data']
        else:
            print('CREATE local view:',n,flush=True)
            ratio=width/max(bounds[2]-bounds[0],bounds[3]-bounds[1])
            nx=max(1,round((bounds[2]-bounds[0])*ratio));ny=max(1,round((bounds[3]-bounds[1])*ratio))
            trans=from_bounds(*bounds,nx,ny)
            with rasterio.open(path) as src:
                with WarpedVRT(src,crs=crs,transform=trans,width=nx,height=ny,
                        resampling=Resampling.nearest,nodata=255,warp_mem_limit=128) as vrt:
                    arr=vrt.read(1)
            tmp=cached.with_suffix('.tmp.npz');np.savez_compressed(tmp,data=arr,bounds=bounds);os.replace(tmp,cached)
        regions.append(frame);previews.append((arr,bounds))
    return regions,previews


def basin_areas(path,basin_path,crs,cache,qtp_area):
    basins=gpd.read_file(basin_path)
    if 'Name' not in basins or basins.crs is None:raise ValueError('Basin file requires Name and CRS')
    basins=basins.to_crs(crs)
    basins['Name']=basins.Name.astype(str).replace({'KIB':'KlB','QIB':'QlB'})
    basins=basins.dissolve(by='Name').reset_index()
    if not basins.geometry.is_valid.all():raise ValueError('Invalid basin geometry')
    # Reject positive-area overlaps, which would otherwise allocate pixels by draw order.
    for i in range(len(basins)):
        for j in range(i):
            if basins.geometry.iloc[i].intersection(basins.geometry.iloc[j]).area>1.:
                raise ValueError('Basin polygons overlap; resolve overlapping boundaries before area statistics')
    order=['KlB','QdB','QlB','NrB','QtB','YtB','SWB','YB','STB']
    names=[n for n in order if n in set(basins.Name)]+sorted(set(basins.Name)-set(order))
    basins=basins.set_index('Name').loc[names].reset_index()
    key=signature([path],[[(n,g.wkb_hex) for n,g in zip(names,basins.geometry)],'basin-area-v1'])
    dest=cache/('basins_'+key[:20]+'.npz')
    if dest.exists():
        print('REUSE basin areas',flush=True)
        with np.load(dest) as z:area=z['area']
    else:
        print('CALCULATE new basin areas from EXISTING projected mean map (one scan).',flush=True)
        area=np.zeros((len(names),3))
        with rasterio.open(path) as src:
            for k,win in enumerate(windows(src.width,src.height,1024)):
                tile=box(*rasterio.windows.bounds(win,src.transform))
                shapes=[(g,i+1) for i,g in enumerate(basins.geometry) if g.intersects(tile)]
                if not shapes:continue
                arr=src.read(1,window=win)
                if not ((arr>=1)&(arr<=3)).any():continue
                ids=rasterize(shapes,out_shape=arr.shape,transform=window_transform(win,src.transform),
                              fill=0,dtype='uint16')
                for c in (1,2,3):
                    area[:,c-1]+=np.bincount(ids[arr==c],minlength=len(names)+1)[1:]*.0001
                if k%1000==0:print('  basin block',k,flush=True)
        tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,area=area);os.replace(tmp,dest)
    return ['QTP']+names,np.vstack([qtp_area[:3],area])


def panel_letter(fig,x,y,letter,fw,fh):
    """Same physical offset and typography as Figure 3."""
    fig.text(x/fw,y/fh,letter,fontsize=16,fontweight='bold',ha='left',va='bottom')


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


def plot(data,extent,geo,s,out,regions,previews,basin_names,basin_area,basin_path):
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':12.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.7,
        'savefig.facecolor':'white','axes.unicode_minus':True})
    
    
    fw=16.;left=.85;mapw=11.45;gap=.68;rightw=2.35
    maph=mapw*(extent[3]-extent[2])/(extent[1]-extent[0])
    bottom=.66;ch=4.20;cgap=.74;midw=7.7;zgap=.12;zh=(midw-2*zgap)/3
    midh=2*zh+zgap;mgap=.44;topgap=.72;toph=1.35
    midy=bottom+ch+cgap;mapy=midy+midh+mgap;top_y=mapy+maph+topgap
    fh=top_y+toph+.21
    fig=plt.figure(figsize=(fw,fh))
    def axes(x,y,w,h):return fig.add_axes([x/fw,y/fh,w/fw,h/fh])
    ax=axes(left,mapy,mapw,maph);top=axes(left,top_y,mapw,toph)
    right=axes(left+mapw+gap,mapy,rightw,maph)
    legend=axes(left+mapw+gap,top_y,rightw,toph);legend.axis('off')
    cmap=ListedColormap(['white']+COLORS[:3]+['#D8D8D8','white'])
    normc=BoundaryNorm(np.arange(-.5,6),6)
    rendered=np.where(data==254,4,np.where(data==255,5,data))
    ax.imshow(rendered,origin='upper',extent=extent,cmap=cmap,norm=normc,interpolation='none')
    geo.boundary.plot(ax=ax,color='#555555',linewidth=.65)
    ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_aspect('equal');ax.set_axis_off()
    panel_letter(fig,left-.32,mapy+maph+.16,'a',fw,fh)
    add_location_grid(ax,geo.crs,step=5)
    for n,region in enumerate(regions,1):
        region.boundary.plot(ax=ax,color='black',linewidth=1.,zorder=5)
        x0,y0,x1,y1=region.total_bounds
        ax.annotate(str(n),(x1,y1),xytext=(3,3),textcoords='offset points',fontsize=18,
                    fontweight='bold',zorder=6,path_effects=[pe.withStroke(linewidth=2.5,foreground='white')])
    ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:])
    lon=(s['lon_edges'][:-1]+s['lon_edges'][1:])/2;lat=(s['lat_edges'][:-1]+s['lat_edges'][1:])/2
    for k in [3,0,1,2]:
        top.plot(lon,s['lon_hist'][k]/1000,color=COLORS[k],lw=1.6)
        right.plot(s['lat_hist'][k]/1000,lat,color=COLORS[k],lw=1.6)
    clean(top);clean(right);right.xaxis.set_major_locator(MaxNLocator(3))
    top.yaxis.set_major_locator(MaxNLocator(4))
    top.set_xlim(s['lon_edges'][0],s['lon_edges'][-1]);right.set_ylim(s['lat_edges'][0],s['lat_edges'][-1])
    top.set_xlabel('Longitude (°E)',labelpad=4);top.set_ylabel('Area (10³ km²)')
    right.set_xlabel('Area (10³ km²)');right.set_ylabel('Latitude (°N)',labelpad=4)
    legend.legend(handles=[Patch(color=COLORS[k],label=NAMES[k] if k<3 else 'Maximum extent') for k in range(4)],
                  loc='center left',frameon=False,fontsize=12.5,handlelength=1.2,
                  handletextpad=.6,labelspacing=.50,borderaxespad=0)
    pie=ax.inset_axes([.022,.055,2.10/mapw,2.10/maph]);pie.pie(s['area'][0,:3],colors=COLORS[:3],autopct='%1.1f%%',
                  textprops={'fontsize':12.5},wedgeprops={'edgecolor':'white','linewidth':.5})
    pie.text(.5,-.06,f"Total: {s['area'][0,3]:,.0f} km²",transform=pie.transAxes,
             ha='center',va='top',fontsize=12.,clip_on=False)
    for k,(arr,bounds) in enumerate(previews):
        zx=left+(k%3)*(zh+zgap);zy=midy+(1-k//3)*(zh+zgap)
        zax=axes(zx,zy,zh,zh)
        zax.imshow(np.where(arr==255,0,arr),extent=[bounds[0],bounds[2],bounds[1],bounds[3]],
                   origin='upper',cmap=cmap,norm=normc,interpolation='none')
        zax.set_xticks([]);zax.set_yticks([]);zax.set_aspect('equal')
        zax.text(.03,.97,str(k+1),transform=zax.transAxes,va='top',fontsize=18,fontweight='bold',
                  bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1.5})
    bx=axes(left+midw+.85,midy,mapw+gap+rightw-midw-.85,midh)
    
    keep=np.array([name!='QTP' for name in basin_names])
    basin_area=basin_area[keep]
    basin_names=[name for name in basin_names if name!='QTP']
    clean(bx);ys=np.arange(len(basin_names));running=np.zeros(len(ys))
    for k in range(3):
        v=basin_area[:,k]/1000;bx.barh(ys,v,left=running,color=COLORS[k],height=.65,label=NAMES[k]);running+=v
    bx.set_yticks(ys,basin_names,fontsize=12.5);bx.invert_yaxis();bx.grid(False)
    bx.tick_params(axis='y',length=0)
    bx.set_xlabel('Water area (10³ km²)');bx.xaxis.set_major_locator(MaxNLocator(4))
    panel_letter(fig,left+midw+.85-.32,midy+midh+.16,'b',fw,fh)
    for y,v in zip(ys,running):bx.text(v+max(running)*.012,y,f'{v:.1f}',va='center',fontsize=11.5)
    bx.set_xlim(0,max(running)*1.17)
    years=np.arange(2016,2026);trend_rows=[]
    totalw=mapw+gap+rightw;column_gap=.95;row_gap=.72
    cw=(totalw-column_gap)/2;rowh=(ch-row_gap)/2
    panel_letter(fig,left-.32,bottom+ch+.14,'c',fw,fh)
    for k in range(4):
        tx=left+(k%2)*(cw+column_gap);ty=bottom+(1-k//2)*(rowh+row_gap)
        bx=axes(tx,ty,cw,rowh);clean(bx);y=s['area'][1:,k]
        coef,pred,ci,p=fit_annual(y)
        bx.plot(years,y/1000,color=COLORS[k],marker='o',ms=4.,lw=1.5)
        bx.plot(years,pred/1000,color=COLORS[k],ls='--',lw=1)
        bx.fill_between(years,(pred-ci)/1000,(pred+ci)/1000,color=COLORS[k],alpha=.16,lw=0)
        stars=('**' if p<.01 else '*' if p<.05 else '') if np.isfinite(p) else ''
        bx.text(.015,.96,f'{NAMES[k]}   Slope = {coef[1]:+.1f} km² yr⁻¹{stars}',
                   transform=bx.transAxes,va='top',fontsize=12.)
        bx.margins(y=.38);bx.yaxis.set_major_locator(MaxNLocator(3))
        bx.set_xlim(2015.8,2025.2);bx.set_xticks(years[::2]);bx.set_ylabel('Area (10³ km²)')
        trend_rows.append([NAMES[k],coef[1],p])
    add_basin_boundaries_and_labels(ax,basin_path,geo.crs,regions)
    for ext in ['png','svg']:fig.savefig(out/f'Figure2_water_types.{ext}',dpi=300,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)


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
    """Hamed–Rao MK on annual class area; all significant ACF lags, lag=None.

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
    # If the detrended ranks are constant, there is no residual ACF correction.
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

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=BASE/'QTP_FH_results')
    p.add_argument('--projected-dir',type=Path,default=PROJECTED)
    p.add_argument('--output',type=Path,default=BASE/'QTP_Figure2_water_types',help='Directory for existing cache files')
    p.add_argument('--figure-output',type=Path,default=FIGURE_OUTPUT)
    p.add_argument('--boundary',type=Path,default=BOUNDARY)
    p.add_argument('--regions-dir',type=Path,default=REGIONS)
    p.add_argument('--basins',type=Path,default=BOUNDARY.with_name('QTP_watershed.shp'))
    p.add_argument('--crs-reference',type=Path,default=REGIONS/'r1.shp')
    p.add_argument('--bin-degrees',type=float,default=.1)
    p.add_argument('--display-width',type=int,default=5000)
    p.add_argument('--rebuild-projection',action='store_true')
    args=p.parse_args()
    if not .05<=args.bin_degrees<=2 or not 500<=args.display_width<=12000:p.error('Invalid bin/display size')
    target=gpd.read_file(args.crs_reference,rows=1).crs
    if target is None or not target.is_projected:raise ValueError('Projected Albers CRS required')
    if 'albers' not in target.coordinate_operation.method_name.lower():raise ValueError('Reference must use Albers equal-area')
    if any(abs(a.unit_conversion_factor-1)>1e-8 for a in target.axis_info):raise ValueError('Metre units required')
    geo=gpd.read_file(args.boundary).to_crs(target).dissolve()
    if geo.empty or not geo.geometry.is_valid.all():raise ValueError('Invalid QTP boundary')
    xmin,ymin,xmax,ymax=geo.total_bounds
    xmin=math.floor(xmin/10)*10;ymin=math.floor(ymin/10)*10
    xmax=math.ceil(xmax/10)*10;ymax=math.ceil(ymax/10)*10
    nx=int(round((xmax-xmin)/10));ny=int(round((ymax-ymin)/10));trans=from_origin(xmin,ymax,10,10)
    sources=[args.input_dir/'multi_year_mean_and_transition'/'Class_mean_2016_2025_revised.tif']
    sources += [args.input_dir/str(y)/f'Class_{y}.tif' for y in range(2016,2026)]
    for f in sources:
        if not f.is_file():raise FileNotFoundError(f)
    args.projected_dir.mkdir(parents=True,exist_ok=True);args.output.mkdir(parents=True,exist_ok=True)
    cache=args.output/'cache';cache.mkdir(exist_ok=True)
    args.figure_output.mkdir(parents=True,exist_ok=True)
    projected=[]
    for source in sources:
        dest=args.projected_dir/(source.stem+'_Albers10m.tif')
        project_one(source,dest,geo,target,trans,nx,ny,args.rebuild_projection);projected.append(dest)
    s=statistics(projected,geo,cache,args.bin_degrees)
    s=classified_profiles(s)
    a,extent=display(projected[0],geo,cache,args.display_width)
    regions,previews=local_views(projected[0],args.regions_dir,target,cache)
    basin_names,basin_area=basin_areas(projected[0],args.basins,target,cache,s['area'][0])
    plot(a,extent,geo,s,args.figure_output,regions,previews,basin_names,basin_area,args.basins)
    print('Figures:',args.figure_output)

if __name__=='__main__':main()

