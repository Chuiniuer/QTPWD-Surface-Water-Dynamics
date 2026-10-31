"""Figure 1: monthly surface water dynamics."""
from pathlib import Path
import argparse, json, warnings
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib import font_manager
from scipy.stats import norm, rankdata

FIGURE_OUTPUT = Path(r"D:\phdl2\QTP surface water dynamics\code availability\test_output")
BASE=Path(r"K:\Monthly surface water dynamics TIB\supervised classification\step4_analysis")
BOUNDARY=Path(r"D:\phdl2\QTP surface water dynamics\regions\QTP.shp")
DATES=pd.date_range("2016-01-01","2025-12-01",freq="MS")
COLS={"Total":"Total_surface_water_area_km2","Small":"Small_waterbody_area_km2",
      "Medium":"Medium_waterbody_area_km2","Large":"Large_waterbody_area_km2"}
COLORS={"Total":"#888888","Small":"#87C39D","Medium":"#91CADA","Large":"#F2D18E"}


def read_monthly(path):
    df=pd.read_csv(path,encoding="utf-8-sig")
    for col in ["Year","Month"]+list(COLS.values()):
        df[col]=pd.to_numeric(df[col],errors="raise")
    df["Date"]=pd.to_datetime(dict(year=df.Year,month=df.Month,day=1))
    df=df.sort_values("Date")
    if not pd.DatetimeIndex(df.Date).equals(DATES):raise ValueError("Monthly dates incomplete or duplicated")
    a=df[list(COLS.values())].to_numpy(float)
    if not np.isfinite(a).all() or (a<0).any():raise ValueError("Invalid monthly area")
    if not np.allclose(a[:,0],a[:,1:].sum(1),rtol=1e-10,atol=1e-7):
        raise ValueError("Size areas do not sum to total")
    return df.reset_index(drop=True)


def existing_results(args):
    if args.results_dir:
        candidates=[args.results_dir]
    else:
        candidates=[p for p in args.base.glob("QTP_trend_extremes*") if p.is_dir()]
    candidates=[p for p in candidates if (p/"Regional_summary.csv").is_file() and
                (p/"Regional_monthly_components.csv").is_file()]
    if not candidates:
        raise FileNotFoundError("Existing regional results missing; specify --results-dir. This plotting script does not rerun STL.")
    folder=max(candidates,key=lambda p:(p/"Regional_monthly_components.csv").stat().st_mtime_ns)
    print("Read existing regional results:",folder)
    summary=pd.read_csv(folder/"Regional_summary.csv",encoding="utf-8-sig")
    comp=pd.read_csv(folder/"Regional_monthly_components.csv",encoding="utf-8-sig")
    comp["Date"]=pd.to_datetime(comp.Date)
    return folder,summary,comp

def grid_analysis(args):
    import geopandas as gpd
    folder=args.grid_results or args.output
    path=folder/"QTP_grid_centres.shp"
    if not path.is_file():
        raise FileNotFoundError(f"Existing grid centres missing: {path}. Set --grid-results to the previous Figure1 output directory; no raster/STL recomputation is performed.")
    points=gpd.read_file(path)
    needed={"mean_pct","pp_dec","mk_p","mean_km2"}
    if not needed<=set(points):raise ValueError("Grid attributes missing: "+str(needed-set(points)))
    if points.crs is None:raise ValueError("Grid centre CRS missing")
    points["x"]=points.geometry.x;points["y"]=points.geometry.y
    boundary=gpd.read_file(args.boundary).to_crs(points.crs)
    print("Read existing grid centres:",path)
    return None,None,None,None,boundary,points


FONT = dict(body=11.5, tick=10.5, note=10.5, inset=9.0, small=8.0, panel=16.0)


def style(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(length=3,width=.65,labelsize=FONT["tick"])
    ax.grid(axis="y",color="#eeeeee",lw=.5)
    ax.set_axisbelow(True)

def label(ax,s):
    
    from matplotlib.transforms import ScaledTranslation
    offset=ScaledTranslation(-.32,.16,ax.figure.dpi_scale_trans)
    return ax.text(0,1,s,transform=ax.transAxes+offset,fontweight="bold",
                   fontsize=FONT["panel"],ha="left",va="bottom",clip_on=False)


def fit_trend_labels(ax, annotations):
    """Nudge number labels within their column; retain the original trend scales."""
    for _ in range(2):
        ax.figure.canvas.draw()
        renderer=ax.figure.canvas.get_renderer()
        bounds=ax.get_window_extent(renderer)
        for text in annotations:
            bb=text.get_window_extent(renderer)
            shift=max(0.,bounds.x0+2-bb.x0)-max(0.,bb.x1-bounds.x1+2)
            if abs(shift)>.2:
                x,y=text.get_position()
                text.set_position((x+shift*72/ax.figure.dpi,y))


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


def draw_maps(fig,axes,grid):
    import geopandas as gpd
    mean,slope,p,extent,boundary,df=grid
    
    df=gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.x,df.y),crs=boundary.crs)
    limits=[[0,1,5,10,25,100],
            [-5,-.5,0,.5,5]]
    cmaps=[LinearSegmentedColormap.from_list("water",["#edf5f4","#b9ddd8","#91CADA","#559CAF","#245675"],N=5),
           ListedColormap(["#CF929D","#EED2D7","#D0E8EB","#8EC2CE"],name="trend")]
    cmaps[1].set_under("#A75E73");cmaps[1].set_over("#4F8FA5")
    names=["Water area percentage (%)","Water area percentage trend\n(% per decade)"]
    for j,(ax,col) in enumerate(zip(axes,["mean_pct","pp_dec"])):
        boundary.plot(ax=ax,color="#eeeeee",edgecolor="none",zorder=0)
        xmin,ymin,xmax,ymax=boundary.total_bounds
        ax.set_xlim(xmin-12000,xmax+12000);ax.set_ylim(ymin-12000,ymax+12000)
        ax.set_aspect("equal");ax.set_anchor("NW");ax.set_axis_off();label(ax,"bc"[j])
        normer=BoundaryNorm(limits[j],cmaps[j].N,clip=False)
        artist=ax.scatter(df.geometry.x,df.geometry.y,c=df[col],marker="s",s=9,
                          cmap=cmaps[j],norm=normer,linewidths=0,zorder=2)
        add_location_grid(ax,boundary.crs,step=5)
        boundary.boundary.plot(ax=ax,color="#333333",linewidth=.55,zorder=4)
        cb=fig.colorbar(artist,ax=ax,pad=.016,fraction=.027,shrink=.85,panchor=False,
                        spacing="uniform",extend="neither" if j==0 else "both")
        ticks=limits[j]
        cb.set_ticks(ticks);cb.set_ticklabels([f"{v:g}" for v in ticks]);cb.ax.minorticks_off()
        cb.set_label(names[j],fontsize=FONT["body"],labelpad=8);cb.ax.tick_params(labelsize=FONT["tick"],length=2.5)
        fig.canvas.draw()
        side=ax.get_window_extent().width/(xmax-xmin+24000)*25000*72/fig.dpi
        artist.set_sizes([(.94*side)**2])
        if j==1:
            sig=df[(df.mk_p<.05)&(df.mean_km2>0)]
            ax.scatter(sig.geometry.x,sig.geometry.y,s=.65,c="#303030",linewidths=0,zorder=3)
        pie=ax.inset_axes([.005,-.055,.29,.32],zorder=8)
        values=df[col].to_numpy(float)
        values=values[np.isfinite(values)]
        if j==0:
            counts,_=np.histogram(values,bins=limits[j])
            colors=[cmaps[j](i) for i in range(cmaps[j].N)]
        else:
            
            codes=np.asarray(normer(values),dtype=int)
            counts=np.array([(codes==i).sum() for i in [-1,0,1,2,3,4]])
            colors=[cmaps[j].get_under()]+[cmaps[j](i) for i in range(4)]+[cmaps[j].get_over()]
        if counts.sum():
            # Select by category rank, not a percentage threshold; ties follow colour order.
            top=set(np.argsort(-counts,kind="stable")[:min(3,np.count_nonzero(counts))])
            wedges,texts,pct_texts=pie.pie(counts,colors=colors,startangle=90,
                    autopct="%.0f%%",pctdistance=.72,
                    textprops=dict(fontsize=FONT["inset"]),wedgeprops=dict(edgecolor="white",linewidth=.35))
            for idx,txt in enumerate(pct_texts):
                if idx not in top:txt.set_text("")


def ols_hac(y, maxlags=12):
    """OLS with Bartlett/Newey-West covariance, no small-sample correction.
    Equivalent to OLS.fit(cov_type="HAC", cov_kwds={"maxlags":12}).
    Uses NumPy only; no patsy/formulaic dependency.
    """
    y=np.asarray(y,dtype=float)
    if y.ndim!=1 or len(y)<3 or not np.isfinite(y).all():
        raise ValueError("OLS requires a finite one-dimensional series")
    x=np.column_stack((np.ones(len(y)),np.arange(len(y))/12))
    pinv=np.linalg.pinv(x)
    params=pinv@y
    residual=y-x@params
    scores=x*residual[:,None]
    meat=scores.T@scores
    for lag in range(1,min(maxlags,len(y)-1)+1):
        cross=scores[lag:].T@scores[:-lag]
        meat+=(1-lag/(maxlags+1))*(cross+cross.T)
    bread=pinv@pinv.T
    cov=bread@meat@bread
    cov=(cov+cov.T)/2
    return x,params,cov

def trend_band(y):
    """Conditional 95% HAC confidence band for a fixed STL trend."""
    x,params,cov=ols_hac(y)
    pred=x@params
    se=np.sqrt(np.maximum(np.einsum("ij,jk,ik->i",x,cov,x),0))
    z=norm.ppf(.975)
    return pred,pred-z*se,pred+z*se


def counts(df):
    required=["Below_0p1_patch_count","Small_waterbody_count","Medium_waterbody_count","Large_waterbody_count","All_water_patch_count"]
    if not set(required)<=set(df):
        raise ValueError("Count fields missing: "+str(set(required)-set(df)))
    v=df[required].apply(pd.to_numeric,errors="raise")
    if not np.isfinite(v).all().all() or (v<0).any().any() or (v!=np.floor(v)).any().any():
        raise ValueError("Invalid patch counts")
    if not np.array_equal(v.iloc[:,:4].sum(1).to_numpy(),v.All_water_patch_count.to_numpy()):
        raise ValueError("Count schema differs: small count + below 0.1 + medium + large must equal all patches")
    return pd.DataFrame({"Year":df.Year,"Small":v.Small_waterbody_count,
                         "Medium":v.Medium_waterbody_count,"Large":v.Large_waterbody_count})

def basin_data(args):
    import geopandas as gpd
    path=args.watershed_monthly or args.base/"QTP_waterbody_statistics_0p1"/"Watershed_monthly_waterbody_statistics.csv"
    raw=pd.read_csv(path,encoding="utf-8-sig")
    raw["Date"]=pd.to_datetime(dict(year=raw.Year,month=raw.Month,day=1))
    rows=[]
    summary=args.regional_summary
    for name,g in raw.groupby("Name"):
        g=g.sort_values("Date").reset_index(drop=True)
        if not pd.DatetimeIndex(g.Date).equals(DATES):raise ValueError(f"{name}: incomplete monthly records")
        n=counts(g)
        areas=g[list(COLS.values())].to_numpy(float)
        if not np.isfinite(areas).all() or not np.allclose(areas[:,0],areas[:,1:].sum(1)):
            raise ValueError(f"{name}: invalid area partition")
        for k in ["Small","Medium","Large","Total"]:
            match=summary[(summary.Region.astype(str)==str(name))&(summary.SizeClass==k)]
            if len(match)!=1:raise ValueError(f"Existing basin trend missing or duplicated: {name}/{k}")
            beta=float(match.iloc[0].beta_yr)
            if not np.isclose(match.iloc[0].mean_km2,g[COLS[k]].mean(),rtol=1e-8,atol=1e-6):
                raise ValueError(f"{name}/{k}: existing results do not match monthly table")
            rows.append(dict(Name=str(name),SizeClass=k,MeanTotal_km2=areas[:,0].mean(),
                TotalSlope=float(summary[(summary.Region.astype(str)==str(name))&(summary.SizeClass=="Total")].iloc[0].beta_yr),
                MeanArea_km2=g[COLS[k]].mean(),AreaShare=100*g[COLS[k]].mean()/areas[:,0].mean(),
                MeanCount=(n[k].mean() if k!="Total" else n[["Small","Medium","Large"]].sum(axis=1).mean()),
                CountShare=(100*n[k].mean()/n[["Small","Medium","Large"]].sum(axis=1).mean() if k!="Total" else 100),
                Slope=beta,P=args.significance.get((str(name),k),np.nan)))
    stats=pd.DataFrame(rows)
    def canonical(v):return str(v).strip().replace("QIB","QlB").replace("KIB","KlB")
    wanted={canonical(v) for v in stats.Name}
    candidates=[args.basins] if args.basins else sorted(args.boundary.parent.glob("*.shp"))
    matches=[]
    for f in candidates:
        geo=gpd.read_file(f)
        for col in ([args.basin_field] if args.basin_field else list(geo.columns)):
            if col not in geo or col=="geometry":continue
            values=geo[col].map(canonical)
            if wanted<=set(values):
                geo=geo.copy();geo["join_name"]=values
                matches.append((f,col,geo[values.isin(wanted)].dissolve(by="join_name").reset_index()))
                break
    if len(matches)!=1:
        raise ValueError("Cannot uniquely resolve basin boundaries. Supply --basins path/to/basins.shp --basin-field Name. Matches: "+str([(str(f),c) for f,c,_ in matches]))
    f,col,geo=matches[0]
    if geo.crs is None:raise ValueError("Basin boundary CRS missing")
    print("Basin boundary:",f,"name field:",col)
    stats["join_name"]=stats.Name.map(canonical)
    return stats,geo

def basin_panels(fig,dspec,espec,stats,basins,boundary):
    from matplotlib.patches import Patch
    keys=["Small","Medium","Large"]
    order=["KlB","QdB","QlB","NrB","QtB","YtB","SWB","YB","STB"]
    order=[k for k in order if k in set(stats.join_name)]
    order+=sorted(set(stats.join_name)-set(order))
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    ax=fig.add_subplot(dspec)
    geo=basins.to_crs(boundary.crs).merge(stats.groupby("join_name").MeanTotal_km2.first(),left_on="join_name",right_index=True)
    cmap=LinearSegmentedColormap.from_list("basin_area",[(0.,"#EBF2F5"),(.04,"#CFDFE9"),(.10,"#AAC8D6"),(.18,"#9CCDC8"),(.28,"#C4D9B5"),(.55,"#E0DFBC"),(1.,"#EBD49C")],N=256)
    
    normalization=Normalize(0,float(geo.MeanTotal_km2.max()))
    geo.plot(ax=ax,column="MeanTotal_km2",cmap=cmap,norm=normalization,edgecolor="#7b8d91",linewidth=.5)
    boundary.boundary.plot(ax=ax,color="#49575b",linewidth=.55)
    xmin,ymin,xmax,ymax=boundary.total_bounds
    ax.set_xlim(xmin,xmax);ax.set_ylim(ymin-(ymax-ymin)*.23,ymax);ax.set_aspect("equal");ax.set_anchor("SW");ax.axis("off")
    slot=dspec.get_position(fig)
    fig.text(slot.x0-.32/fig.get_figwidth(),slot.y1+.16/fig.get_figheight(),
             "d",fontsize=FONT["panel"],fontweight="bold",ha="left",va="bottom")
    fig.canvas.draw()
    for _,r in geo.iterrows():
        pt=r.geometry.representative_point()
        x,y=ax.transAxes.inverted().transform(ax.transData.transform((pt.x,pt.y)))
        sub=stats[(stats.join_name==r.join_name)&stats.SizeClass.isin(keys)].set_index("SizeClass").loc[keys]
        ins=ax.inset_axes([x-.045,y-.065,.09,.13])
        ins.pie(sub.AreaShare,colors=[COLORS[k] for k in keys],startangle=90,wedgeprops=dict(edgecolor="white",linewidth=.3))
        ins.set_title(sub.Name.iloc[0],fontsize=7.5,pad=0)
    cb=fig.colorbar(ScalarMappable(norm=normalization,cmap=cmap),ax=ax,fraction=.027,pad=.015,shrink=.7,panchor=False)
    cb.ax.minorticks_off()
    cb.set_label("Mean water area (km²)",fontsize=9,labelpad=3);cb.ax.tick_params(labelsize=7,length=2,pad=2)
    pie=ax.inset_axes([.00,.005,.16,.20])
    pie.pie([stats.attrs["qtp_area"][k] for k in keys],colors=[COLORS[k] for k in keys],startangle=90,
        autopct="%.0f%%",textprops=dict(fontsize=6.5),wedgeprops=dict(edgecolor="white",linewidth=.3))
    pie.set_title("QTP area share",fontsize=7,pad=0)
    bar=ax.inset_axes([.22,.025,.25,.18]);counts=stats.attrs["qtp_count"]
    bar.bar(range(3),[counts[k] for k in keys],color=[COLORS[k] for k in keys],width=.6)
    bar.set_yscale("log");bar.set_ylim(.8,max(counts.values())*5)
    bar.set_xticks(range(3));bar.set_xticklabels(keys,fontsize=6.5)
    bar.set_yticks([1,100,10000]);bar.set_yticklabels(["1","10²","10⁴"],fontsize=6.5)
    bar.minorticks_off();bar.tick_params(length=1,pad=1);bar.spines[["top","right"]].set_visible(False)
    for i,k in enumerate(keys):bar.text(i,counts[k]*1.2,f"{counts[k]:,.0f}",ha="center",fontsize=6.5)
    bar.set_title("QTP mean count",fontsize=7,pad=1)
    handles=[Patch(color=COLORS[k],label=k) for k in keys]
    ax.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,-.04),ncol=3,frameon=False,fontsize=9,handlelength=1.1,handletextpad=.4,columnspacing=.8)
    keys=["Small","Medium","Large","Total"]
    stats=pd.concat([stats,pd.DataFrame(stats.attrs["qtp_trends"])],ignore_index=True)
    order=["QTP"]+order
    epair=espec.subgridspec(1,4,wspace=.18)
    eax=[fig.add_subplot(epair[i]) for i in range(4)]
    y=np.arange(len(order))
    for j,(ax,k) in enumerate(zip(eax,keys)):
        v=np.array([stats[(stats.join_name==n)&(stats.SizeClass==k)].Slope.iloc[0] for n in order])
        lo=min(0,float(v.min()));hi=max(0,float(v.max()));span=max(hi-lo,1)
        ax.axvline(0,color="#9a9a9a",lw=.6)
        ax.hlines(y,0,v,color=COLORS[k],lw=1.2)
        ax.scatter(v,y,s=21,color=COLORS[k],edgecolor="white",linewidth=.4,zorder=3)
        annotations=[]
        for yy,vv,n in zip(y,v,order):
            pv=stats[(stats.join_name==n)&(stats.SizeClass==k)].P.iloc[0]
            mark="**" if np.isfinite(pv) and pv<.01 else "*" if np.isfinite(pv) and pv<.05 else ""
            annotations.append(ax.annotate(f"{vv:+.1f}{mark}",(vv,yy),xytext=(0,5),
                textcoords="offset points",ha="center",va="bottom",fontsize=6.5))
        ax.set_xlim(lo-.25*span,hi+.25*span);ax.set_ylim(len(order)-.35,-.9)
        ax.set_yticks(y);ax.set_yticklabels(order if j==0 else [""]*len(order))
        ax.set_title(k,fontsize=11,pad=7);ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0,labelsize=9.5);ax.tick_params(axis="x",labelsize=8.5,length=2)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(3))
        fit_trend_labels(ax,annotations)
    left=eax[0].get_position().x0;right=eax[-1].get_position().x1
    bottom=eax[0].get_position().y0
    fig.text((left+right)/2,bottom-.025,"Long-term area trend (km² yr⁻¹)",ha="center",va="top",fontsize=10.5)
    label(eax[0],"e")
    

def monthly_distributions(fig,spec,df,cnt):
    panels=spec.subgridspec(2,1,height_ratios=[1.35,1],hspace=.25)
    ax=fig.add_subplot(panels[0])
    cycle_ax=fig.add_subplot(panels[1],sharex=ax)
    vals=df[COLS["Total"]].to_numpy()/1000
    groups=[vals[np.asarray(df.Month)==m] for m in range(1,13)]
    bp=ax.boxplot(groups,widths=.6,patch_artist=True,showfliers=False,
        medianprops=dict(color="#345761",linewidth=1),
        whiskerprops=dict(color="#74888f",linewidth=.65),
        capprops=dict(color="#74888f",linewidth=.65))
    for body in bp["boxes"]:body.set(facecolor="#b9dde5",edgecolor="#74888f",linewidth=.65)
    for m,v in enumerate(groups,1):
        ax.scatter(m+np.linspace(-.14,.14,len(v)),v,s=6,color="#66777d",alpha=.5,zorder=3,linewidths=0)
    months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xlim(.4,12.6);ax.set_ylabel("Water area (10³ km²)",fontsize=9.5)
    ax.tick_params(axis="x",labelbottom=False)
    style(ax);ax.tick_params(axis="y",labelsize=FONT["tick"]);label(ax,"f")
    # Calendar-month climatology of observed area, NOT the STL seasonal component.
    for k in ["Small","Medium","Large"]:
        climatology=df.groupby("Month")[COLS[k]].mean().reindex(range(1,13))
        average=float(df[COLS[k]].mean())
        if average<=0:raise ValueError(f"Cannot normalize a zero-mean area series: {k}")
        relative=100*(climatology.to_numpy()/average-1)
        by_month=df.groupby("Month")[COLS[k]]
        lower=100*(by_month.quantile(.25).reindex(range(1,13)).to_numpy()/average-1)
        upper=100*(by_month.quantile(.75).reindex(range(1,13)).to_numpy()/average-1)
        # Distribution of ten observations per calendar month, not a confidence interval.
        cycle_ax.fill_between(range(1,13),lower,upper,color=COLORS[k],alpha=.22,linewidth=0,zorder=1)
        cycle_ax.plot(range(1,13),relative,color=COLORS[k],lw=1.4,marker="o",markersize=3,label=k)
        peak=int(np.argmax(relative))
        cycle_ax.scatter(peak+1,relative[peak],s=18,color=COLORS[k],edgecolor="#617078",linewidth=.35,zorder=4)
        cycle_ax.annotate(months[peak],(peak+1,relative[peak]),xytext=(3,5),
            textcoords="offset points",fontsize=9.5,ha="left",va="bottom",color="#48585f")
    cycle_ax.axhline(0,color="#9aa5a9",lw=.55,ls="--")
    cycle_ax.set_ylabel("Relative area deviation (%)",fontsize=9.5)
    cycle_ax.set_xticks(range(1,13));cycle_ax.set_xticklabels(months,rotation=0,ha="center",fontsize=FONT["tick"])
    cycle_ax.margins(y=.38)
    style(cycle_ax);cycle_ax.tick_params(axis="y",labelsize=FONT["tick"])
    
    fig.canvas.draw()
    upper_label=ax.yaxis.label
    upper_display=upper_label.get_transform().transform(upper_label.get_position())
    lower_x=cycle_ax.transAxes.inverted().transform(upper_display)[0]
    cycle_ax.yaxis.set_label_coords(lower_x,.5)
    cycle_ax.legend(loc="upper right",ncol=3,frameon=False,fontsize=9.5,
        handlelength=1.1,columnspacing=.6,handletextpad=.3,borderaxespad=.1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base",type=Path,default=BASE)
    ap.add_argument("--monthly",type=Path)
    ap.add_argument("--watershed-monthly",type=Path)
    ap.add_argument("--basins",type=Path,default=BOUNDARY.parent/"QTP_watershed.shp")
    ap.add_argument("--basin-field",default="Name")
    ap.add_argument("--results-dir",type=Path,help="Existing extremes folder with Regional_summary.csv and Regional_monthly_components.csv")
    ap.add_argument("--grid-results",type=Path,help="Existing Figure1 folder containing Grid_metrics.csv and QTP_grid_centres.shp")
    ap.add_argument("--grid-dir",type=Path)
    ap.add_argument("--boundary",type=Path,default=BOUNDARY)
    ap.add_argument("--output",type=Path,help="Directory containing existing figure inputs")
    ap.add_argument("--figure-output",type=Path,default=FIGURE_OUTPUT)
    ap.add_argument("--min-valid",type=float,default=.9)
    args=ap.parse_args()
    if not 0<=args.min_valid<=1:ap.error("--min-valid must be in [0,1]")
    args.monthly=args.monthly or args.base/"QTP_waterbody_statistics_0p1"/"QTP_monthly_waterbody_statistics.csv"
    args.grid_dir=args.grid_dir or args.base/"QTP_water_area_25km_QTP_Albers"/"monthly"
    args.output=args.output or args.base/"QTP_Figure1_spatiotemporal"
    df=read_monthly(args.monthly);cnt=counts(df)
    args.figure_output.mkdir(parents=True,exist_ok=True)
    source,summary,existing_comp=existing_results(args)
    args.regional_summary=summary
    
    args.significance={}
    for (region,size),block in existing_comp.groupby(["Region","SizeClass"]):
        source_row=summary[(summary.Region==region)&(summary.SizeClass==size)]
        pv=float(source_row.iloc[0].get("mk_p",np.nan)) if len(source_row)==1 else np.nan
        if not np.isfinite(pv):
            pv=hr_mk_test(block.sort_values("Date").Trend_km2.to_numpy()).get("mk_p",np.nan)
        args.significance[(str(region),size)]=pv
    fits={};metrics=[];components=[]
    old_metrics_path=args.output/"Size_STL_metrics.csv"
    old_metrics=pd.read_csv(old_metrics_path) if old_metrics_path.is_file() else None
    for name,col in COLS.items():
        block=existing_comp[(existing_comp.Region=="QTP")&(existing_comp.SizeClass==name)].sort_values("Date")
        if not pd.DatetimeIndex(block.Date).equals(DATES):raise ValueError(f"Existing QTP/{name}: incomplete components")
        y=df[col].to_numpy()
        if not np.allclose(block.Area_km2,y,rtol=1e-9,atol=1e-6):
            raise ValueError(f"Existing QTP/{name} components differ from input monthly areas")
        row=summary[(summary.Region=="QTP")&(summary.SizeClass==name)]
        if len(row)!=1:raise ValueError(f"Missing/duplicate QTP/{name} summary")
        row=row.iloc[0]
        fit=SimpleNamespace(trend=block.Trend_km2.to_numpy(),seasonal=block.Seasonal_km2.to_numpy(),resid=block.Residual_km2.to_numpy())
        fits[name]=fit
        pv=args.significance.get(("QTP",name),np.nan)
        if old_metrics is not None:
            previous=old_metrics[old_metrics.SizeClass==name]
            if len(previous)==1 and np.isclose(previous.iloc[0].Trend_km2_yr,row.beta_yr,rtol=1e-8,atol=1e-6):
                pv=previous.iloc[0].HR_MK_p
        metrics.append(dict(SizeClass=name,MeanArea_km2=row.mean_km2,
            AreaShare_pct=100*row.mean_km2/df[COLS["Total"]].mean(),Trend_km2_yr=row.beta_yr,
            Trend_km2_decade=10*row.beta_yr,HR_MK_p=pv,
            SeasonalSD_km2=row.s_sd_km2,ResidualSD_km2=row.r_sd_km2))
        components.append(block[["Date","SizeClass","Area_km2","Trend_km2","Seasonal_km2","Residual_km2"]])
    metrics=pd.DataFrame(metrics);comp=pd.concat(components,ignore_index=True)
    
    annual=df.groupby("Year")[list(COLS.values())].mean()
    countannual=cnt.groupby("Year").mean()
    climate=comp.assign(Month=comp.Date.dt.month).groupby(["SizeClass","Month"]).Seasonal_km2.agg(["mean","std"]).reset_index()
    basin_stats,basins=basin_data(args)
    basin_stats.attrs["qtp_area"]={k:float(df[COLS[k]].mean()) for k in ["Small","Medium","Large"]}
    basin_stats.attrs["qtp_count"]={k:float(cnt[k].mean()) for k in ["Small","Medium","Large"]}
    basin_stats.attrs["qtp_trends"]=[dict(join_name="QTP",SizeClass=r.SizeClass,Slope=r.Trend_km2_yr,P=r.HR_MK_p) for _,r in metrics.iterrows()]
    grid=grid_analysis(args)
    if "Times New Roman" not in {f.name for f in font_manager.fontManager.ttflist}:
        warnings.warn("Times New Roman unavailable; Matplotlib will use a fallback.")
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
                         "font.size":FONT["body"],"axes.labelsize":FONT["body"],
                         "xtick.labelsize":FONT["tick"],"ytick.labelsize":FONT["tick"],
                         "legend.fontsize":FONT["body"],"axes.linewidth":.7,"savefig.facecolor":"white",
                         "svg.fonttype":"none","pdf.fonttype":42,"axes.unicode_minus":True})
    fig=plt.figure(figsize=(16,10))
    outer=fig.add_gridspec(2,1,height_ratios=[1.65,1],left=.045,right=.97,bottom=.055,top=.97,hspace=.20)
    upper=outer[0].subgridspec(1,3,width_ratios=[1.15,1.15,1.05],wspace=.19)
    bottom=outer[1].subgridspec(1,3,width_ratios=[1.15,1.15,1.05],wspace=.23)
    agrid=upper[:2].subgridspec(4,1,hspace=.12)
    axes=[fig.add_subplot(agrid[k]) for k in range(4)]
    total=fits["Total"]
    vals=[df[COLS["Total"]].to_numpy(),total.trend,total.seasonal,total.resid]
    for i,(ax,y,nm) in enumerate(zip(axes,vals,["Observed","Trend","Seasonal","Residual"])):
        if i==3:
            ax.scatter(DATES,y/1000,color="#BE727E",s=7,alpha=.85,linewidths=0)
        else:
            ax.plot(DATES,y/1000,color=["#87929C","#333333","#559CAF"][i],lw=1)
        ax.set_ylabel(nm+"\n(10³ km²)");style(ax);ax.set_xlim(DATES[0],DATES[-1])
        if i<3:ax.tick_params(labelbottom=False)
        if i>=2:ax.axhline(0,color="#aaaaaa",lw=.5)
    axes[2].fill_between(DATES,total.seasonal/1000,0,color="#91CADA",alpha=.25,lw=0)
    label(axes[0],"a")
    beta=metrics.iloc[0].Trend_km2_yr
    time=np.arange(120)/12
    linear,lower_ci,upper_ci=trend_band(total.trend)
    axes[1].fill_between(DATES,lower_ci/1000,upper_ci/1000,color="#BE727E",alpha=.16,lw=0)
    axes[1].plot(DATES,linear/1000,color="#BE727E",ls="--",lw=.9)
    pval=metrics.iloc[0].HR_MK_p
    ptext=("P unavailable" if not np.isfinite(pval) else "P < 0.001" if pval<.001 else f"P = {pval:.3f}")
    axes[1].text(.02,.93,f"Slope = {beta:+.1f} km² yr⁻¹"+("*" if np.isfinite(pval) and pval<.05 else ""),transform=axes[1].transAxes,va="top",fontsize=FONT["body"],
                 bbox=dict(facecolor="white",alpha=.8,edgecolor="none",pad=1))
    maps=upper[2].subgridspec(2,1,hspace=.13)
    draw_maps(fig,[fig.add_subplot(maps[0]),fig.add_subplot(maps[1])],grid)
    basin_panels(fig,bottom[0],bottom[1],basin_stats,basins,grid[4])
    monthly_distributions(fig,bottom[2],df,cnt)
    monthly_out=df[["Year","Month",COLS["Total"]]].copy()
    monthly_out["Count_ge_0p1"]=cnt[["Small","Medium","Large"]].sum(axis=1)
    for ext in ["png","svg","pdf"]:fig.savefig(args.figure_output/f"Figure1_QTP_dynamics.{ext}",dpi=300,bbox_inches="tight",pad_inches=.02)
    plt.close(fig)
    
    print("Figures:",args.figure_output)


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

if __name__=="__main__":
    main()

