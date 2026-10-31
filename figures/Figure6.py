# -*- coding: utf-8 -*-
"""Hydroclimatic sensitivity of total surface water seasonality, Figure 6."""
from pathlib import Path
from datetime import datetime
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import PathPatch, Patch, FancyBboxPatch
from matplotlib.path import Path as MplPath
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import ScaledTranslation
import matplotlib.patheffects as pe


BASE = Path(r'K:\Monthly surface water dynamics TIB\supervised classification\step6_explaination')
RESULTS = BASE / 'QTP_RF_sensitivity_PE9_2016_2025'
GRID_CSV = RESULTS / 'grid_sensitivity.csv'
REGIONAL_CSV = RESULTS / 'regional_sensitivity.csv'
VALIDATION_CSV = BASE / 'QTP_RF_SHAP_PE9_2016_2025' / 'validation' / 'validation_results.csv'
GRID_GPKG = BASE / 'QTP_grid25km_waterbody_sizes_2016_2025_parallel' / 'QTP_grid25km_polygons.gpkg'
GRID_LAYER = 'grid25km'
QTP_BOUNDARY = Path(r'D:\phdl2\QTP surface water dynamics\regions\QTP.shp')
WATERSHEDS = QTP_BOUNDARY.parent / 'QTP_watershed.shp'
OUTPUT = Path(r'D:\phdl2\QTP surface water dynamics\code availability\test_output')
DPI = 300
MAKE_SPATIAL_MAPS = True
NAMED_RANK_FEATURES = ['T_C','SSRO_mm','SM0_100_m3m3_lag1','P_mm']
LABEL_ALL_MAPS = True
BASIN_LABEL_OFFSETS = {}
MAP_BOUNDS = np.arange(-50,51,10,dtype=float)
MIN_RANDOM_R2 = .20
MIN_TARGET_SD = .01
SIGN_EPS = 1e-8
VERSION = '8.0.0'
FEATURES = ['P_mm', 'T_C', 'SMELT_mm', 'PE_mm', 'SRO_mm', 'SSRO_mm',
            'SWNET_Wm2', 'SM0_100_m3m3_lag1', 'SWE_mm_lag1']
LABELS = ['Precipitation', 'Mean air temperature', 'Snowmelt', 'Potential evaporation',
          'Surface runoff', 'Subsurface runoff', 'Surface net shortwave radiation',
          'Soil moisture', 'Snow water equivalent']
NAMES = dict(zip(FEATURES, LABELS))
TARGETS = ['Total_Area', 'Total_Seasonal', 'Small_Seasonal']
TARGET_NAMES = {'Total_Area': 'Total water area', 'Total_Seasonal': 'Total water: seasonal component',
                'Small_Seasonal': 'Small water: seasonal component'}

COLOURS = dict(zip(TARGETS, ['#8CA2C4', '#83B99C', '#D68D99']))
REGIONS = ['QTP', 'KlB', 'QdB', 'QlB', 'NrB', 'QtB', 'YtB', 'SWB', 'YB', 'STB']
MAP_COLOURS = ['#326FA9','#5397C6','#74B7DD','#95D1E7','#C0E9F0',
               '#FFE8AD','#FED38A','#FCAF66','#E98153','#C45143']
CMAP = ListedColormap(MAP_COLOURS,name='sensitivity_steps')
CMAP.set_under('#234E80');CMAP.set_over('#9D3339');CMAP.set_bad((0,0,0,0))
NORM = BoundaryNorm(MAP_BOUNDS,CMAP.N,clip=False)
RANK_CATEGORIES = NAMED_RANK_FEATURES + ['Others']
PRIMARY_COLOURS = dict(zip(RANK_CATEGORIES,['#C94B48','#2A779D','#3E8B68','#B28D35','#7D858B']))
SECONDARY_COLOURS = dict(zip(RANK_CATEGORIES,['#F2B1A1','#ACD1E2','#BDD9C5','#ECD696','#D6DBDE']))
FULL_NAMES = dict(NAMES)
FULL_NAMES['SM0_100_m3m3_lag1']='0–100 cm soil moisture (1-month lag)'
FULL_NAMES['SWE_mm_lag1']='Snow water equivalent (1-month lag)'
FIGURE_WIDTH = 15.5
VALIDATION_R2_XLIM = (-1.0, 1.0)
S3_DISPLAY_QUANTILE = .98


def log(s):
    print(f'{datetime.now():%H:%M:%S} {s}', flush=True)


def flags(s):
    q = s.astype(str).str.strip().str.lower()
    if not q.isin(['true','false','1','0','1.0','0.0']).all():
        raise ValueError('map_eligible字段不能识别。')
    return q.isin(['true','1','1.0'])


def style():
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = 'Times New Roman' if 'Times New Roman' in available else 'DejaVu Serif'
    plt.rcParams.update({'font.family': font, 'font.size': 10, 'axes.labelsize': 10,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.linewidth': .65,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    return font


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2.7, width=.6)
    ax.set_axisbelow(True)


def heading(ax, value, title='', size=11.1, pad=8, letter_offset=.22):
    """编号和名称共用同一baseline及纵向偏移，不受标题默认位置影响。"""
    fig=ax.figure
    title_trans=ax.transAxes+ScaledTranslation(0,pad/72,fig.dpi_scale_trans)
    letter_trans=ax.transAxes+ScaledTranslation(-letter_offset,pad/72,fig.dpi_scale_trans)
    name=ax.text(0,1,title,transform=title_trans,fontsize=size,ha='left',va='baseline')
    label=ax.text(0,1,value,transform=letter_trans,fontsize=14,fontweight='bold',ha='left',va='baseline')
    return label,name


def wrap(s):
    if '(1-month lag)' in s:
        return s.replace(' (1-month lag)', '\n(1-month lag)')
    return '\n'.join(textwrap.wrap(s, width=34, break_long_words=False))


def load_data():
    g = pd.read_csv(GRID_CSV, dtype={'zone_id': str})
    required = ['zone_id','target','basin','center_x','center_y','map_eligible',
                'target_SD_km2','mean_area_km2','Random_months_R2']
    required += [f+s for f in FEATURES for s in ['_X_SD','_valid_months','_signed_pct','_abs_pct','_outside_local_range_pct']]
    if set(required)-set(g.columns):
        raise ValueError(f'网格表缺字段：{sorted(set(required)-set(g.columns))}')
    g = g[g.target.isin(TARGETS)].copy()
    g['basin'] = g.basin.replace({'QIB':'QlB','KIB':'KlB'})
    g['map_eligible'] = flags(g.map_eligible)
    if g.duplicated(['zone_id','target']).any() or set(g.target) != set(TARGETS):
        raise ValueError('应包含三个目标，且格网-目标不能重复。')
    for t in TARGETS:
        if set(g[g.target.eq(t)].zone_id) != set(g.zone_id):
            raise ValueError(f'{t}网格集合不完整。')
    if not g.groupby('zone_id')[['basin','center_x','center_y']].nunique(dropna=False).eq(1).all().all():
        raise ValueError('同格网的流域/坐标不一致。')
    if g.basin.isna().any() or set(g.basin)-set(REGIONS[1:]):
        raise ValueError('存在缺失或未识别的流域。')
    # 此版只用Random-month检验，不读取/核对/展示Temporal。
    eligible = (g.Random_months_R2.ge(MIN_RANDOM_R2) & g.target_SD_km2.ge(MIN_TARGET_SD)
                & g.mean_area_km2.gt(0))
    if not np.array_equal(eligible.to_numpy(), g.map_eligible.to_numpy()):
        raise ValueError('已有map_eligible不是本版Random-month口径，不能直接复用旧月度表。')
    v = pd.read_csv(VALIDATION_CSV, dtype={'zone_id': str})
    v = v[v.row_type.eq('grid') & v.scheme.eq('Random_months') & v.target.isin(TARGETS)].copy()
    if v.duplicated(['zone_id','target']).any():
        raise ValueError('Random-month验证记录重复。')
    for t in TARGETS:
        a = g[g.target.eq(t)].set_index('zone_id')
        b = v[v.target.eq(t)].set_index('zone_id')
        if set(a.index) != set(b.index) or not np.allclose(a.Random_months_R2, b.reindex(a.index).test_R2, equal_nan=True):
            raise ValueError(f'{t}验证与敏感性不是同一版本。')
    r = pd.read_csv(REGIONAL_CSV)
    r['region'] = r.region.replace({'QIB':'QlB','KIB':'KlB'})
    r = r[r.target.isin(TARGETS) & r.scope.eq('Map_eligible_grids')].copy()
    if r.duplicated(['target','region','variable','calendar_month']).any():
        raise ValueError('区域月度表记录重复。')
    return g, r, v


def subset(g, target, feature, region='QTP'):
    q = g[g.target.eq(target) & g.map_eligible & g[feature+'_X_SD'].gt(1e-12)].copy()
    if region != 'QTP':
        q = q[q.basin.eq(region)]
    q = q[np.isfinite(q[feature+'_signed_pct']) & np.isfinite(q[feature+'_abs_pct'])]
    if len(q) and ((q[feature+'_abs_pct'] < np.abs(q[feature+'_signed_pct'])-1e-7).any()
                   or (q[feature+'_valid_months'] <= 0).any()):
        raise ValueError(f'响应字段异常：{target}/{feature}')
    return q.set_index('zone_id').sort_index()


def comparison_subset(g, target, feature):

    shared = set.intersection(*(set(subset(g, t, feature).index) for t in TARGETS))
    return subset(g, target, feature).reindex(sorted(shared))


def distribution_stats(q, f):
    signed = q[f+'_signed_pct'].to_numpy(float)
    absolute = q[f+'_abs_pct'].to_numpy(float)
    def percentile(x, p):
        return float(np.percentile(x, p)) if len(x) else np.nan
    return dict(n_grids=len(q), signed_p10=percentile(signed,10), signed_q25=percentile(signed,25),
        signed_median=percentile(signed,50), signed_q75=percentile(signed,75), signed_p90=percentile(signed,90),
        abs_q25=percentile(absolute,25), abs_median=percentile(absolute,50), abs_q75=percentile(absolute,75),
        mean_signed_pct=float(signed.mean()) if len(q) else np.nan,
        mean_abs_pct=float(absolute.mean()) if len(q) else np.nan,
        positive_pct=100*np.mean(signed>SIGN_EPS) if len(q) else np.nan,
        negative_pct=100*np.mean(signed<-SIGN_EPS) if len(q) else np.nan,
        near_zero_pct=100*np.mean(np.abs(signed)<=SIGN_EPS) if len(q) else np.nan,
        outside_range_months_pct=q[f+'_outside_local_range_pct'].mean(),
        mean_valid_months=q[f+'_valid_months'].mean())


def analyse(g, r, v):
    rows = []
    for t in TARGETS:
        for f in FEATURES:
            for region in REGIONS:
                q = subset(g, t, f, region)
                st = distribution_stats(q, f)
                old = r[r.target.eq(t) & r.region.eq(region) & r.variable.eq(f) & r.calendar_month.eq(0)]
                if len(old)!=1 or int(old.iloc[0].n_defined_response_grids)!=len(q):
                    raise ValueError(f'区域样本数不匹配：{t}/{region}/{f}')
                for field in ['mean_signed_pct','mean_abs_pct']:
                    if not np.isclose(st[field], old.iloc[0][field], rtol=1e-6, atol=1e-7, equal_nan=True):
                        raise ValueError(f'区域结果不匹配：{t}/{region}/{f}/{field}')
                rows.append(dict(row_type='regional_distribution',scope='Target_eligible',target=t,region=region,
                    variable=f,variable_name=NAMES[f],**st))
            common = comparison_subset(g, t, f)
            for region in REGIONS:
                q=common if region=='QTP' else common[common.basin.eq(region)]
                rows.append(dict(row_type='regional_distribution',scope='Common_three_targets',target=t,region=region,
                                variable=f,variable_name=NAMES[f],**distribution_stats(q,f)))
    # 验证表仅Random-month。汇总时不先删除低R²格网。
    view_limits=validation_limits(v)
    for t in TARGETS:
        q = v[v.target.eq(t)]
        r2 = q.test_R2.replace([np.inf,-np.inf],np.nan).dropna()
        tail_counts={}
        for field,(lo,hi) in view_limits.items():
            x=q[field].replace([np.inf,-np.inf],np.nan).dropna()
            tail_counts.update({field+'_view_min':lo,field+'_view_max':hi,
                               field+'_below_view_n':int(x.lt(lo).sum()),field+'_above_view_n':int(x.gt(hi).sum()),
                               field+'_min':x.min(),field+'_max':x.max()})
        rows.append(dict(row_type='random_month_validation',scope='All_modelled_grids',target=t,region='QTP',
            n_grids=len(q), n_defined_R2=len(r2), R2_median=r2.median(), R2_q25=r2.quantile(.25),
            R2_q75=r2.quantile(.75), R2_positive_pct=100*r2.gt(0).mean(),
            RMSE_median_km2=q.test_RMSE_km2.median(), MAE_median_km2=q.test_MAE_km2.median(),**tail_counts))
    stats = pd.DataFrame(rows)

    for f in FEATURES:
        arrays,upper=violin_data(g,f)
        for t,x in zip(TARGETS,arrays):
            mask=(stats.row_type.eq('regional_distribution') & stats.scope.eq('Common_three_targets')
                  & stats.region.eq('QTP') & stats.target.eq(t) & stats.variable.eq(f))
            if mask.sum()!=1:raise ValueError(f'S3共同格网统计缺失或重复：{t}/{f}')
            stats.loc[mask,'S3_abs_view_min_pct']=0.
            stats.loc[mask,'S3_abs_view_max_pct']=upper
            stats.loc[mask,'S3_above_view_n']=int(np.count_nonzero(x>upper))
            stats.loc[mask,'S3_above_view_pct']=100*np.mean(x>upper)
    monthly = r[r.calendar_month.between(1,12) & r.variable.isin(FEATURES)].copy()
    for t in TARGETS:
        for region in REGIONS:
            for f in FEATURES:
                a = monthly[monthly.target.eq(t) & monthly.region.eq(region) & monthly.variable.eq(f)]
                if sorted(a.calendar_month.tolist()) != list(range(1,13)):
                    raise ValueError(f'月度记录不完整：{t}/{region}/{f}')
    monthly.insert(4, 'variable_name', monthly.variable.map(NAMES))
    return stats, monthly


def rank_tables(g):
    """复用原第一/第二排名，并用原始km²绝对扰动响应核对排名。"""
    required=['first_response_variable','second_response_variable']+[f+'_abs_delta_km2' for f in FEATURES]
    if set(required)-set(g.columns):raise ValueError('排名字段不完整。')
    selected=g[g.map_eligible].copy()
    val=selected[[f+'_abs_delta_km2' for f in FEATURES]].to_numpy(float)
    sd=selected[[f+'_X_SD' for f in FEATURES]].to_numpy(float)
    useful=np.isfinite(val)&(sd>1e-12)&(val>1e-12)
    order=np.argsort(-np.where(useful,val,-np.inf),axis=1,kind='stable')
    for rank,col in [(0,'first_response_variable'),(1,'second_response_variable')]:
        values=np.where(np.take_along_axis(useful,order[:,rank:rank+1],axis=1)[:,0],
                        np.array(FEATURES)[order[:,rank]],'None')
        if not np.array_equal(values,selected[col].fillna('None').to_numpy()):
            raise ValueError(f'原{col}与绝对响应排序不一致，不能混用排名。')
    complete=selected.first_response_variable.isin(FEATURES)&selected.second_response_variable.isin(FEATURES)
    rank_grids=selected[complete].copy()
    if (rank_grids.first_response_variable==rank_grids.second_response_variable).any():
        raise ValueError('第一第二名不能是同一变量。')
    full,grouped=[],[]
    groups={}
    for target in TARGETS:
        for region in REGIONS:
            q=rank_grids[rank_grids.target.eq(target)]
            if region!='QTP':q=q[q.basin.eq(region)]
            counts=np.array([[int(q[c].eq(f).sum()) for f in FEATURES]
                             for c in ['first_response_variable','second_response_variable']])
            pct=100*counts/len(q) if len(q) else np.zeros((2,9))
            keep=[FEATURES.index(f) for f in NAMED_RANK_FEATURES]
            omitted=[f for f in FEATURES if f not in NAMED_RANK_FEATURES]
            groups[(target,region)]=RANK_CATEGORIES.copy()
            eligible=g[g.target.eq(target)&g.map_eligible&(True if region=='QTP' else g.basin.eq(region))]
            for rank in [0,1]:
                for j,f in enumerate(FEATURES):
                    full.append(dict(target=target,region=region,rank=rank+1,variable=f,
                        variable_name=FULL_NAMES[f],n_eligible_grids=len(eligible),n_ranked_grids=len(q),
                        grid_count=int(counts[rank,j]),grid_share_pct=float(pct[rank,j]),
                        shown_separately=j in keep,others_members=';'.join(omitted)))
                for f in groups[(target,region)]:
                    count=int(counts[rank,FEATURES.index(f)]) if f!='Others' else int(sum(counts[rank,j] for j in range(9) if j not in keep))
                    grouped.append(dict(target=target,region=region,rank=rank+1,variable=f,
                        variable_name=FULL_NAMES.get(f,'Others'),n_ranked_grids=len(q),grid_count=count,
                        grid_share_pct=100*count/len(q) if len(q) else np.nan,others_members=';'.join(omitted)))
    return pd.DataFrame(full),pd.DataFrame(grouped),groups


def light(colour, amount=.40):
    return tuple((1-amount)*np.array(to_rgb(colour))+amount)


def ring(ax, grouped, groups, target, region, fontsize=10):
    variables=groups[(target,region)]
    for rank,radius,width in [(1,1.0,.25),(2,.69,.23)]:
        q=grouped[grouped.target.eq(target)&grouped.region.eq(region)&grouped['rank'].eq(rank)].set_index('variable')
        values=q.reindex(variables).grid_count.fillna(0).to_numpy()
        colours=[PRIMARY_COLOURS[f] if rank==1 else SECONDARY_COLOURS[f] for f in variables]
        if values.sum():
            ax.pie(values,radius=radius,colors=colours,startangle=90,counterclock=False,
                   wedgeprops=dict(width=width,edgecolor='white',linewidth=.5))
        else:
            ax.pie([1],radius=radius,colors=['#ECEDEC'],wedgeprops=dict(width=width,edgecolor='white'))
    ax.text(0,0,region,ha='center',va='center',fontsize=fontsize,fontweight='bold',color='#374B53')
    ax.set_xlim(-1.08,1.08);ax.set_ylim(-1.08,1.08);ax.set_aspect('equal');ax.axis('off')


def qtp_box(ax, g, target, labelled=True):
    y=np.arange(9)
    for j,f in enumerate(FEATURES):
        x=subset(g,target,f)[f+'_signed_pct'].to_numpy(float)
        if not len(x):continue
        ax.boxplot([x],positions=[j],vert=False,widths=.43,showfliers=False,whis=(10,90),
            patch_artist=True,boxprops=dict(facecolor=light(COLOURS[target],.30),edgecolor=COLOURS[target],linewidth=.7),
            medianprops=dict(color='#314E5B',linewidth=1.3),whiskerprops=dict(color=COLOURS[target],linewidth=.8),
            capprops=dict(color=COLOURS[target],linewidth=.8))
    ends=[]
    for f in FEATURES:
        x=subset(g,target,f)[f+'_signed_pct'].to_numpy(float)
        if len(x):ends.extend(np.percentile(x,[10,90]))
    low=min(0,min(ends));high=max(0,max(ends));span=max(high-low,1)
    ax.set_xlim(low-.06*span,high+.06*span);ax.set_ylim(8.65,-.65)
    ax.axvline(0,color='#92A2A8',lw=.65,ls='--',zorder=0)
    ax.set_yticks(y,[wrap(n) for n in LABELS] if labelled else list('abcdefghi'),fontsize=8)
    ax.set_xlabel('Sensitivity (%)',fontsize=9)
    clean(ax);ax.tick_params(axis='y',length=0,pad=4)


def map_panel(fig, rectangle, g, geo, target, feature, index):
    ax=fig.add_axes(rectangle)
    ax.add_collection(PatchCollection(geo['land'],facecolor='#F1F1ED',edgecolor='none',zorder=0))
    ax.add_collection(PatchCollection(geo['patches'],facecolor='#D0D3D1',edgecolor='none',zorder=1))
    q=subset(g,target,feature)
    value=q[feature+'_signed_pct'].reindex(geo['owners']).to_numpy(float)
    pc=PatchCollection(geo['patches'],cmap=CMAP,norm=NORM,edgecolor='none',zorder=2)
    pc.set_array(np.ma.masked_invalid(value));ax.add_collection(pc)
    zero=np.isfinite(value)&(np.abs(value)<=SIGN_EPS)
    if zero.any():
        ax.add_collection(PatchCollection([p for p,k in zip(geo['patches'],zero) if k],
                          facecolor='#F7F7F1',edgecolor='none',zorder=2.1))
    ax.add_collection(LineCollection(geo['lines'],colors='#6F8289',linewidths=.33,zorder=3))
    ax.add_collection(LineCollection(geo['outer'],colors='#405A65',linewidths=.70,zorder=4))
    x0,y0,x1,y1=geo['extent'];dx=x1-x0;dy=y1-y0
    ax.set_xlim(x0-.018*dx,x1+.018*dx);ax.set_ylim(y0-.022*dy,y1+.025*dy)
    ax.set_aspect('equal');ax.axis('off')
    ax._publication_feature=feature
    ax._publication_heading=heading(ax,chr(97+index),NAMES[feature])
    if LABEL_ALL_MAPS or index==0:
        for x,y,name in geo.get('labels',[]):
            ox,oy=BASIN_LABEL_OFFSETS.get(name,(0,0))
            ax.text(x+ox,y+oy,name,ha='center',va='center',fontsize=7.7,color='#2F4B55',zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.8,foreground='white',alpha=.94)])
    return ax


def pill(ax, x, y, colour, width=.18, height=.055):
    ax.add_patch(FancyBboxPatch((x,y-height/2),width,height,
        boxstyle=f'round,pad=0,rounding_size={height/2}',facecolor=colour,edgecolor='none',transform=ax.transAxes))


def paired_legend(fig, rect):
    """像参考图一样：左侧灰色示意＋箭头；右侧Primary/Secondary深浅两列。"""
    x,y,w,h=rect
    diagram=fig.add_axes([x,y+h*.09,w*.29,h*.85])
    diagram.pie([1],radius=1,colors=['#7D858B'],wedgeprops=dict(width=.28,edgecolor='white',linewidth=.5))
    diagram.pie([1],radius=.66,colors=['#CDD3D7'],wedgeprops=dict(width=.25,edgecolor='white',linewidth=.5))
    diagram.annotate('Primary',xy=(-.63,.66),xytext=(-1.08,1.47),ha='center',va='center',fontsize=9.7,
                     arrowprops=dict(arrowstyle='-|>',color='#3F5058',lw=.8))
    diagram.annotate('Secondary',xy=(-.36,-.40),xytext=(-1.07,-1.33),ha='center',va='center',fontsize=9.7,
                     arrowprops=dict(arrowstyle='-|>',color='#3F5058',lw=.8))
    diagram.set_xlim(-1.78,1.25);diagram.set_ylim(-1.63,1.75);diagram.set_aspect('equal');diagram.axis('off')
    palette=fig.add_axes([x+w*.33,y+h*.07,w*.67,h*.87]);palette.axis('off')
    palette.text(.005,.96,'Primary',fontstyle='italic',fontweight='bold',fontsize=10.5)
    palette.text(.23,.96,'Secondary',fontstyle='italic',fontweight='bold',fontsize=10.5)
    for i,f in enumerate(RANK_CATEGORIES):
        yy=.80-i*.165
        pill(palette,.01,yy,PRIMARY_COLOURS[f],width=.17,height=.065)
        pill(palette,.245,yy,SECONDARY_COLOURS[f],width=.17,height=.065)
        palette.text(.465,yy,NAMES.get(f,'Others'),va='center',ha='left',fontsize=10.2)
    return diagram,palette


def lower_section(fig, g, grouped, groups, target, bottom, height):
    """坐标以英寸控制，不因大面板留下大量左右空白。"""
    W,H=fig.get_size_inches()
    def rect(x,y,w,h):return [x/W,y/H,w/W,h/H]
    box_left=2.25;box_bottom=bottom+.08;box_h=height-.20
    ax=fig.add_axes(rect(box_left,box_bottom,3.35,box_h));qtp_box(ax,g,target)
    # j在Precipitation标签左上方；k与j共用同一图坐标baseline。
    first_tick=ax.get_yticklabels()[0]
    renderer=fig.canvas.get_renderer()
    text_width=renderer.get_text_width_height_descent(first_tick.get_text(),first_tick.get_fontproperties(),ismath=False)[0]/fig.dpi
    header_y=box_bottom+box_h*(8.65/9.3)+12/72
    j_x=box_left-4/72-text_width-.12
    j=fig.text(j_x/W,header_y/H,'j',fontsize=14,fontweight='bold',va='baseline')
    size=(height-.12)/3;gap=.08;left=6.00
    for i,region in enumerate(REGIONS[1:]):
        xx=left+(i%3)*(size+gap);yy=bottom+(2-i//3)*(size+.04)
        ring(fig.add_axes(rect(xx,yy,size,size)),grouped,groups,target,region,fontsize=9.7)
    k=fig.text((left-.18)/W,header_y/H,'k',fontsize=14,fontweight='bold',va='baseline')
    fig._regional_letters=(j,k)
    lx=left+3*(size+gap)+.12
    paired_legend(fig,rect(lx,bottom+.08,W-lx-.24,height-.16))


def combined_figure(g, grouped, groups, geo, target, maps=True):
    W=FIGURE_WIDTH
    lower_h=2.84
    bottom=.43
    lower_top=bottom+lower_h
    map_bottom=lower_top+.75
    left=.48;right=W-.23;col_gap=.13
    map_w=(right-left-2*col_gap)/3
    if maps:
        x0,y0,x1,y1=geo['extent']
        ratio=((x1-x0)*1.036)/((y1-y0)*1.047)
        map_h=map_w/ratio
        row_gap=.35
        H=map_bottom+3*map_h+2*row_gap+.37
    else:
        map_h=0.;H=lower_top+.48
    fig=plt.figure(figsize=(W,H))
    if maps:
        for j,f in enumerate(FEATURES):
            xx=left+(j%3)*(map_w+col_gap)
            yy=map_bottom+(2-j//3)*(map_h+row_gap)
            map_panel(fig,[xx/W,yy/H,map_w/W,map_h/H],g,geo,target,f,j)
        cax=fig.add_axes([.20,.0,.60,.01])
        cax.set_position([2.50/W,(lower_top+.36)/H,(W-5.0)/W,.12/H])
        cb=fig.colorbar(ScalarMappable(norm=NORM,cmap=CMAP),cax=cax,orientation='horizontal',
                       boundaries=MAP_BOUNDS,spacing='uniform',extend='both')
        cb.set_ticks(MAP_BOUNDS);cb.set_label('Sensitivity (%)',fontsize=10,labelpad=3)
        cb.ax.tick_params(length=2,labelsize=8.7,pad=2)
    lower_section(fig,g,grouped,groups,target,bottom,lower_h)
    return fig


def violin_data(g, feature):
    """三个模型逐变量采用同一批格网，纵轴由合并样本决定。"""
    frames=[comparison_subset(g,t,feature) for t in TARGETS]
    if not len(frames[0]) or not all(q.index.equals(frames[0].index) for q in frames[1:]):
        raise ValueError(f'S3缺少三个模型共同格网：{feature}')
    arrays=[q[feature+'_abs_pct'].to_numpy(float) for q in frames]
    if any(not np.isfinite(x).all() or np.any(x<0) for x in arrays):
        raise ValueError(f'S3绝对敏感性数据无效：{feature}')
    upper=max(1.,float(np.quantile(np.concatenate(arrays),S3_DISPLAY_QUANTILE)))*1.10
    return arrays,upper


def validation_limits(v):
    x=v.test_RMSE_km2.to_numpy(float);x=x[np.isfinite(x)]
    upper=max(.1,float(np.percentile(x,98)))*1.10
    return {'test_R2':tuple(VALIDATION_R2_XLIM),'test_RMSE_km2':(0.,upper)}


def read_geometry(ids):
    import fiona
    from shapely.geometry import shape
    from shapely.geometry.polygon import orient
    from shapely.ops import transform
    from pyproj import CRS, Transformer

    def polygons(g):
        if g.is_empty:
            return []
        if g.geom_type == 'Polygon':
            return [g]
        return [p for child in getattr(g, 'geoms', []) for p in polygons(child)]

    def patches(g):
        result = []
        for p in polygons(g):
            p = orient(p, sign=1.0)
            vertices, codes = [], []
            for ring in [p.exterior] + list(p.interiors):
                xy = np.asarray(ring.coords)
                vertices.extend(xy)
                codes.extend([MplPath.MOVETO] + [MplPath.LINETO]*(len(xy)-2) + [MplPath.CLOSEPOLY])
            result.append(PathPatch(MplPath(vertices, codes)))
        return result

    all_patches, owners, found = [], [], set()
    with fiona.open(GRID_GPKG, layer=GRID_LAYER) as src:
        crs = CRS.from_user_input(src.crs_wkt or src.crs)
        if not crs.is_projected or any(abs(a.unit_conversion_factor-1) > 1e-9 for a in crs.axis_info[:2]):
            raise ValueError('统计网格必须使用米制投影。')
        for f in src:
            z = str(f['properties']['zone_id'])
            if z not in ids:
                continue
            if z in found:
                raise ValueError(f'网格重复：{z}')
            found.add(z)
            pp = patches(shape(f['geometry']))
            if not pp:
                raise ValueError(f'空网格几何：{z}')
            all_patches.extend(pp); owners.extend([z]*len(pp))
    if set(ids)-found:
        raise ValueError(f'网格面缺少 {len(set(ids)-found)} 个编号。')
    lines, outer_lines, land, bounds, basin_geoms, outer_geoms = [], [], [], [], {}, []
    for file, is_outer in [(WATERSHEDS, False), (QTP_BOUNDARY, True)]:
        with fiona.open(file) as src:
            if not src.crs_wkt and not src.crs:
                raise ValueError(f'缺少 CRS：{file}')
            project = Transformer.from_crs(src.crs_wkt or src.crs, crs, always_xy=True)
            for f in src:
                g = transform(project.transform, shape(f['geometry']))
                if is_outer:outer_geoms.append(g)
                if not is_outer:
                    name=str(f['properties']['Name']).replace('QIB','QlB').replace('KIB','KlB')
                    basin_geoms.setdefault(name,[]).append(g)
                for p in polygons(g):
                    (outer_lines if is_outer else lines).append(np.asarray(p.exterior.coords))
                    if is_outer:
                        land.extend(patches(p)); bounds.append(p.bounds)
    if not bounds:
        raise ValueError('QTP 边界为空。')
    a = np.asarray(bounds)
    extent = [a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()]
    from shapely.ops import unary_union
    labels=[]
    boundary=unary_union(outer_geoms)
    for name,geoms in basin_geoms.items():
        clipped=unary_union(geoms).intersection(boundary)
        if not clipped.is_empty and name in REGIONS[1:]:
            point=clipped.representative_point()
            labels.append((point.x,point.y,name))
    return dict(patches=all_patches, owners=owners, lines=lines, outer=outer_lines,
                land=land, extent=extent, crs=crs, labels=labels)


def optimize_map_svg(source, destination, precision=3, batch_chars=18000):
    """合并同色、无描边的格网路径；仍为纯矢量SVG，图例/文字/边界不合并。
    仅针对本程序PatchCollection中的不相交统计格网，不是通用SVG优化器。
    数字舍入最多0.0005 SVG坐标单位；每个合并路径限制长度，避免单路径过长。
    """
    import re
    import xml.etree.ElementTree as ET
    from collections import OrderedDict
    ns='http://www.w3.org/2000/svg';tag='{'+ns+'}'
    for prefix,uri in [('',ns),('xlink','http://www.w3.org/1999/xlink'),
                       ('dc','http://purl.org/dc/elements/1.1/'),
                       ('cc','http://creativecommons.org/ns#'),
                       ('rdf','http://www.w3.org/1999/02/22-rdf-syntax-ns#')]:
        ET.register_namespace(prefix,uri)
    root=ET.parse(source).getroot()
    if root.tag!=tag+'svg':raise ValueError(f'文件不是SVG：{source}')
    before=sum(e.tag==tag+'path' for e in root.iter())
    number=re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')
    def compact(d):
        def rounded(match):
            value=float(match.group())
            if not np.isfinite(value):raise ValueError('SVG路径存在非有限坐标。')
            s=f'{value:.{precision}f}'.rstrip('0').rstrip('.')
            return '0' if s in ['','-0'] else s
        return ' '.join(number.sub(rounded,d).split())
    changed=0;removed_invisible=0
    for group in root.iter(tag+'g'):
        children=list(group)
        if not group.get('id','').startswith('PatchCollection_') or len(children)<100:continue
        if any(e.tag!=tag+'path' or 'id' in e.attrib for e in children):continue

        styles=[dict(part.split(':',1) for part in e.get('style','').split(';') if ':' in part) for e in children]
        styles=[{k.strip():v.strip() for k,v in s.items()} for s in styles]
        if any(s.get('stroke','none')!='none' or any(k.endswith('opacity') for k in s) for s in styles):continue
        buckets=OrderedDict()
        for e,style in zip(children,styles):
            if style.get('fill')=='none':
                removed_invisible+=1;continue
            attrs={k:v for k,v in e.attrib.items() if k!='d'}
            key=tuple(sorted(attrs.items()))
            buckets.setdefault(key,[]).append(compact(e.get('d','')))
        replacements=[]
        for key,paths in buckets.items():
            chunk=[];length=0
            for d in paths:
                if chunk and length+len(d)+1>batch_chars:
                    replacements.append(ET.Element(tag+'path',dict(key,d=' '.join(chunk))))
                    chunk=[];length=0
                chunk.append(d);length+=len(d)+1
            if chunk:replacements.append(ET.Element(tag+'path',dict(key,d=' '.join(chunk))))
        group[:]=replacements;changed+=1
    after=sum(e.tag==tag+'path' for e in root.iter())
    report=dict(paths_before=before,paths_after=after,collections_merged=changed,
                invisible_paths_removed=removed_invisible,bytes_before=Path(source).stat().st_size)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_name(destination.name+'.working')
    ET.ElementTree(root).write(temp,encoding='utf-8',xml_declaration=True)

    check=ET.parse(temp).getroot()
    if any(e.tag==tag+'image' for e in check.iter()):raise ValueError('此优化流程只接受原本纯矢量SVG。')
    ids=[e.get('id') for e in check.iter() if e.get('id')]
    if len(ids)!=len(set(ids)):raise ValueError('SVG存在重复id。')
    refs=[]
    for e in check.iter():
        for k,v in e.attrib.items():
            refs.extend(re.findall(r'url\(#([^)]*)\)',v))
            if k.endswith('href') and v.startswith('#'):refs.append(v[1:])
    if set(refs)-set(ids):raise ValueError('SVG存在失效引用。')
    temp.replace(destination);report['bytes_after']=destination.stat().st_size
    return report


def main():
    g, r, v = load_data()
    analyse(g, r, v)
    _, grouped, groups = rank_tables(g)
    style()
    geo = read_geometry(set(g.zone_id)) if MAKE_SPATIAL_MAPS else None
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig = combined_figure(g, grouped, groups, geo, 'Total_Seasonal', maps=MAKE_SPATIAL_MAPS)
    for ext in ('png', 'svg', 'pdf'):
        path = OUTPUT / f'Figure6.{ext}'
        fig.savefig(path, dpi=DPI, facecolor='white')
        if ext == 'svg' and MAKE_SPATIAL_MAPS:
            optimize_map_svg(path, path)
        log(f'[SAVE] {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
