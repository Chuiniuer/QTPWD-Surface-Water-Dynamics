# -*- coding: utf-8 -*-
"""Estimate grid-level responses to a one-standard-deviation increase in each variable."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import sqlite3
import zlib
import numpy as np
import pandas as pd
import sklearn
import statsmodels
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import STL
from joblib import Parallel, delayed
import fiona
from pyproj import CRS, Transformer
from shapely.geometry import shape, mapping, Point
from shapely.ops import transform as geometry_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib import font_manager

# Settings
STEP6 = Path(r'K:\Monthly surface water dynamics TIB\supervised classification\step6_explaination')
CLIMATE_SOURCE = 'CHM'
OLD_ROOT = STEP6 / f'QTP_RF_SHAP_GRID_SIZES_{CLIMATE_SOURCE}_2016_2025'
INPUT_CSV = STEP6 / 'QTP_nine_variables_PE_VIF_TOL_2016_2025' / 'grid25km_nine_variables_2016_2025.csv'
BASIN_REFERENCE = OLD_ROOT / 'grid_Total' / 'grid_centres_driver_attributes.csv'
VALIDATION_DIR = STEP6 / 'QTP_RF_SHAP_PE9_2016_2025' / 'validation'
GRID_GPKG = STEP6 / 'QTP_grid25km_waterbody_sizes_2016_2025_parallel' / 'QTP_grid25km_polygons.gpkg'
GRID_LAYER = 'grid25km'
WATERSHEDS = Path(r'D:\phdl2\QTP surface water dynamics\regions\QTP_watershed.shp')
OUTPUT = STEP6 / 'QTP_RF_sensitivity_PE9_2016_2025'
VALIDATION_FOR_MAP = 'Random_months'  # 'Random_months' or 'Temporal'
MIN_R2 = 0.20
MIN_TARGET_SD_KM2 = 0.01
N_JOBS = min(6, os.cpu_count() or 1)
BATCH_SIZE = 48
FEATURES = ['P_mm', 'T_C', 'SMELT_mm', 'PE_mm', 'SRO_mm', 'SSRO_mm', 'SWNET_Wm2', 'SM0_100_m3m3_lag1', 'SWE_mm_lag1']
TARGETS = ('Small_Seasonal', 'Total_Seasonal', 'Total_Area')
AREA_COLS = ['Total_area_km2', 'Small_area_km2']
RF_PARAMS = dict(n_estimators=200, max_depth=5, min_samples_leaf=3,
                 max_features=1.0, bootstrap=True, max_samples=0.8,
                 random_state=42, n_jobs=1)
STL_PARAMS = dict(period=12, seasonal=13, robust=False)
BOUNDS = {'P_mm': (0, None), 'SMELT_mm': (0, None), 'SRO_mm': (0, None),
          'SSRO_mm': (0, None), 'SWE_mm_lag1': (0, None),
          'SM0_100_m3m3_lag1': (0, 1)}
LABELS = ['Precipitation', 'Temperature', 'Snowmelt', 'Potential evaporation', 'Surface runoff', 'Subsurface runoff', 'Net shortwave', 'Soil moisture (lag 1)', 'SWE (lag 1)']
TARGET_LABELS = {'Small_Seasonal': 'Small water: seasonal',
                 'Total_Seasonal': 'All water: seasonal', 'Total_Area': 'All water: monthly area'}
DATES = pd.date_range('2016-01-01', '2025-12-01', freq='MS')
MONTHS = DATES.month.to_numpy()
VERSION = '2.0_PE9'


def log(s):
    print(f'{datetime.now():%H:%M:%S} {s}', flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def physical_valid(values, variable):
    valid = np.isfinite(values)
    low, high = BOUNDS.get(variable, (None, None))
    if low is not None:
        valid &= values >= low
    if high is not None:
        valid &= values <= high
    return valid


def load_inputs():
    required = ['zone_id', 'date', 'model_eligible', 'center_x', 'center_y'] + FEATURES + AREA_COLS
    header = pd.read_csv(INPUT_CSV, nrows=0).columns
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f'Missing input columns: {missing}')
    columns = required + (['basin'] if 'basin' in header else [])
    d = pd.read_csv(INPUT_CSV, usecols=columns, dtype={'zone_id': str})
    flags = d.model_eligible.astype(str).str.strip().str.lower()
    if not flags.isin(['true', 'false', '1', '0', '1.0', '0.0']).all():
        raise ValueError('Missing or unrecognised model_eligible values.')
    d = d.loc[flags.isin(['true', '1', '1.0'])].copy()
    d['date'] = pd.to_datetime(d.date, errors='raise')
    d = d.sort_values(['zone_id', 'date']).reset_index(drop=True)
    ids = d.zone_id.drop_duplicates().to_numpy()
    if not len(ids) or d.duplicated(['zone_id', 'date']).any() or len(d) != len(ids)*120:
        raise ValueError('Input is empty, has duplicate months or does not contain 120 months per grid.')
    if not np.array_equal(d.date.to_numpy(), np.tile(DATES.to_numpy(), len(ids))):
        raise ValueError('Expected 120 consecutive months from 2016 to 2025.')
    for c in FEATURES + AREA_COLS:
        d[c] = pd.to_numeric(d[c], errors='raise')
    if not np.isfinite(d[FEATURES + AREA_COLS + ['center_x', 'center_y']].to_numpy(float)).all():
        raise ValueError('Input contains NaN or infinite values.')
    if (d[AREA_COLS].to_numpy() < 0).any() or (d.Small_area_km2 > d.Total_area_km2 + 1e-6).any():
        raise ValueError('Areas are negative or small-water area exceeds total area.')
    if 'basin' not in d:
        import validation as validation_module
        assigned, _ = validation_module.load_data()
        ref=assigned.groupby('zone_id').basin.first()
        d['basin']=d.zone_id.map(ref)
    if d.basin.isna().any():
        raise ValueError('Missing basin assignments.')
    d['basin'] = d.basin.astype(str).replace({'QIB': 'QlB'})
    if not d.groupby('zone_id')[['basin', 'center_x', 'center_y']].nunique().eq(1).all().all():
        raise ValueError('Grid coordinates or basin assignment vary between months.')
    validation_path = VALIDATION_DIR / 'validation_results.csv'
    note = (VALIDATION_DIR / '说明.txt').read_text(encoding='utf-8')
    marker = json.loads(note.split('__COMPLETION_JSON__\n', 1)[1])
    prior = marker['signature']
    cols = ['zone_id', 'date', 'basin', 'center_x', 'center_y'] + FEATURES + AREA_COLS
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(d[cols], index=False).to_numpy().tobytes()).hexdigest()
    if prior['data_sha256'] != data_hash or marker['csv_sha256'] != sha(validation_path):
        raise ValueError('Input data or validation output does not match its completion signature.')
    if prior['rf'] != RF_PARAMS or prior['stl'] != STL_PARAMS or prior['features'] != FEATURES:
        raise ValueError('RF parameters, STL parameters or variables differ from the validated configuration.')
    if prior['sklearn'] != sklearn.__version__ or prior['statsmodels'] != statsmodels.__version__:
        raise ValueError('scikit-learn or statsmodels version differs from the validation environment.')
    validation = pd.read_csv(validation_path, dtype={'zone_id': str})
    validation = validation.loc[validation.row_type.eq('grid')].copy()
    if validation.duplicated(['zone_id', 'target', 'scheme']).any():
        raise ValueError('Validation table contains duplicate grid, target and scheme records.')
    for scheme in ['Temporal', 'Random_months']:
        for target in TARGETS:
            part = validation[(validation.scheme == scheme) & (validation.target == target)]
            if set(part.zone_id) != set(ids) or not part.n_train.eq(96).all() or not part.n_test.eq(24).all():
                raise ValueError('Validation grid coverage or sample counts do not match.')
    signature = dict(version=VERSION, data_hash=data_hash, validation_hash=sha(validation_path),
                     grid_hash=sha(GRID_GPKG), rf=RF_PARAMS, stl=STL_PARAMS,
                     map_validation=VALIDATION_FOR_MAP, min_r2=MIN_R2,
                     min_target_sd=MIN_TARGET_SD_KM2, climate_source=CLIMATE_SOURCE,
                     sklearn=sklearn.__version__, statsmodels=statsmodels.__version__)
    if WATERSHEDS.is_file():
        signature['watersheds'] = {s: sha(WATERSHEDS.with_suffix(s)) for s in ['.shp','.shx','.dbf','.prj']
                                   if WATERSHEDS.with_suffix(s).is_file()}
    return d, ids, validation, signature


def grid_response(zone, x, total, small, rf_params, stl_params, minimum_sd):
    """Fit one model per target and predict the original and perturbed inputs."""
    targets = {'Small_Seasonal': STL(small, **stl_params).fit().seasonal,
               'Total_Seasonal': STL(total, **stl_params).fit().seasonal,
               'Total_Area': total}
    x_sd = x.std(axis=0, ddof=0)
    stack = np.tile(x, (len(FEATURES)+1, 1))
    masks, out_of_range = [], []
    for j, variable in enumerate(FEATURES):
        block = stack[(j+1)*120:(j+2)*120]
        block[:, j] += x_sd[j]
        masks.append(physical_valid(x[:, j], variable) & physical_valid(block[:, j], variable))
        out_of_range.append((block[:, j] < x[:, j].min()) | (block[:, j] > x[:, j].max()))
    output = []
    for target, y in targets.items():
        model = RandomForestRegressor(**rf_params)
        mu = float(y.mean())
        model.fit(x, y-mu)
        prediction = model.predict(stack).reshape(len(FEATURES)+1, 120)
        deltas = prediction[1:] - prediction[0]
        y_sd = float(y.std(ddof=0))
        row = dict(zone_id=str(zone), target=target, target_SD_km2=y_sd,
                   mean_area_km2=float((small if target.startswith('Small') else total).mean()))
        monthly = []
        strengths = []
        for j, variable in enumerate(FEATURES):
            valid = masks[j]
            effect = deltas[j].copy()
            effect[~valid] = np.nan
            n = int(valid.sum())
            signed = float(np.nanmean(effect)) if n else None
            absolute = float(np.nanmean(np.abs(effect))) if n else None
            ratio = 100/y_sd if y_sd >= minimum_sd else None
            varying = bool(x_sd[j] > 1e-12)
            row.update({variable+'_X_SD': float(x_sd[j]), variable+'_valid_months': n,
                        variable+'_delta_km2': signed, variable+'_abs_delta_km2': absolute,
                        variable+'_signed_pct': signed*ratio if signed is not None and ratio is not None else None,
                        variable+'_abs_pct': absolute*ratio if absolute is not None and ratio is not None else None,
                        variable+'_outside_local_range_pct': float(100*out_of_range[j][valid].mean()) if n else None})
            strengths.append(absolute if varying and absolute is not None else -1)
            for month in [0] + list(range(1, 13)):
                use = valid if month == 0 else valid & (MONTHS == month)
                count = int(use.sum())
                avg = float(deltas[j, use].mean()) if count else None
                avgabs = float(np.abs(deltas[j, use]).mean()) if count else None
                monthly.append(dict(variable=variable, calendar_month=month, valid_months=count,
                                    predictor_varies=varying, delta_km2=avg, abs_delta_km2=avgabs,
                                    signed_pct=avg*ratio if avg is not None and ratio is not None else None,
                                    abs_pct=avgabs*ratio if avgabs is not None and ratio is not None else None))
        order = np.argsort(-np.asarray(strengths), kind='stable')
        useful = [int(j) for j in order if strengths[j] > 1e-12]
        row['first_response_variable'] = FEATURES[useful[0]] if useful else 'None'
        row['second_response_variable'] = FEATURES[useful[1]] if len(useful)>1 else 'None'
        first, second = (strengths[useful[0]] if useful else 0), (strengths[useful[1]] if len(useful)>1 else 0)
        row['first_second_gap_fraction'] = float((first-second)/first) if first>0 else None
        output.append(dict(row=row, monthly=monthly))
    return dict(zone_id=str(zone), results=output)


def load_geometry(ids):
    geometries = {}
    with fiona.open(GRID_GPKG, layer=GRID_LAYER) as src:
        crs = CRS.from_user_input(src.crs_wkt or src.crs)
        if not crs.is_projected or any(abs(a.unit_conversion_factor-1)>1e-9 for a in crs.axis_info[:2]):
            raise ValueError('The 25 km grid must use a projected CRS in metres.')
        for f in src:
            zone = str(f['properties']['zone_id'])
            if zone in geometries:
                raise ValueError('The grid polygons contain duplicate zone_id values.')
            geometries[zone] = shape(f['geometry'])
    missing = set(ids) - set(geometries)
    if missing:
        raise ValueError(f'Grid polygons are missing {len(missing)} input identifiers.')
    bounds, labels = [], []
    if WATERSHEDS.is_file():
        with fiona.open(WATERSHEDS) as src:
            transformer = Transformer.from_crs(src.crs_wkt or src.crs, crs, always_xy=True)
            for f in src:
                g = geometry_transform(transformer.transform, shape(f['geometry']))
                polys = list(g.geoms) if g.geom_type == 'MultiPolygon' else [g]
                for p in polys:
                    bounds.append(np.asarray(p.exterior.coords))
                pt = g.representative_point()
                labels.append((pt.x, pt.y, str(f['properties'].get('Name', '')).replace('QIB','QlB')))
    else:
        log('[INFO] Watershed boundaries unavailable; grid results and existing basin assignments will still be used.')
    return geometries, crs, bounds, labels


def compile_tables(con, ids, data, validation):
    records = [json.loads(zlib.decompress(con.execute('SELECT payload FROM grids WHERE zone_id=?',
                                                     (str(z),)).fetchone()[0]).decode()) for z in ids]
    meta = data.iloc[::120][['zone_id','basin','center_x','center_y']]
    grid = pd.DataFrame([result['row'] for record in records for result in record['results']])
    grid = grid.merge(meta, on='zone_id', how='left', validate='many_to_one')
    for scheme in ['Temporal','Random_months']:
        v = validation.loc[validation.scheme.eq(scheme), ['zone_id','target','test_R2','MSE_skill_vs_calendar']]
        v = v.rename(columns={'test_R2':scheme+'_R2', 'MSE_skill_vs_calendar':scheme+'_MSE_skill'})
        grid = grid.merge(v, on=['zone_id','target'], validate='one_to_one')
    grid['map_eligible'] = (grid[VALIDATION_FOR_MAP+'_R2'].ge(MIN_R2)
                            & grid.target_SD_km2.ge(MIN_TARGET_SD_KM2) & grid.mean_area_km2.gt(0))
    series = pd.DataFrame([dict(zone_id=record['zone_id'], target=result['row']['target'], **item)
                          for record in records for result in record['results'] for item in result['monthly']])
    series = series.merge(grid[['zone_id','target','basin','map_eligible']], on=['zone_id','target'], validate='many_to_one')
    all_regions = ['QTP'] + sorted(set(grid.basin)-{'QTP','Unassigned'})
    summary = []
    for target in TARGETS:
        for region in all_regions:
            local_grid = grid.loc[grid.target.eq(target) & (True if region=='QTP' else grid.basin.eq(region))]
            local = series.loc[series.target.eq(target) & (True if region=='QTP' else series.basin.eq(region))]
            for scope in ['All_grids','Map_eligible_grids']:
                subset = local if scope=='All_grids' else local.loc[local.map_eligible]
                subset = subset.loc[subset.predictor_varies & subset.signed_pct.notna()]
                groups = subset.groupby(['variable','calendar_month'])
                for variable in FEATURES:
                    for month in [0]+list(range(1,13)):
                        key = (variable,month)
                        s = groups.get_group(key) if key in groups.groups else subset.iloc[:0]
                        first_count = int(((local_grid.first_response_variable==variable)
                                           & local_grid.zone_id.isin(s.zone_id)).sum())
                        summary.append(dict(target=target, region=region, scope=scope, variable=variable,
                                            calendar_month=month, n_input_grids=len(local_grid),
                                            n_map_eligible_grids=int(local_grid.map_eligible.sum()),
                                            n_defined_response_grids=len(s), n_first_response_grids=first_count,
                                            mean_signed_pct=float(s.signed_pct.mean()) if len(s) else None,
                                            mean_abs_pct=float(s.abs_pct.mean()) if len(s) else None,
                                            mean_delta_km2=float(s.delta_km2.mean()) if len(s) else None,
                                            sum_mean_delta_km2=float(s.delta_km2.sum()) if len(s) else None,
                                            mean_valid_months=float(s.valid_months.mean()) if len(s) else None))
    return grid, pd.DataFrame(summary)


def write_centres(grid, crs, path):
    layers = []
    for target in TARGETS:
        frame = grid.loc[grid.target.eq(target)].copy()
        schema = {'geometry':'Point', 'properties':{c: ('str:100' if frame[c].dtype==object else
                                                       'int' if pd.api.types.is_integer_dtype(frame[c]) or frame[c].dtype==bool
                                                       else 'float') for c in frame.columns}}
        with fiona.open(path, 'w', driver='GPKG', layer=target, crs_wkt=crs.to_wkt(), schema=schema) as dst:
            for row in frame.to_dict('records'):
                props = {c: (None if pd.isna(v) else int(v) if isinstance(v,(bool,np.bool_)) else v)
                         for c,v in row.items()}
                dst.write({'geometry':mapping(Point(row['center_x'],row['center_y'])), 'properties':props})
        layers.append(target)
    if set(fiona.listlayers(path)) != set(layers):
        raise RuntimeError('Incomplete GeoPackage layers.')
    for target in layers:
        with fiona.open(path, layer=target) as src:
            if len(src)!=len(grid.loc[grid.target.eq(target)]) or not CRS.from_user_input(src.crs_wkt).equals(crs):
                raise RuntimeError('GeoPackage count or CRS check failed.')


def draw_maps(grid, geometries, boundaries, basin_labels, png, svg):
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = 'Times New Roman' if 'Times New Roman' in available else 'DejaVu Serif'
    plt.rcParams.update({'font.family':font,'font.size':9,'svg.fonttype':'none'})
    vertices, zones = [], []
    for zone, g in geometries.items():
        polys = list(g.geoms) if g.geom_type=='MultiPolygon' else [g]
        for p in polys:
            vertices.append(np.asarray(p.exterior.coords)); zones.append(zone)
    all_xy = np.concatenate(vertices)
    lo, hi = all_xy.min(axis=0), all_xy.max(axis=0)
    colours = list(plt.get_cmap('tab10').colors)
    limits = {}
    for variable in ['T_C','P_mm']:
        v = grid.loc[grid.map_eligible,variable+'_signed_pct'].dropna().abs()
        limits[variable] = max(float(v.quantile(.98)) if len(v) else 1, 1)
    fig, axes = plt.subplots(3,3,figsize=(12,9.3))
    fig.subplots_adjust(left=.035,right=.985,top=.93,bottom=.20,wspace=.025,hspace=.065)
    mappables = {}
    for i,target in enumerate(TARGETS):
        table = grid.loc[grid.target.eq(target)].set_index('zone_id')
        for j in range(3):
            ax = axes[i,j]
            ax.add_collection(PolyCollection(vertices,facecolors='#e4e4e4',edgecolors='none',zorder=1))
            values, shown = [], []
            for k,zone in enumerate(zones):
                if zone not in table.index or not table.at[zone,'map_eligible']:
                    continue
                if j==0:
                    variable = table.at[zone,'first_response_variable']
                    value = FEATURES.index(variable) if variable in FEATURES else np.nan
                else:
                    value = table.at[zone, ('T_C' if j==1 else 'P_mm')+'_signed_pct']
                if pd.notna(value):
                    shown.append(vertices[k]);values.append(value)
            if j==0:
                artist = PolyCollection(shown,edgecolors='none',cmap=ListedColormap(colours),zorder=2)
                artist.set_clim(-.5,9.5)
            else:
                variable = 'T_C' if j==1 else 'P_mm'
                limit = limits[variable]
                artist = PolyCollection(shown,edgecolors='none',cmap='RdBu_r',
                                        norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),zorder=2)
                mappables[j] = artist
            artist.set_array(np.asarray(values,dtype=float));ax.add_collection(artist)
            if boundaries:
                ax.add_collection(LineCollection(boundaries,colors='#444444',linewidths=.38,zorder=3))
            if j==0:
                for x,y,name in basin_labels:
                    ax.text(x,y,name,fontsize=6.5,ha='center',va='center',zorder=4,
                            bbox=dict(facecolor='white',edgecolor='none',alpha=.65,pad=.3))
            ax.set_xlim(lo[0]-.02*(hi[0]-lo[0]),hi[0]+.02*(hi[0]-lo[0]))
            ax.set_ylim(lo[1]-.02*(hi[1]-lo[1]),hi[1]+.02*(hi[1]-lo[1]))
            ax.set_aspect('equal');ax.axis('off')
            ax.text(.01,.98,chr(97+i*3+j),transform=ax.transAxes,fontweight='bold',fontsize=12,va='top')
            if i==0:
                ax.set_title(['Dominant response factor','Temperature: signed response','Precipitation: signed response'][j],fontsize=10,pad=9)
            if j==0:
                ax.text(.5,-.065,TARGET_LABELS[target],transform=ax.transAxes,ha='center',va='top',fontsize=9)
                ax.text(.98,.02,f'n = {int(table.map_eligible.sum()):,} / {len(table):,}',transform=ax.transAxes,
                        ha='right',fontsize=7,bbox=dict(facecolor='white',edgecolor='none',alpha=.7,pad=1))
    fig.canvas.draw()
    for j in [1,2]:
        position = axes[2,j].get_position()
        colour_axis = fig.add_axes([position.x0,.155,position.width,.013])
        cbar = fig.colorbar(mappables[j],cax=colour_axis,orientation='horizontal',extend='both')
        cbar.set_label('Mean response / target SD (%)',fontsize=8);cbar.ax.tick_params(labelsize=7)
    handles = [Patch(facecolor=c,label=l) for c,l in zip(colours,LABELS)]
    fig.legend(handles=handles,ncol=5,loc='lower center',bbox_to_anchor=(.5,.01),frameon=False,fontsize=8,
               columnspacing=1.5,handlelength=1.1)
    fig.suptitle('Local random-forest sensitivity to +1 local SD (2016–2025)',fontsize=13,y=.985)
    fig.text(.5,.083,f'Grid filter: {VALIDATION_FOR_MAP}, R² ≥ {MIN_R2:g}; grey: unavailable or below display criteria. '
             'Colour limits: 98th percentile; data remain uncapped.',ha='center',fontsize=7)
    fig.savefig(png,dpi=300,facecolor='white');fig.savefig(svg,facecolor='white');plt.close(fig)


def description(signature, stamps):
    text = f"""Grid-level random forest sensitivity, 2016–2025.
Each variable is increased by one within-grid standard deviation while the
others remain fixed. Signed and absolute responses are normalised by the
response standard deviation. The perturbations are not causal effects.
Grid summaries use {VALIDATION_FOR_MAP} validation, R² >= {MIN_R2:g} and
response SD >= {MIN_TARGET_SD_KM2:g} km². Monthly and regional summaries
are stored in regional_sensitivity.csv.
"""
    return text+'\n__COMPLETION_JSON__\n'+json.dumps(dict(signature=signature,files=stamps),ensure_ascii=False,sort_keys=True)


def main():
    if VALIDATION_FOR_MAP not in ['Temporal','Random_months'] or N_JOBS<1:
        raise ValueError('Invalid validation scheme or worker count.')
    log(f'[START] Grid-level RF sensitivity; {N_JOBS} workers; using validation results.')
    data, ids, validation, signature = load_inputs()
    geometries, crs, boundaries, labels = load_geometry(ids)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    checkpoint = OUTPUT/'_progress.sqlite'
    filenames = ['grid_sensitivity.csv','regional_sensitivity.csv','grid_centres.gpkg',
                 'spatial_sensitivity.png','spatial_sensitivity.svg']
    note_path = OUTPUT/'说明.txt'
    if note_path.exists():
        marker = json.loads(note_path.read_text(encoding='utf-8').split('__COMPLETION_JSON__\n',1)[1])
        if marker['signature'] != json.loads(json.dumps(signature)):
            raise RuntimeError('Completed output differs from the current input or settings; use a new OUTPUT directory.')
        for name,h in marker['files'].items():
            if not (OUTPUT/name).is_file() or sha(OUTPUT/name)!=h:
                raise RuntimeError(f'Completed file is missing or modified: {name}')
        if checkpoint.exists():
            checkpoint.unlink()
        log(f'[SKIP] Completed output matches: {OUTPUT}')
        return
    if any((OUTPUT/n).exists() for n in filenames) and not checkpoint.exists():
        raise RuntimeError('Output exists without a matching checkpoint or completion signature; use a new OUTPUT directory.')
    con = sqlite3.connect(checkpoint)
    try:
        con.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT)')
        con.execute('CREATE TABLE IF NOT EXISTS grids (zone_id TEXT PRIMARY KEY,payload BLOB)')
        encoded = json.dumps(signature,sort_keys=True,ensure_ascii=False)
        old = con.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
        if old and old[0]!=encoded:
            raise RuntimeError('Checkpoint input or settings differ; use a new OUTPUT directory.')
        if not old:
            con.execute('INSERT INTO meta VALUES (?,?)',('signature',encoded));con.commit()
        done = {r[0] for r in con.execute('SELECT zone_id FROM grids')}
        if not done.issubset(set(ids)):
            raise RuntimeError('Checkpoint contains unknown grid IDs.')
        x = data[FEATURES].to_numpy(float).reshape(len(ids),120,len(FEATURES))
        areas = data[AREA_COLS].to_numpy(float).reshape(len(ids),120,2)
        pending = [i for i,z in enumerate(ids) if str(z) not in done]
        log(f'[DATA] {len(ids):,} grids; {len(done):,} completed; {len(pending):,} remaining.')
        with Parallel(n_jobs=N_JOBS,backend='loky',inner_max_num_threads=1) as parallel:
            for first in range(0,len(pending),BATCH_SIZE):
                batch = pending[first:first+BATCH_SIZE]
                results = parallel(delayed(grid_response)(ids[i],x[i],areas[i,:,0],areas[i,:,1],
                                                         RF_PARAMS,STL_PARAMS,MIN_TARGET_SD_KM2) for i in batch)
                for result in results:
                    blob = zlib.compress(json.dumps(result,allow_nan=False).encode())
                    con.execute('INSERT INTO grids VALUES (?,?)',(result['zone_id'],blob))
                con.commit()
                log(f'[RF] {len(done)+first+len(batch):,}/{len(ids):,} grids completed.')
        log('[OUTPUT] Writing regional summaries, grid centres and maps.')
        grid, regional = compile_tables(con,ids,data,validation)
        temp_names = ['grid_sensitivity.tmp.csv','regional_sensitivity.tmp.csv','grid_centres.tmp.gpkg',
                      'spatial_sensitivity.tmp.png','spatial_sensitivity.tmp.svg']
        temp_paths = [OUTPUT/n for n in temp_names]
        for frame,path in zip([grid,regional],temp_paths[:2]):
            frame.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.12g')
        if temp_paths[2].exists():
            temp_paths[2].unlink()
        write_centres(grid,crs,temp_paths[2])
        draw_maps(grid,geometries,boundaries,labels,temp_paths[3],temp_paths[4])
        for temporary,name in zip(temp_paths,filenames):
            temporary.replace(OUTPUT/name)
        stamps = {n:sha(OUTPUT/n) for n in filenames}
        temp_note = OUTPUT/'说明.tmp.txt'
        temp_note.write_text(description(signature,stamps),encoding='utf-8');temp_note.replace(note_path)
        for target in TARGETS:
            g = grid.loc[grid.target.eq(target)]
            log(f'[{target}] Map-eligible grids: {int(g.map_eligible.sum()):,}/{len(g):,}')
    finally:
        con.close()
    checkpoint.unlink()
    log(f'[DONE] {OUTPUT}')


if __name__=='__main__':
    main()
