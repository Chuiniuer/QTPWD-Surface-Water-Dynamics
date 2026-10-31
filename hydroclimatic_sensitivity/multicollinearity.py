# -*- coding: utf-8 -*-
"""Prepare nine-variable data and calculate pooled VIF and tolerance."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np
import pandas as pd

STEP6 = Path(r'K:\Monthly surface water dynamics TIB\supervised classification\step6_explaination')
INPUT = STEP6/'QTP_grid25km_dataset_CHM_PE_2016_2025'/'grid25km_sizes_environment_PE_2016_2025.csv'
OUTPUT = STEP6/'QTP_nine_variables_PE_VIF_TOL_2016_2025'
FEATURES = ['P_mm','T_C','PE_mm','SMELT_mm','SRO_mm','SSRO_mm',
            'SWNET_Wm2','SM0_100_m3m3_lag1','SWE_mm_lag1']
LABELS = ['Precipitation','Air temperature','Potential evaporation','Snowmelt',
          'Surface runoff','Subsurface runoff','Surface net solar radiation',
          'Soil moisture (lag 1 month)','Snow water equivalent (lag 1 month)']
UNITS = ['mm/month','degC','mm/month','mm/month','mm/month','mm/month',
         'W/m2','m3/m3','mm']
VERSION = '1.0'


def log(s):
    print(f'{datetime.now():%H:%M:%S} {s}',flush=True)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def calculate(x):
    # Centring incorporates an intercept; scaling improves numerical stability.
    x=np.asarray(x,dtype=float)
    if not np.isfinite(x).all():raise ValueError('The nine variables contain NaN or Inf; no values are imputed.')
    mean=x.mean(axis=0);sd=x.std(axis=0,ddof=0)
    if np.any(sd<=1e-12*np.maximum(1,abs(mean))):
        raise ValueError('A variable is constant; VIF and tolerance cannot be calculated.')
    z=(x-mean)/sd;rows=[]
    for j in range(z.shape[1]):
        y=z[:,j];others=np.delete(z,j,axis=1)
        beta=np.linalg.lstsq(others,y,rcond=None)[0]
        residual=y-others@beta
        tol=float(residual@residual/(y@y))
        tol=min(1.,max(0.,tol))
        if tol<=1e-12:tol=0.;vif=np.inf
        else:vif=1./tol
        flag=tol<0.1 or vif>10
        rows.append({'Variable':LABELS[j],'Code':FEATURES[j],'Unit':UNITS[j],
                     'Auxiliary_R2':1-tol,'TOL':tol,'VIF':vif,
                     'Assessment':'Potential multicollinearity' if flag else 'Below screening thresholds'})
    return pd.DataFrame(rows)


def main():
    log('[START] Pooled VIF and tolerance for nine variables')
    signature={'version':VERSION,'input_sha256':sha(INPUT),'features':FEATURES,
               'numpy':np.__version__,'pandas':pd.__version__}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    dataset=OUTPUT/'grid25km_nine_variables_2016_2025.csv'
    table=OUTPUT/'Table_S_VIF_TOL.csv'
    note=OUTPUT/'VIF_TOL_说明与论文文本.txt'
    if any(p.exists() for p in [dataset,table,note]):
        if not all(p.is_file() for p in [dataset,table,note]):
            raise RuntimeError('Incomplete existing outputs; choose a different OUTPUT directory.')
        marker=json.loads(note.read_text(encoding='utf-8').split('__COMPLETION_JSON__\n')[1])
        if marker['signature']!=signature or marker['dataset_sha256']!=sha(dataset) or marker['table_sha256']!=sha(table):
            raise RuntimeError('Existing outputs do not match inputs or parameters; choose a different OUTPUT directory.')
        log(f'[SKIP] Already complete: {OUTPUT}');return
    d=pd.read_csv(INPUT,dtype={'zone_id':str})
    required=['zone_id','date','model_eligible']+FEATURES
    if set(required)-set(d):raise ValueError(f'Missing input columns: {set(required)-set(d)}')
    flags=d.model_eligible.astype(str).str.strip().str.lower()
    if not flags.isin(['true','false','1','0','1.0','0.0']).all():raise ValueError('Unrecognised model_eligible value.')
    d=d.loc[flags.isin(['true','1','1.0'])].copy()
    d['date']=pd.to_datetime(d.date,errors='raise')
    d=d.sort_values(['zone_id','date']).reset_index(drop=True)
    ids=d.zone_id.drop_duplicates().to_list();dates=pd.date_range('2016-01-01','2025-12-01',freq='MS')
    if not ids or d.duplicated(['zone_id','date']).any() or len(d)!=120*len(ids):
        raise ValueError('No eligible data, duplicate keys, or the number of records per grid is not 120.')
    if not np.array_equal(d.date.to_numpy(),np.tile(dates.to_numpy(),len(ids))):
        raise ValueError('Eligible grids require 120 consecutive months from 2016 to 2025.')
    for k in FEATURES:d[k]=pd.to_numeric(d[k],errors='raise')
    report=calculate(d[FEATURES].to_numpy(float))
    # Retain response and location fields alongside the nine variables.
    optional=['Year','Month','row','col','center_x','center_y','basin',
              'Total_area_km2','Small_area_km2','Medium_area_km2','Large_area_km2',
              'ValidFraction','PE_valid_fraction','elevation_mean_m',
              'glacier_fraction','permafrost_zone_fraction','model_eligible']
    columns=['zone_id','date']+[k for k in optional if k in d]+FEATURES
    selected=d[columns]
    n=len(d);g=len(ids);passed=not report.Assessment.eq('Potential multicollinearity').any()
    method=(f'All {n:,} grid-month observations from {g:,} 25-km grid cells during 2016–2025 were pooled. '
            'For each of the nine dynamic predictors, an ordinary least-squares regression with an intercept '
            'was fitted using the other eight predictors. Tolerance (TOL) and the variance inflation factor '
            '(VIF) were calculated as TOL = 1 - R2 and VIF = 1 / TOL. Following Li et al. (2025), '
            'TOL < 0.1 or VIF > 10 was treated as an indication of potential multicollinearity.')
    result=(f'TOL ranged from {report.TOL.min():.3f} to {report.TOL.max():.3f}, and VIF ranged from '
            f'{report.VIF.min():.3f} to {report.VIF.max():.3f}. '+
            ('None of the nine predictors exceeded the specified screening thresholds.' if passed else
             'One or more predictors exceeded the specified screening thresholds; see Table Sx.'))
    lines=['Nine-variable dataset and pooled VIF/TOL diagnostics',f'Input: {INPUT}',f'Samples: {n:,} records; {g:,} grids; 120 months per grid.',
           'Dynamic variables: '+', '.join(FEATURES),'ET_mm and LWDOWN_Wm2 are excluded from the output dataset.',
           'Retains water-area response, location and spatial background fields; seasonal components are calculated by the model scripts.',
           'Only model_eligible=True records are used; no additional sampling or imputation.',
           'VIF uses pooled grid-month records and auxiliary OLS regressions with an intercept.',
           'TOL=1-R²; VIF=1/TOL. TOL<0.1 or VIF>10 indicates potential collinearity.',
           'No Spearman screening, LightGBM/RFE or automatic variable removal is performed.',
           'Pooled diagnostics are descriptive and do not treat grid-month records as independent observations.',
           'Pooled thresholds do not establish grid-level collinearity or causal attribution.',
           'The three static background variables are excluded from the diagnostics.',
           'Reference: Li, B., et al. (2025). Spatiotemporal warning of rainfall-induced railway geohazards in China. '
           'Engineering Geology, 355, 108234. https://doi.org/10.1016/j.enggeo.2025.108234',
           '', 'Table title: Table Sx. Pooled multicollinearity diagnostics for the nine dynamic predictors.',
           'Table note: R2 is the coefficient of determination of the auxiliary OLS regression of each predictor '
           'on the other eight predictors, with an intercept. Potential multicollinearity is flagged when '
           'TOL < 0.1 or VIF > 10. Values are rounded for presentation; flags use unrounded values.', '',
           '| Variable | Code | Unit | R² | TOL | VIF | Assessment |',
           '|---|---|---|---:|---:|---:|---|']
    for r in report.itertuples(index=False):
        lines.append(f'| {r.Variable} | {r.Code} | {r.Unit} | {r.Auxiliary_R2:.3f} | {r.TOL:.3f} | {r.VIF:.3f} | {r.Assessment} |')
    lines+=['','Methods:',method,'','Results:',result,'',
            f'TOL: {report.TOL.min():.3f}–{report.TOL.max():.3f}; VIF: {report.VIF.min():.3f}–{report.VIF.max():.3f}. '+
            ('No variables exceeded the thresholds.' if passed else 'One or more variables exceeded the thresholds.')]
    temp_data=dataset.with_suffix('.tmp.csv');selected.to_csv(temp_data,index=False,encoding='utf-8-sig')
    temp_table=table.with_suffix('.tmp.csv');report.to_csv(temp_table,index=False,encoding='utf-8-sig',float_format='%.6f')
    lines+=['','__COMPLETION_JSON__',json.dumps({'signature':signature,'dataset_sha256':sha(temp_data),
                                                              'table_sha256':sha(temp_table)},ensure_ascii=False)]
    temp_note=note.with_suffix('.tmp.txt');temp_note.write_text('\n'.join(lines),encoding='utf-8')
    temp_data.replace(dataset);temp_table.replace(table);temp_note.replace(note)
    print(report[['Code','TOL','VIF','Assessment']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    log(f'[DONE] {OUTPUT}; saved three files')


if __name__=='__main__':
    main()
