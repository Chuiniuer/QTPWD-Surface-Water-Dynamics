# -*- coding: utf-8 -*-
"""Validate grid-specific random forest models with temporal and random-month splits."""
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



STEP6 = Path(r"K:\Monthly surface water dynamics TIB\supervised classification\step6_explaination")
CLIMATE_SOURCE = "CHM"
MODEL_ROOT = STEP6 / f"QTP_RF_SHAP_GRID_SIZES_{CLIMATE_SOURCE}_2016_2025"
INPUT_CSV = STEP6 / "QTP_nine_variables_PE_VIF_TOL_2016_2025" / "grid25km_nine_variables_2016_2025.csv"

BASIN_REFERENCE = MODEL_ROOT / "grid_Total" / "grid_centres_driver_attributes.csv"
OUTPUT = STEP6 / "QTP_RF_SHAP_PE9_2016_2025" / "validation"

N_JOBS = min(6, os.cpu_count() or 1)       
BATCH_SIZE = 48                          
SEED = 42
N_TEST_PER_CALENDAR_MONTH = 2             
STL_PARAMS = dict(period=12, seasonal=13, robust=False)

RF_PARAMS = dict(n_estimators=200, max_depth=5, min_samples_leaf=3,
                 max_features=1.0, bootstrap=True, max_samples=0.8,
                 random_state=SEED, n_jobs=1)
FEATURES = ['P_mm', 'T_C', 'SMELT_mm', 'PE_mm', 'SRO_mm', 'SSRO_mm', 'SWNET_Wm2', 'SM0_100_m3m3_lag1', 'SWE_mm_lag1']
AREA_COLS = ["Total_area_km2", "Small_area_km2"]
TARGETS = ("Small_Seasonal", "Total_Seasonal", "Total_Area")

VERSION = "2.0_PE9"
DATES = pd.date_range("2016-01-01", "2025-12-01", freq="MS")
MONTHS = DATES.month.to_numpy()


def log(message):
    print(f"{datetime.now():%H:%M:%S} {message}", flush=True)


def display_number(value):
    return f"{value:.3f}" if value is not None and np.isfinite(value) else "NA (constant response)"


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_indices():
    """Use the same random test dates for all grids."""
    rng = np.random.default_rng(SEED)
    chosen = [rng.choice(np.flatnonzero(MONTHS == m),
                         N_TEST_PER_CALENDAR_MONTH, replace=False)
              for m in range(1, 13)]
    test = np.sort(np.concatenate(chosen))
    train = np.setdiff1d(np.arange(120), test)
    return {"Temporal": (np.arange(96), np.arange(96, 120)),
            "Random_months": (train, test)}


def score(y, predicted):
    y = np.asarray(y, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if y.size == 0 or not np.isfinite(y).all() or not np.isfinite(predicted).all():
        raise ValueError("Scoring data are empty or contain non-finite values.")
    error = predicted - y
    ss = float(np.square(y - y.mean()).sum())
    r2 = float(1 - np.square(error).sum() / ss) if ss > 1e-12 else None
    return {"R2": r2, "RMSE_km2": float(np.sqrt(np.square(error).mean())),
            "MAE_km2": float(np.abs(error).mean()), "bias_km2": float(error.mean())}


def calendar_baseline(train_y, train_idx, test_idx):
    means = np.array([train_y[MONTHS[train_idx] == m].mean()
                      for m in range(1, 13)])
    return means[MONTHS[test_idx] - 1]


def comparison(y, predicted, baseline, train_mean=0.0):
    result = {"test_" + k: v for k, v in score(y, predicted).items()}
    result.update({"baseline_" + k: v for k, v in score(y, baseline).items()})
    mse_base = float(np.square(np.asarray(y) - baseline).mean())
    result["MSE_skill_vs_calendar"] = (float(1 - np.square(np.asarray(y) - predicted).mean()
                                               / mse_base) if mse_base > 1e-12 else None)
    
    result["training_mean_centered_R2"] = score(np.asarray(y) - train_mean,
                                               np.asarray(predicted) - train_mean)["R2"]
    return result


def load_data():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Input table not found: {INPUT_CSV}")
    header = pd.read_csv(INPUT_CSV, nrows=0).columns
    required = ["zone_id", "date", "model_eligible"] + FEATURES + AREA_COLS
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    optional = [c for c in ["center_x", "center_y", "basin"] if c in header]
    d = pd.read_csv(INPUT_CSV, usecols=required + optional, dtype={"zone_id": str})
    flags = d.model_eligible.astype(str).str.strip().str.lower()
    if not flags.isin(["true", "false", "1", "0", "1.0", "0.0"]).all():
        raise ValueError("Unrecognised or missing model_eligible values.")
    d = d.loc[flags.isin(["true", "1", "1.0"])].copy()
    d["date"] = pd.to_datetime(d.date, errors="raise")
    d = d.sort_values(["zone_id", "date"]).reset_index(drop=True)
    if d.empty or d.duplicated(["zone_id", "date"]).any():
        raise ValueError("Eligible data are empty or contain duplicate grid-month records.")
    ids = d.zone_id.drop_duplicates().to_numpy()
    expected = np.tile(DATES.to_numpy(), len(ids))
    if len(d) != len(ids) * 120 or not np.array_equal(d.date.to_numpy(), expected):
        raise ValueError("Each eligible grid must have 120 consecutive months from 2016 to 2025.")
    for c in FEATURES + AREA_COLS:
        d[c] = pd.to_numeric(d[c], errors="raise")
    if not np.isfinite(d[FEATURES + AREA_COLS].to_numpy(float)).all():
        raise ValueError("Inputs contain NaN or Inf; months are not interpolated or removed.")
    if (d[AREA_COLS].to_numpy() < 0).any() or (d.Small_area_km2 > d.Total_area_km2 + 1e-6).any():
        raise ValueError("Negative area or small-water area greater than total area.")
    if "basin" not in d:
        if BASIN_REFERENCE.is_file():
            ref = pd.read_csv(BASIN_REFERENCE, usecols=["zone_id", "basin"], dtype={"zone_id": str})
            if not ref.zone_id.is_unique:
                raise ValueError("Duplicate zone_id in basin reference.")
            d["basin"] = d.zone_id.map(ref.set_index("zone_id").basin)
        else:
            log("[BASIN] Assigning grids by largest basin intersection.")
            import fiona
            from shapely.geometry import shape
            from shapely.ops import transform
            from shapely.strtree import STRtree
            from pyproj import Transformer
            grid_file=STEP6/'QTP_grid25km_waterbody_sizes_2016_2025_parallel'/'QTP_grid25km_polygons.gpkg'
            basin_file=Path(r'D:\phdl2\QTP surface water dynamics\regions\QTP_watershed.shp')
            with fiona.open(grid_file,layer='grid25km') as src:
                grid_crs=src.crs_wkt or src.crs
                geometries={str(f['properties']['zone_id']):shape(f['geometry']) for f in src}
            with fiona.open(basin_file) as src:
                project=Transformer.from_crs(src.crs_wkt or src.crs,grid_crs,always_xy=True)
                polygons=[];names=[]
                for f in src:
                    polygons.append(transform(project.transform,shape(f['geometry'])))
                    names.append(str(f['properties']['Name']).replace('QIB','QlB'))
            tree=STRtree(polygons);assigned={}
            for zone in ids:
                geom=geometries[str(zone)];intersections={}
                for j in tree.query(geom):
                    name=names[int(j)];intersections[name]=intersections.get(name,0)+geom.intersection(polygons[int(j)]).area
                if not intersections or max(intersections.values())<=0:raise ValueError(f'No matching basin for grid: {zone}')
                assigned[str(zone)]=max(intersections,key=intersections.get)
            d['basin']=d.zone_id.map(assigned)
    if d.basin.isna().any():
        raise ValueError("Some eligible grids have no assigned basin.")
    d["basin"] = d.basin.astype(str).replace({"QIB": "QlB"})
    if not d.groupby("zone_id").basin.nunique().eq(1).all():
        raise ValueError("Basin assignment changes within a grid.")
    for c in ["center_x", "center_y"]:
        if c not in d:
            d[c] = np.nan
        else:
            if not np.isfinite(d[c].to_numpy(float)).all() or not d.groupby("zone_id")[c].nunique().eq(1).all():
                raise ValueError(f"{c} is missing or changes over time.")
    signature_cols = ["zone_id", "date", "basin", "center_x", "center_y"] + FEATURES + AREA_COLS
    hashed = pd.util.hash_pandas_object(d[signature_cols], index=False).to_numpy().tobytes()
    signature = dict(version=VERSION, data_sha256=hashlib.sha256(hashed).hexdigest(),
                     climate_source=CLIMATE_SOURCE, input_csv=str(INPUT_CSV),
                     rf=RF_PARAMS, stl=STL_PARAMS, features=FEATURES, targets=TARGETS,
                     random_seed=SEED, random_test_per_calendar_month=N_TEST_PER_CALENDAR_MONTH,
                     numpy=np.__version__, sklearn=sklearn.__version__, statsmodels=statsmodels.__version__)
    return d, signature


def fit_grid(zone_id, x, total, small, basin, cx, cy, rf_params=None, stl_params=None, splits=None):
    """Fit three responses under two validation splits."""
    rf_params = RF_PARAMS if rf_params is None else rf_params
    stl_params = STL_PARAMS if stl_params is None else stl_params
    splits = split_indices() if splits is None else splits
    full_s = {"Small": STL(small, **stl_params).fit().seasonal,
              "Total": STL(total, **stl_params).fit().seasonal}
    train_s = {"Small": STL(small[:96], **stl_params).fit().seasonal,
               "Total": STL(total[:96], **stl_params).fit().seasonal}
    rows, predictions = [], []
    for scheme, (tr, te) in splits.items():
        for target in TARGETS:
            if target == "Total_Area":
                train_y, test_y = total[tr], total[te]
                label_source = "Observed_monthly_area"
            else:
                size = target.split("_")[0]
                train_y = train_s[size] if scheme == "Temporal" else full_s[size][tr]
                test_y = full_s[size][te]
                label_source = ("Training_STL_fit_full_STL_test_reference" if scheme == "Temporal"
                                else "Fixed_full_period_STL_reconstruction")
            
            mu = float(train_y.mean())
            model = RandomForestRegressor(**rf_params)
            model.fit(x[tr], train_y - mu)
            train_pred = model.predict(x[tr]) + mu
            pred = model.predict(x[te]) + mu
            base = calendar_baseline(train_y, tr, te)
            result = comparison(test_y, pred, base, mu)
            train_score = score(train_y, train_pred)
            status = ("No_water" if (small.max() if target.startswith("Small") else total.max()) <= 1e-12
                      else "Constant_test_target" if result["test_R2"] is None else "Scored")
            row = dict(row_type="grid", scope=str(zone_id), zone_id=str(zone_id), basin=basin,
                       center_x=float(cx) if np.isfinite(cx) else None,
                       center_y=float(cy) if np.isfinite(cy) else None,
                       scheme=scheme, target=target, label_source=label_source, status=status,
                       n_grids=1, n_train=len(tr), n_test=len(te),
                       train_R2=train_score["R2"], train_RMSE_km2=train_score["RMSE_km2"],
                       target_test_SD_km2=float(np.std(test_y)),
                       negative_area_predictions=int((pred < 0).sum()) if target == "Total_Area" else None,
                       **result)
            rows.append(row)
            predictions.append(dict(scheme=scheme, target=target,
                                    y=np.asarray(test_y).tolist(), pred=pred.tolist(),
                                    baseline=base.tolist(), train_mean=mu))
    return dict(zone_id=str(zone_id), basin=basin, rows=rows, predictions=predictions)


def open_checkpoint(path, signature):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS grids (zone_id TEXT PRIMARY KEY, payload BLOB)")
    encoded = json.dumps(signature, sort_keys=True, ensure_ascii=False)
    old = con.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
    if old is not None and old[0] != encoded:
        con.close()
        raise RuntimeError("Checkpoint differs from current data or settings; choose a new output directory.")
    if old is None:
        con.execute("INSERT INTO meta VALUES (?, ?)", ("signature", encoded))
        con.commit()
    return con


def unpack(payload):
    return json.loads(zlib.decompress(payload).decode("utf-8"))


def aggregate_results(con, ids):
    
    records = [unpack(con.execute("SELECT payload FROM grids WHERE zone_id=?", (str(z),)).fetchone()[0])
               for z in ids]
    rows = [row for record in records for row in record["rows"]]
    basins = sorted(set(r["basin"] for r in records) - {"Unassigned", "QTP"})
    for scheme, (_, test_idx) in split_indices().items():
        for target in TARGETS:
            for region in ["QTP"] + basins:
                selected = records if region == "QTP" else [r for r in records if r["basin"] == region]
                p = [next(v for v in r["predictions"] if v["scheme"] == scheme and v["target"] == target)
                     for r in selected]
                y = np.array([v["y"] for v in p]); pred = np.array([v["pred"] for v in p])
                base = np.array([v["baseline"] for v in p]); mu = np.array([v["train_mean"] for v in p])
                local = [next(v for v in r["rows"] if v["scheme"] == scheme and v["target"] == target)
                         for r in selected]
                defined = [v["test_R2"] for v in local if v["test_R2"] is not None]
                skill_defined = [v["MSE_skill_vs_calendar"] for v in local
                                 if v["MSE_skill_vs_calendar"] is not None]
                coverage = dict(n_grids=len(selected), n_defined_grid_R2=len(defined),
                                n_grid_R2_above_0=sum(v > 0 for v in defined),
                                n_grid_R2_at_least_0p2=sum(v >= .2 for v in defined),
                                median_grid_R2=float(np.median(defined)) if defined else None,
                                n_grids_beating_calendar=sum(v > 0 for v in skill_defined))
                
                for kind in ["regional_sum", "pooled_samples"]:
                    if kind == "regional_sum":
                        obs, estimate, reference = y.sum(axis=0), pred.sum(axis=0), base.sum(axis=0)
                        centred_mean = mu.sum()
                    else:
                        obs, estimate, reference = y.ravel(), pred.ravel(), base.ravel()
                        centred_mean = np.repeat(mu, len(test_idx))
                    rows.append(dict(row_type=kind, scope=region, zone_id=None,
                                     basin=region, scheme=scheme, target=target,
                                     label_source=local[0]["label_source"], status="Aggregated_all_input_grids",
                                     n_train=96 if kind == "regional_sum" else 96 * len(selected),
                                     n_test=len(obs), train_R2=None, train_RMSE_km2=None,
                                     target_test_SD_km2=float(np.std(obs)),
                                     negative_area_predictions=int((estimate < 0).sum()) if target == "Total_Area" else None,
                                     **coverage, **comparison(obs, estimate, reference, centred_mean)))
    return pd.DataFrame(rows)


def explanation(signature, csv_hash, n_grids):
    random_dates = ", ".join(DATES[split_indices()["Random_months"][1]].strftime("%Y-%m"))
    text = f"""Grid-specific random forest validation

The output contains grid, basin and plateau validation results for {n_grids} eligible grids.
Each grid has 120 monthly observations and three responses: total area, total seasonal
component and small-water seasonal component.

Temporal validation uses 2016–2023 for training and 2024–2025 for testing. For seasonal
responses the training labels are decomposed from the training period and the test
reference is decomposed from the full record.

Random-month validation withholds two observations per calendar month. All grids use
the same 24 test dates: {random_dates}. Seasonal responses are decomposed from the
complete record, so their test scores assess reconstruction of those components.

The table reports grid-level, regional-sum and pooled-sample metrics. Calendar-month
baselines use only training labels. R² is undefined for constant test responses.
"""
    marker = dict(signature=signature, csv_sha256=csv_hash)
    return text + "\n__COMPLETION_JSON__\n" + json.dumps(marker, ensure_ascii=False, sort_keys=True)


def main():
    if N_TEST_PER_CALENDAR_MONTH != 2:
        raise ValueError("Both splits require 96 training and 24 test months; set N_TEST_PER_CALENDAR_MONTH to 2.")
    if N_JOBS < 1 or BATCH_SIZE < 1:
        raise ValueError("N_JOBS and BATCH_SIZE must be positive.")
    log(f"[START] Grid-specific RF; two validation schemes; {N_JOBS} workers")
    d, signature = load_data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path, note_path = OUTPUT / "validation_results.csv", OUTPUT / "说明.txt"
    checkpoint = OUTPUT / "_progress.sqlite"
    if csv_path.exists() and note_path.exists():
        try:
            marker = json.loads(note_path.read_text(encoding="utf-8").split("__COMPLETION_JSON__\n", 1)[1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError("Existing results lack a valid completion signature; choose a new output directory.") from exc
        expected = json.loads(json.dumps(signature))  
        if marker["signature"] != expected or file_hash(csv_path) != marker["csv_sha256"]:
            raise RuntimeError("Existing results differ from the current input or settings; choose a new output directory.")
        if checkpoint.exists():
            con = open_checkpoint(checkpoint, signature)
            con.close()
            checkpoint.unlink()
        log(f"[SKIP] Completed and consistent: {csv_path}")
        return
    if (csv_path.exists() or note_path.exists()) and not checkpoint.exists():
        raise RuntimeError("Incomplete output without a checkpoint; choose a new output directory.")
    con = open_checkpoint(checkpoint, signature)
    try:
        ids = d.zone_id.drop_duplicates().to_numpy()
        done = {r[0] for r in con.execute("SELECT zone_id FROM grids")}
        if not done.issubset(set(ids)):
            raise RuntimeError("The checkpoint contains grids not present in the input.")
        x = d[FEATURES].to_numpy(float).reshape(len(ids), 120, len(FEATURES))
        area = d[AREA_COLS].to_numpy(float).reshape(len(ids), 120, 2)
        metadata = d.iloc[::120][["zone_id", "basin", "center_x", "center_y"]].reset_index(drop=True)
        pending = [i for i, z in enumerate(ids) if str(z) not in done]
        log(f"[DATA] {len(ids):,} grids; completed {len(done):,}; pending {len(pending):,}")
        with Parallel(n_jobs=N_JOBS, backend="loky", inner_max_num_threads=1) as parallel:
            for first in range(0, len(pending), BATCH_SIZE):
                batch = pending[first:first + BATCH_SIZE]
                results = parallel(delayed(fit_grid)(ids[i], x[i], area[i, :, 0], area[i, :, 1],
                                                    metadata.at[i, "basin"], metadata.at[i, "center_x"],
                                                    metadata.at[i, "center_y"], RF_PARAMS,
                                                    STL_PARAMS, split_indices()) for i in batch)
                for result in results:
                    payload = zlib.compress(json.dumps(result, allow_nan=False).encode("utf-8"))
                    con.execute("INSERT INTO grids VALUES (?, ?)", (result["zone_id"], payload))
                con.commit()
                log(f"[RF] Completed {len(done) + first + len(batch):,}/{len(ids):,} grids (six models per grid)")
        result = aggregate_results(con, ids)
        temp = csv_path.with_suffix(".tmp.csv")
        result.to_csv(temp, index=False, encoding="utf-8-sig", float_format="%.12g")
        temp_note = note_path.with_suffix(".tmp.txt")
        temp_note.write_text(explanation(signature, file_hash(temp), len(ids)), encoding="utf-8")
        temp.replace(csv_path)
        temp_note.replace(note_path)
        for scheme in ["Temporal", "Random_months"]:
            qtp = result[(result.row_type == "regional_sum") & (result.scope == "QTP")
                         & (result.scheme == scheme)]
            for row in qtp.itertuples():
                log(f"[{scheme}/{row.target}] QTP total R²={display_number(row.test_R2)}; "
                    f"median grid R²={display_number(row.median_grid_R2)}; "
                    f"calendar baseline R²={display_number(row.baseline_R2)}")
    finally:
        con.close()
    checkpoint.unlink()
    log(f"[DONE] {OUTPUT}; validation results complete")


if __name__ == "__main__":
    main()
