"""Analyse seasonal patterns in plateau and watershed water area."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd

BASE = Path(r"K:\Monthly surface water dynamics TIB\supervised classification\step4_analysis")
COLS = {
    "Total": "Total_surface_water_area_km2",
    "Small": "Small_waterbody_area_km2",
    "Medium": "Medium_waterbody_area_km2",
    "Large": "Large_waterbody_area_km2",
}
STL_CONFIG = dict(period=12, seasonal=7, trend=23, low_pass=13,
                  seasonal_deg=1, trend_deg=1, low_pass_deg=1,
                  seasonal_jump=1, trend_jump=1, low_pass_jump=1,
                  robust=False)
FIT_CONFIG = dict(inner_iter=5, outer_iter=0)
# Absolute tolerance in km2, with a scale-dependent floating point allowance.
ATOL = 1e-7
RTOL = 1e-10
EXPECTED = pd.date_range("2016-01-01", "2025-12-01", freq="MS")


def prepare(df, watershed=False):
    keys = (["Name"] if watershed else []) + ["Year", "Month"]
    needed = keys + list(COLS.values())
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.copy()
    for col in ["Year", "Month"]:
        values = pd.to_numeric(df[col], errors="raise")
        if values.isna().any() or (values != np.floor(values)).any():
            raise ValueError(f"Invalid integer column: {col}")
        df[col] = values.astype(int)
    if watershed:
        if df.Name.isna().any():
            raise ValueError("Missing watershed name")
        df["Name"] = df.Name.astype(str)
    if df.duplicated(keys).any():
        raise ValueError(f"Duplicate records: {keys}")
    df["Date"] = pd.to_datetime(dict(year=df.Year, month=df.Month, day=1))
    for col in COLS.values():
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError(f"Missing, infinite or negative area: {col}; no zero filling/interpolation")
    groups = df.groupby("Name", sort=True) if watershed else [("QTP", df)]
    for name, group in groups:
        dates = pd.DatetimeIndex(group.sort_values("Date").Date)
        if not dates.equals(EXPECTED):
            raise ValueError(f"{name}: requires exactly 120 months, 2016-01 to 2025-12")
    return df.sort_values(keys).reset_index(drop=True)


def check_workbook(csv_df, excel_df, watershed):
    keys = (["Name"] if watershed else []) + ["Year", "Month"]
    if not csv_df[keys].equals(excel_df[keys]):
        raise ValueError("CSV/Excel key mismatch")
    # Verify all source numeric fields, including diagnostics and counts.
    for col in csv_df.columns:
        if col in keys + ["Date"]:
            continue
        if col not in excel_df:
            raise ValueError(f"Excel missing {col}")
        a = pd.to_numeric(csv_df[col], errors="raise").to_numpy(float)
        b = pd.to_numeric(excel_df[col], errors="raise").to_numpy(float)
        if not np.allclose(a, b, rtol=1e-12, atol=1e-8, equal_nan=True):
            raise ValueError(f"CSV/Excel mismatch: {col}")


def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")


def metrics(total, summed):
    diff = total - summed
    limit = ATOL + RTOL * np.maximum(np.abs(total), np.abs(summed))
    return dict(max_abs_difference_km2=float(np.max(np.abs(diff))),
                rmse_km2=float(np.sqrt(np.mean(diff**2))),
                mean_difference_km2=float(np.mean(diff)),
                all_months_pass=bool(np.all(np.abs(diff) <= limit)))


def seasonality_metrics(y):
    """Complete 10 x 12 monthly data; population SD throughout."""
    annual = np.asarray(y, dtype=float).reshape(10, 12)
    sigma = float(annual.std(axis=1, ddof=0).mean())
    delta = float(annual.mean(axis=1).std(ddof=0))
    denom = sigma + delta
    c = 100 * sigma / denom if denom > 0 else np.nan
    dominance = ("Undefined" if not np.isfinite(c) else
                 "Seasonality" if c > 50 else "Non-seasonal" if c < 50 else "Equal")
    return dict(MeanArea_km2=float(annual.mean()), SigmaBar_km2=sigma,
                Delta_km2=delta, Cseasonality_pct=c, Dominance=dominance)


def size_contributions(total, parts):
    """Population covariance / independently fitted total seasonal variance."""
    total = np.asarray(total, float)
    parts = np.asarray(parts, float)
    if not metrics(total, parts.sum(axis=0))["all_months_pass"]:
        raise ValueError("Seasonal additivity failed; contribution allocation stopped")
    centred_total = total - total.mean()
    variance = float(np.mean(centred_total**2))
    # Treat signals indistinguishable from numerical roundoff as undefined.
    if float(np.max(np.abs(centred_total))) <= ATOL:
        return variance, np.full(3, np.nan), np.full(3, np.nan), "Undefined_near_zero_variance"
    cov = np.mean((parts - parts.mean(axis=1, keepdims=True)) * centred_total, axis=1)
    pct = 100 * cov / variance
    if not np.isclose(pct.sum(), 100, atol=1e-6, rtol=0):
        raise ValueError("Contributions do not sum to 100%; check numerical conditioning")
    return variance, cov, pct, "Valid"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=BASE / "QTP_waterbody_statistics_0p1")
    parser.add_argument("--output-dir", type=Path, default=BASE / "QTP_seasonality_results")
    args = parser.parse_args()
    qpath = args.input_dir / "QTP_monthly_waterbody_statistics.csv"
    wpath = args.input_dir / "Watershed_monthly_waterbody_statistics.csv"
    xpath = args.input_dir / "QTP_waterbody_statistics_2016_2025.xlsx"
    qtp = prepare(pd.read_csv(qpath, encoding="utf-8-sig"))
    watersheds = prepare(pd.read_csv(wpath, encoding="utf-8-sig"), True)
    with pd.ExcelFile(xpath) as book:
        check_workbook(qtp, prepare(pd.read_excel(book, "QTP_monthly")), False)
        check_workbook(watersheds, prepare(pd.read_excel(book, "Watershed_monthly"), True), True)

    from statsmodels.tsa.seasonal import STL
    import statsmodels
    groups = [("QTP", qtp)] + [(str(n), g) for n, g in watersheds.groupby("Name", sort=True)]
    components, metrics_rows, contribution_rows = [], [], []
    for region, frame in groups:
        frame = frame.sort_values("Date")
        y = {k: frame[v].to_numpy(dtype=np.float64) for k, v in COLS.items()}
        if not metrics(y["Total"], y["Small"] + y["Medium"] + y["Large"])["all_months_pass"]:
            raise ValueError(f"{region}: total area is not the sum of three class areas")
        fits = {k: STL(v, **STL_CONFIG).fit(**FIT_CONFIG) for k, v in y.items()}
        for label, fit in fits.items():
            for values in [fit.trend, fit.seasonal, fit.resid]:
                if not np.isfinite(values).all():
                    raise ValueError(f"{region}/{label}: non-finite STL result")
            components.append(pd.DataFrame(dict(
                Region=region, Date=frame.Date.dt.strftime("%Y-%m").to_numpy(),
                SizeClass=label, Area_km2=y[label], Trend_km2=fit.trend,
                Seasonal_km2=fit.seasonal, Residual_km2=fit.resid)))
            metrics_rows.append(dict(Region=region, SizeClass=label,
                                     **seasonality_metrics(y[label])))
        variance, cov, pct, status = size_contributions(
            fits["Total"].seasonal,
            np.array([fits[k].seasonal for k in ["Small", "Medium", "Large"]]))
        for i, label in enumerate(["Small", "Medium", "Large"]):
            contribution_rows.append(dict(Region=region, SizeClass=label,
                TotalSeasonalVariance_km4=variance, CovarianceWithTotal_km4=cov[i],
                SeasonalContribution_pct=pct[i], Status=status))

    # Write only after every region has passed validation and decomposition.
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    all_components = pd.concat(components, ignore_index=True)
    save_csv(all_components[all_components.Region == "QTP"], out / "QTP_STL_components.csv")
    save_csv(all_components[all_components.Region != "QTP"], out / "Watershed_STL_components.csv")
    save_csv(pd.DataFrame(metrics_rows), out / "Seasonality_contribution.csv")
    save_csv(pd.DataFrame(contribution_rows), out / "Size_class_seasonal_contribution.csv")
    print(f"Completed: QTP and {len(groups)-1} watersheds; four CSV files.\nOutput: {out}")


if __name__ == "__main__":
    main()

