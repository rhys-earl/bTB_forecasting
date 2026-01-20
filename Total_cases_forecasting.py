# Total_cases_forecasting.py
# ------------------------------------------------------------
# Rolling seasonal forecast for ONE season ending in December of Y (monthly)
# Origins: Sep (Y-1) .. Aug (Y)  => horizons 15..4 months
# Evaluate ALL leads (1..h) and save per-lead metrics (WIS, MAPE, etc.)
#
# CLI usage:
#   python Total_cases_forecasting.py ARIMA 2016
#   python Total_cases_forecasting.py Prophet 2016
#   python Total_cases_forecasting.py "Linear Regress" 2020
#
# You only specify:
#   1) model
#   2) season_end_year (December year)
#
# Everything else is configured below as constants (easy to edit).
#
# This script ENFORCES probabilistic forecasts:
#   - always calls predict(..., num_samples=NUM_SAMPLES)
#   - requires forecast TimeSeries to have n_samples > 1
#     (otherwise raises with a clear message)
# ------------------------------------------------------------

import os
import argparse
import logging
import numpy as np
import pandas as pd
import pyreadr

import torch
from darts import TimeSeries
from darts.utils.likelihood_models import QuantileRegression
from darts.models import (
    StatsForecastAutoARIMA,
    Prophet,
    DLinearModel,
    NLinearModel,
    TiDEModel,
    NHiTSModel,
    NBEATSModel,
    LinearRegressionModel,
    StatsForecastAutoETS,
)

# -----------------------------
# USER-EDITABLE SETTINGS
# -----------------------------
RDS_PATH = "all_cases_20_Jan_2026.rds"   # <-- your file
OUT_DIR = "results"                     # <-- change if you want
DATE_COL = "merged_date_min"            # <-- change if needed

# Probabilistic forecasting settings
NUM_SAMPLES = 1000                      # <-- always used (enforced)
MIN_TRAIN_MONTHS = 36                   # <-- minimum expanding-window history
LAG_LENGTH_OVERRIDE = None              # <-- set int to force; else auto

# Quantiles for WIS + output columns
QUANTILES = [
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99
]

# -----------------------------
# Logging + compute config
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

if torch.cuda.is_available():
    pl_trainer_kwargs = {"accelerator": "gpu", "devices": [1]}
    logging.info("GPU detected. Using GPU for training.")
else:
    pl_trainer_kwargs = {"accelerator": "cpu", "devices": 1}
    logging.info("No GPU detected. Using CPU for training.")

torch.set_num_threads(4)

# -----------------------------
# Model factory (monthly seasonality => season_length=12)
# Deep models use QuantileRegression -> already probabilistic.
# For stats models (Prophet/ARIMA/ETS), we still request num_samples at predict time.
# -----------------------------
def get_model(model_name: str, forecast_length: int, lag_length: int):
    if model_name == "TiDE":
        return TiDEModel(
            input_chunk_length=lag_length,
            output_chunk_length=forecast_length,
            likelihood=QuantileRegression(quantiles=QUANTILES),
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
    elif model_name == "NBEATS":
        return NBEATSModel(
            input_chunk_length=lag_length,
            output_chunk_length=forecast_length,
            likelihood=QuantileRegression(quantiles=QUANTILES),
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
    elif model_name == "NHiTS":
        return NHiTSModel(
            input_chunk_length=lag_length,
            output_chunk_length=forecast_length,
            likelihood=QuantileRegression(quantiles=QUANTILES),
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
    elif model_name == "DLinear":
        return DLinearModel(
            input_chunk_length=lag_length,
            output_chunk_length=forecast_length,
            likelihood=QuantileRegression(quantiles=QUANTILES),
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
    elif model_name == "NLinear":
        return NLinearModel(
            input_chunk_length=lag_length,
            output_chunk_length=forecast_length,
            likelihood=QuantileRegression(quantiles=QUANTILES),
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
    elif model_name == "Linear Regress":
        return LinearRegressionModel(
            lags=lag_length,
            output_chunk_length=forecast_length,
            likelihood="quantile",
            quantiles=QUANTILES,
        )
    elif model_name == "Prophet":
        # Note: probabilistic behaviour depends on Darts/Prophet backend;
        # we enforce num_samples at predict time and fail if n_samples==1.
        return Prophet(yearly_seasonality=True)
    elif model_name == "ARIMA":
        return StatsForecastAutoARIMA(season_length=12)
    elif model_name == "ETS":
        return StatsForecastAutoETS(season_length=12)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# -----------------------------
# WIS components
# -----------------------------
def calculate_wis_components(row: pd.Series):
    alphas = QUANTILES
    y = row["actual"]
    weight = 1

    wis_total = 0.0
    sharpness_total = 0.0
    calibration_total = 0.0
    below_count = 0
    above_count = 0
    total_intervals = len(alphas) - 1

    for i in range(total_intervals):
        l = row[f"forecast_{int(alphas[i] * 1000)}"]
        u = row[f"forecast_{int(alphas[i + 1] * 1000)}"]
        alpha = alphas[i + 1] - alphas[i]

        sharpness = 0.5 * (u - l)
        sharpness_total += sharpness

        calibration = (2 / alpha) * max(0.0, l - y) + (2 / alpha) * max(0.0, y - u)
        calibration_total += calibration

        wis_total += weight * (sharpness + calibration)

        if y < l:
            below_count += 1
        elif y > u:
            above_count += 1

    bias = (below_count - above_count) / total_intervals
    return wis_total, sharpness_total, calibration_total, bias

# -----------------------------
# Load RDS -> monthly series
# Uses 'case' column if numeric; otherwise counts rows per month.
# Missing months filled with 0.
# -----------------------------
def load_cases_monthly(rds_path: str, date_col: str) -> pd.Series:
    result = pyreadr.read_r(rds_path)
    df = next(iter(result.values()))

    if date_col not in df.columns:
        raise ValueError(
            f"DATE_COL='{date_col}' not found in RDS. Available columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    case_col = None
    if "case" in df.columns:
        s = pd.to_numeric(df["case"], errors="coerce")
        if s.notna().any():
            case_col = "case"

    if case_col is not None:
        monthly = df.groupby("month")[case_col].sum().astype(float)
        logging.info("Using column 'case' and summing per month.")
    else:
        monthly = df.groupby("month").size().astype(float)
        logging.info("No usable numeric 'case' column found; counting rows per month.")

    monthly = monthly.sort_index()
    full_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_idx, fill_value=0.0)
    monthly.index = pd.DatetimeIndex(monthly.index, freq="MS")
    monthly.name = "cases"
    return monthly

# -----------------------------
# Auto lag length
# -----------------------------
def choose_lag_length(monthly_series: pd.Series) -> int:
    if LAG_LENGTH_OVERRIDE is not None:
        return int(LAG_LENGTH_OVERRIDE)

    n = len(monthly_series)
    if n >= 60:
        return 24
    if n >= 36:
        return 18
    if n >= 24:
        return 12
    return max(4, min(12, n // 2))

# -----------------------------
# integer months between Periods
# end_p later than origin_p
# -----------------------------
def months_ahead(origin_p: pd.Period, end_p: pd.Period) -> int:
    return int(end_p.ordinal - origin_p.ordinal)

# -----------------------------
# Enforce probabilistic forecast output
# -----------------------------
def predict_probabilistic(model, h: int) -> TimeSeries:
    # Always request samples
    fc = model.predict(h, num_samples=int(NUM_SAMPLES))
    n_samp = getattr(fc, "n_samples", 1)

    # Fail fast if model did not return a stochastic forecast
    if n_samp <= 1:
        raise RuntimeError(
            "Model forecast is deterministic (n_samples=1) even though num_samples was requested.\n"
            "This means quantiles/WIS are not valid for this model in this configuration.\n"
            "Try a different model, or check Darts/Prophet/StatsForecast versions.\n"
            f"Requested NUM_SAMPLES={NUM_SAMPLES}, got n_samples={n_samp}."
        )
    return fc

# -----------------------------
# Seasonal rolling forecast for ONE season ending Dec of Y
# -----------------------------
def run_one_season_to_december(
    series_pd: pd.Series,      # monthly, freq MS
    model_name: str,
    season_end_year: int,
    lag_length: int,
):
    Y = int(season_end_year)
    end_period = pd.Period(f"{Y}-12", freq="M")
    last_target_ts = end_period.to_timestamp(how="start")

    if last_target_ts not in series_pd.index:
        raise ValueError(
            f"Data does not include December {Y}. Last available month is {series_pd.index.max().date()}."
        )

    origin_start = pd.Period(f"{Y-1}-09", freq="M")
    origin_end = pd.Period(f"{Y}-08", freq="M")
    origin_periods = pd.period_range(origin_start, origin_end, freq="M")

    rows = []

    for origin_p in origin_periods:
        h = months_ahead(origin_p, end_period)  # int 15..4
        if h <= 0:
            continue

        origin_ts = origin_p.to_timestamp(how="start")
        if origin_ts not in series_pd.index:
            continue

        train_slice = series_pd.loc[:origin_ts]

        if MIN_TRAIN_MONTHS is not None and len(train_slice) < MIN_TRAIN_MONTHS:
            continue
        if len(train_slice) < max(8, lag_length + 1):
            continue

        train_ts = TimeSeries.from_series(train_slice)

        model = get_model(model_name=model_name, forecast_length=h, lag_length=lag_length)
        model.fit(train_ts)

        # --- probabilistic forecast (enforced) ---
        fc = predict_probabilistic(model, h)
        # ----------------------------------------

        for k in range(1, h + 1):
            target_p = origin_p + k
            target_ts = target_p.to_timestamp(how="start")
            y_true = float(series_pd.loc[target_ts])

            entry = {
                "season_end_year": Y,
                "origin_year": int(origin_p.year),
                "origin_month": int(origin_p.month),
                "origin_timestamp": origin_ts,
                "target_timestamp": target_ts,
                "lead": k,
                "horizon": int(h),
                "actual": y_true,
                "model": model_name,
                "lag_length": lag_length,
                "num_samples": int(NUM_SAMPLES),
            }

            # Quantiles at lead k (index k-1)
            for q in QUANTILES:
                entry[f"forecast_{int(q * 1000)}"] = float(fc.quantile(q).values()[k - 1, 0])

            rows.append(entry)

        logging.info(
            f"Origin {origin_p} -> Dec {Y}: horizon={h}, "
            f"trained on {train_slice.index.min().date()}..{train_slice.index.max().date()} ({len(train_slice)} months)"
        )

    df_fc = pd.DataFrame(rows)
    if df_fc.empty:
        raise RuntimeError("No forecasts generated for this season (likely insufficient history).")

    # Metrics per target row
    df_fc[["WIS_all", "Sharpness", "Calibration", "Bias"]] = df_fc.apply(
        calculate_wis_components, axis=1, result_type="expand"
    )

    # Point forecast (median)
    yhat = df_fc["forecast_500"].astype(float)
    y = df_fc["actual"].astype(float)
    ae = (yhat - y).abs()

    df_fc["AE"] = ae
    df_fc["SE"] = (yhat - y) ** 2

    df_fc["MAPE"] = np.where(y != 0, (ae / np.abs(y)) * 100.0, np.nan)
    denom = (np.abs(y) + np.abs(yhat))
    df_fc["sMAPE"] = np.where(denom != 0, (2.0 * ae / denom) * 100.0, np.nan)

    df_fc["wMAPE_num"] = ae
    df_fc["wMAPE_den"] = np.abs(y)

    return df_fc

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seasonal rolling probabilistic forecast for ONE season ending in Dec of year Y."
    )
    parser.add_argument(
        "model",
        type=str,
        help='Model name: TiDE, NBEATS, NHiTS, DLinear, NLinear, Prophet, ARIMA, ETS, "Linear Regress"',
    )
    parser.add_argument(
        "season_end_year",
        type=int,
        help="December year Y. Runs origins Sep (Y-1) .. Aug (Y), forecasting to Dec Y.",
    )
    args = parser.parse_args()

    # Load monthly cases
    monthly_cases = load_cases_monthly(RDS_PATH, DATE_COL)
    logging.info(f"Monthly series spans {monthly_cases.index.min().date()} to {monthly_cases.index.max().date()}")
    logging.info(f"Total months: {len(monthly_cases)}")

    # Choose lag length automatically (unless overridden)
    lag_length = choose_lag_length(monthly_cases)
    logging.info(f"Using lag_length={lag_length} months")
    logging.info(f"Enforcing NUM_SAMPLES={NUM_SAMPLES} for probabilistic forecasts")

    # Run the season
    df_fc = run_one_season_to_december(
        series_pd=monthly_cases,
        model_name=args.model,
        season_end_year=args.season_end_year,
        lag_length=lag_length,
    )

    # Output paths
    out_dir = os.path.join(OUT_DIR, args.model.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)

    detailed_path = os.path.join(
        out_dir,
        f"seasonal_rolling{args.season_end_year}.csv"
    )
    df_fc.to_csv(detailed_path, index=False)
    logging.info(f"Saved detailed results: {detailed_path}")

    # Summary: by origin_month and lead for this one season
    summary = df_fc.groupby(["origin_month", "lead"]).agg(
        n=("actual", "count"),
        mean_WIS=("WIS_all", "mean"),
        mean_AE=("AE", "mean"),
        mean_MAPE=("MAPE", "mean"),
        mean_sMAPE=("sMAPE", "mean"),
        wMAPE_num=("wMAPE_num", "sum"),
        wMAPE_den=("wMAPE_den", "sum"),
    ).reset_index()

    summary["wMAPE_pct"] = np.where(
        summary["wMAPE_den"] != 0, (summary["wMAPE_num"] / summary["wMAPE_den"]) * 100.0, np.nan
    )

    summary_path = os.path.join(
        out_dir,
        f"seasonal_summary_{args.season_end_year}.csv"
    )
    summary.to_csv(summary_path, index=False)
    logging.info(f"Saved summary results: {summary_path}")

    logging.info("Done.")
