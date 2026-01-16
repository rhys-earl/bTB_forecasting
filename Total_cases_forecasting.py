# seasonal_rolling_forecast_single_year.py
# ------------------------------------------------------------
# Rolling seasonal forecast for ONE season ending in December of Y
# Origins: Sep (Y-1) .. Aug (Y)  => horizons 15..4 months
# Evaluate ALL leads (1..h) and save per-lead metrics (WIS, MAPE, etc.)
#
# CLI usage:
#   python seasonal_rolling_forecast_single_year.py ARIMA 2016
#   python seasonal_rolling_forecast_single_year.py Prophet 2016
#   python seasonal_rolling_forecast_single_year.py "Linear Regress" 2020
#
# You only specify:
#   1) model
#   2) season_end_year (December year)
#
# Everything else is configured below as constants (easy to edit).
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
CSV_PATH = "all_cases_12_Jan_2026.rds"   # <-- change me
OUT_DIR = "results"                     # <-- change me

DATE_COL = "merged_date_min"            # <-- change me if needed

# If you want to FORCE a lag length, set this to an int (months), e.g., 24.
# If None, it will be chosen automatically from the available history.
LAG_LENGTH_OVERRIDE = None

# If you want to FORCE number of samples for probabilistic models, set an int (e.g., 1000).
# If None, it will be chosen automatically per model.
NUM_SAMPLES_OVERRIDE = 1000

# Optional minimum training history (months). If None, only basic checks apply.
MIN_TRAIN_MONTHS = 36

# Quantiles for probabilistic forecasts
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
# Helpers: model factory
# Monthly seasonality => season_length=12
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
    elif model_name == "Prophet":
        return Prophet(yearly_seasonality=True)
    elif model_name == "ARIMA":
        return StatsForecastAutoARIMA(season_length=12)
    elif model_name == "ETS":
        return StatsForecastAutoETS(season_length=12)
    elif model_name == "Linear Regress":
        return LinearRegressionModel(
            lags=lag_length,
            output_chunk_length=forecast_length,
            likelihood="quantile",
            quantiles=QUANTILES,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def model_is_probabilistic(model_name: str) -> bool:
    return model_name in {"TiDE", "NBEATS", "NHiTS", "DLinear", "NLinear", "Linear Regress"}

def choose_num_samples(model_name: str) -> int:
    if NUM_SAMPLES_OVERRIDE is not None:
        return int(NUM_SAMPLES_OVERRIDE)
    return 1000 if model_is_probabilistic(model_name) else 1

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
# Load case-level RDS -> monthly series
# -----------------------------
def load_cases_monthly(rds_path: str, date_col: str) -> pd.Series:
    result = pyreadr.read_r(rds_path)
    df = next(iter(result.values()))

    if date_col not in df.columns:
        raise ValueError(
            f"DATE_COL='{date_col}' not found in RDS. "
            f"Available columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    # Automatically detect case column
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

    # Fill missing months with 0
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
# Compute integer month horizon between two Periods
# end_p is later than origin_p
# -----------------------------
def months_ahead(origin_p: pd.Period, end_p: pd.Period) -> int:
    # This avoids getting a MonthEnd offset like "<15 * MonthEnds>"
    return int(end_p.ordinal - origin_p.ordinal)

# -----------------------------
# Seasonal rolling forecast for ONE season ending Dec of Y
# -----------------------------
def run_one_season_to_december(
    series_pd: pd.Series,      # monthly, freq MS
    model_name: str,
    season_end_year: int,
    lag_length: int,
    num_samples: int,
):
    Y = int(season_end_year)
    end_period = pd.Period(f"{Y}-12", freq="M")  # December Y
    last_target_ts = end_period.to_timestamp(how="start")

    if last_target_ts not in series_pd.index:
        raise ValueError(
            f"Data does not include December {Y}. Last available month is {series_pd.index.max().date()}."
        )

    origin_start = pd.Period(f"{Y-1}-09", freq="M")  # Sep (Y-1)
    origin_end = pd.Period(f"{Y}-08", freq="M")      # Aug (Y)
    origin_periods = pd.period_range(origin_start, origin_end, freq="M")

    rows = []

    for origin_p in origin_periods:
        h = months_ahead(origin_p, end_period)  # 15..4 as an INT
        if h <= 0:
            continue

        origin_ts = origin_p.to_timestamp(how="start")
        if origin_ts not in series_pd.index:
            continue

        # training = all history up to origin (inclusive)
        train_slice = series_pd.loc[:origin_ts]

        if MIN_TRAIN_MONTHS is not None and len(train_slice) < MIN_TRAIN_MONTHS:
            continue
        if len(train_slice) < max(8, lag_length + 1):
            continue

        train_ts = TimeSeries.from_series(train_slice)

        model = get_model(model_name=model_name, forecast_length=h, lag_length=lag_length)
        model.fit(train_ts)

        # Prophet is deterministic in Darts; don't pass num_samples
        if model_name == "Prophet":
            fc = model.predict(h)
        else:
            fc = model.predict(h, num_samples=num_samples)

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
                "num_samples": num_samples,
            }

            # Quantiles at lead k (index k-1)
            try:
                for q in QUANTILES:
                    entry[f"forecast_{int(q * 1000)}"] = float(fc.quantile(q).values()[k - 1, 0])
            except Exception:
                # deterministic fallback -> replicate across quantiles
                point = float(fc.values()[k - 1, 0])
                for q in QUANTILES:
                    entry[f"forecast_{int(q * 1000)}"] = point

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
        description="Seasonal rolling forecast for ONE season ending in Dec of year Y."
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
    monthly_cases = load_cases_monthly(CSV_PATH, DATE_COL)
    logging.info(f"Monthly series spans {monthly_cases.index.min().date()} to {monthly_cases.index.max().date()}")
    logging.info(f"Total months: {len(monthly_cases)}")

    # Choose lag length + samples (unless overridden at top)
    lag_length = choose_lag_length(monthly_cases)
    num_samples = choose_num_samples(args.model)
    logging.info(f"Using lag_length={lag_length} months")
    logging.info(f"Using num_samples={num_samples}")

    # Run the season
    df_fc = run_one_season_to_december(
        series_pd=monthly_cases,
        model_name=args.model,
        season_end_year=args.season_end_year,
        lag_length=lag_length,
        num_samples=num_samples,
    )

    # Output paths
    out_dir = os.path.join(OUT_DIR, args.model.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)

    detailed_path = os.path.join(
        out_dir,
        f"seasonal_rolling_to_dec_{args.season_end_year}_lag{lag_length}.csv"
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
        f"seasonal_summary_to_dec_{args.season_end_year}_lag{lag_length}.csv"
    )
    summary.to_csv(summary_path, index=False)
    logging.info(f"Saved summary results: {summary_path}")

    logging.info("Done.")
