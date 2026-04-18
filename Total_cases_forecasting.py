# Total_cases_forecasting_v2.py
# ------------------------------------------------------------
# Rolling seasonal forecast aligned to R (fpp3) methodology.
#
# KEY CHANGES FROM v1:
#   1. Origin range: Sep (Y-1) -> Dec (Y), matching R exactly (16 origins per year)
#   2. Max horizon:  16 months (from Sep Y-1 to Dec Y)
#   3. Min horizon:  1 month  (origin = Dec Y, forecasting Jan Y+1 -- but we
#                              filter to year Y only, so effective min = 1)
#   4. Training data: expanding window up to but NOT including month_forecasting_from
#                     (R does: filter(year_month_date < month_forecasting_from))
#   5. Annual total output: for each origin, monthly forecast samples are summed
#                           over the target year only (Jan-Dec Y), then
#                           lag_cumulative_observed_cases is added to EACH SAMPLE
#                           PATH before quantiles are computed -- matching R's:
#                           mutate(.sim = .sim + lag_cumulative_observed_cases)
#
# TRANSFORMATIONS (added):
#   Optional transformation applied to training data before fitting and inverted
#   on the forecast TimeSeries BEFORE sample extraction, so all annual summing,
#   lag-cumulative addition, and quantile computation remain on the original scale.
#   Options: none (default), log1p, scaler (min-max), boxcox
#   CLI: --transformation log1p
#
# OUTPUT: one row per (origin x year_of_interest x model), giving an annual
#         total forecast with quantiles + point stats -- directly comparable
#         to the R output monthly_forecasts_all_months_just_year.
#
# CLI usage:
#   python Total_cases_forecasting_v2.py ARIMA 2016
#   python Total_cases_forecasting_v2.py ETS 2020 --transformation log1p
#   python Total_cases_forecasting_v2.py Prophet 2018 --transformation scaler
#   python Total_cases_forecasting_v2.py ARIMA 2016 --transformation boxcox
# ------------------------------------------------------------

import os
import argparse
import logging
import numpy as np
import pandas as pd
import pyreadr

import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import BoxCox, Scaler
from darts.utils.likelihood_models import QuantileRegression
from darts.models import (
    AutoARIMA,
    Prophet,
    DLinearModel,
    NLinearModel,
    TiDEModel,
    NHiTSModel,
    NBEATSModel,
    LinearRegressionModel,
    AutoETS,
)

# -----------------------------
# USER-EDITABLE SETTINGS
# -----------------------------
RDS_PATH      = "all_cases_20_Jan_2026.rds"
OUT_DIR       = "results"
DATE_COL      = "merged_date_min"

NUM_SAMPLES          = 1000  # sample paths -- matches R's times=1000
MIN_TRAIN_MONTHS     = 1     # R has no minimum; set to 1 to match (was 36 in v1)
INPUT_CHUNK_LENGTH   = 36   # fixed input window for all deep/regression models (weeks)

# Quantiles to save -- richer than R's 80/95 only, but 80+95 are included
QUANTILES = [
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99
]

# Years to run -- matches R's years_to_explore = 2015:2025
YEARS_TO_EXPLORE = list(range(2015, 2026))

# -----------------------------
# Logging + compute config
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

if torch.cuda.is_available():
    pl_trainer_kwargs = {"accelerator": "gpu", "devices": [2]}
    logging.info("GPU detected.")
else:
    pl_trainer_kwargs = {"accelerator": "cpu", "devices": 1}
    logging.info("No GPU detected. Using CPU.")

torch.set_num_threads(4)

# -----------------------------
# Model factory
# input_chunk_length is fixed at INPUT_CHUNK_LENGTH for all models
# -----------------------------
def get_model(model_name: str, forecast_length: int):
    if model_name == "TiDE":
        return TiDEModel(input_chunk_length=INPUT_CHUNK_LENGTH, output_chunk_length=forecast_length,
                         likelihood=QuantileRegression(quantiles=QUANTILES), pl_trainer_kwargs=pl_trainer_kwargs)
    elif model_name == "NBEATS":
        return NBEATSModel(input_chunk_length=INPUT_CHUNK_LENGTH, output_chunk_length=forecast_length,
                           likelihood=QuantileRegression(quantiles=QUANTILES), pl_trainer_kwargs=pl_trainer_kwargs)
    elif model_name == "NHiTS":
        return NHiTSModel(input_chunk_length=INPUT_CHUNK_LENGTH, output_chunk_length=forecast_length,
                          likelihood=QuantileRegression(quantiles=QUANTILES), pl_trainer_kwargs=pl_trainer_kwargs)
    elif model_name == "DLinear":
        return DLinearModel(input_chunk_length=INPUT_CHUNK_LENGTH, output_chunk_length=forecast_length,
                            likelihood=QuantileRegression(quantiles=QUANTILES), pl_trainer_kwargs=pl_trainer_kwargs)
    elif model_name == "NLinear":
        return NLinearModel(input_chunk_length=INPUT_CHUNK_LENGTH, output_chunk_length=forecast_length,
                            likelihood=QuantileRegression(quantiles=QUANTILES), pl_trainer_kwargs=pl_trainer_kwargs)
    elif model_name == "Linear Regress":
        return LinearRegressionModel(lags=INPUT_CHUNK_LENGTH, output_chunk_length=forecast_length,
                                     likelihood="quantile", quantiles=QUANTILES)
    elif model_name == "Prophet":
        return Prophet(yearly_seasonality=True)
    elif model_name == "ARIMA":
        return AutoARIMA(season_length=12)
    elif model_name == "ETS":
        return AutoETS(season_length=12)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# Maps each quantile to its PI column name.
# Symmetric pairs share a PI level; 0.50 is the median (no lower/upper).
QUANTILE_TO_COL = {
    0.01:  "pi_98_lower",
    0.025: "pi_95_lower",
    0.05:  "pi_90_lower",
    0.10:  "pi_80_lower",
    0.15:  "pi_70_lower",
    0.20:  "pi_60_lower",
    0.25:  "pi_50_lower",
    0.30:  "pi_40_lower",
    0.35:  "pi_30_lower",
    0.40:  "pi_20_lower",
    0.45:  "pi_10_lower",
    0.50:  "pi_50",
    0.55:  "pi_10_upper",
    0.60:  "pi_20_upper",
    0.65:  "pi_30_upper",
    0.70:  "pi_40_upper",
    0.75:  "pi_50_upper",
    0.80:  "pi_60_upper",
    0.85:  "pi_70_upper",
    0.90:  "pi_80_upper",
    0.95:  "pi_90_upper",
    0.975: "pi_95_upper",
    0.99:  "pi_98_upper",
}

def qcol(q: float) -> str:
    return QUANTILE_TO_COL[q]

# -----------------------------
# Transformation helpers
# Mirrors the apply/invert pattern from regional_univariate.py.
# Inversion is called on the full probabilistic TimeSeries BEFORE sample
# extraction, so all downstream math (annual sum, lag addition, quantiles)
# stays on the original case-count scale.
# -----------------------------
def apply_transformation(
    train_series: TimeSeries,
    transformation: str,
):
    """
    Apply transformation to training series before model fitting.

    Returns (transformed_series, transformer_object).
    transformer_object is passed to invert_transformation after forecasting.

    Options:
        none    -- no-op, returns series unchanged
        log1p   -- np.log1p applied element-wise (handles zeros; requires values > -1)
        scaler  -- Darts Scaler (min-max normalisation)
        boxcox  -- Darts BoxCox (requires strictly positive values)
    """
    if transformation == "none":
        return train_series, None

    if transformation == "log1p":
        vals = train_series.values(copy=False).reshape(-1)
        if np.any(vals <= -1):
            raise ValueError("log1p requires all values > -1")
        df = train_series.to_dataframe()
        df = df.apply(np.log1p)
        transformed = TimeSeries.from_dataframe(df)
        return transformed, "log1p"

    if transformation == "scaler":
        t = Scaler()
        return t.fit_transform(train_series), t

    if transformation == "boxcox":
        vals = train_series.values(copy=False).reshape(-1)
        if np.any(vals <= 0):
            raise ValueError("BoxCox requires strictly positive values")
        t = BoxCox()
        return t.fit_transform(train_series), t

    raise ValueError(
        f"Unknown transformation: '{transformation}'. "
        f"Choose from: none, log1p, scaler, boxcox"
    )


def invert_transformation(
    forecast_series: TimeSeries,
    transformer,
) -> TimeSeries:
    """
    Invert transformation on a probabilistic forecast TimeSeries.

    Uses all_values() for log1p so every sample path is back-transformed
    correctly, preserving the full distribution shape.
    """
    if transformer is None:
        return forecast_series

    if transformer == "log1p":
        all_vals = forecast_series.all_values(copy=False)  # (h, 1, NUM_SAMPLES)
        back = np.expm1(all_vals)
        return TimeSeries.from_times_and_values(
            times=forecast_series.time_index,
            values=back,
            columns=forecast_series.components,
        )

    return transformer.inverse_transform(forecast_series)

# -----------------------------
# Load RDS -> monthly series
# -----------------------------
def load_cases_monthly(rds_path: str, date_col: str) -> pd.Series:
    result = pyreadr.read_r(rds_path)
    df     = next(iter(result.values()))

    if date_col not in df.columns:
        raise ValueError(f"DATE_COL='{date_col}' not found. Available: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df["month"]  = df[date_col].dt.to_period("M").dt.to_timestamp()

    if "case" in df.columns and pd.to_numeric(df["case"], errors="coerce").notna().any():
        monthly = df.groupby("month")["case"].sum().astype(float)
        logging.info("Using 'case' column, summing per month.")
    else:
        monthly = df.groupby("month").size().astype(float)
        logging.info("Counting rows per month.")

    monthly  = monthly.sort_index()
    full_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly  = monthly.reindex(full_idx, fill_value=0.0)
    monthly.index = pd.DatetimeIndex(monthly.index, freq="MS")
    monthly.name  = "cases"
    return monthly

# -----------------------------
# Enforce probabilistic output
# -----------------------------
def predict_probabilistic(model, h: int) -> TimeSeries:
    fc     = model.predict(h, num_samples=int(NUM_SAMPLES))
    n_samp = getattr(fc, "n_samples", 1)
    if n_samp <= 1:
        raise RuntimeError(
            f"Model returned deterministic forecast (n_samples={n_samp}). "
            f"Requested NUM_SAMPLES={NUM_SAMPLES}. Check model/Darts version."
        )
    return fc

# -----------------------------
# Build the origins dataframe -- mirrors R's every_month_to_map_through_df
#
# For each year Y in YEARS_TO_EXPLORE:
#   origins run Sep (Y-1) through Dec (Y)  [16 months, matching R]
#   months_to_forecast = months from origin to Jan (Y+1)  [i.e. through Dec Y]
#   lag_cumulative_observed_cases = observed cases Jan-Y up to (but not
#     including) month_forecasting_from, 0 if origin is before Jan Y
#
# This replicates the R logic exactly:
#   flag_last_12_rows, cumulative_observed_cases, lag_cumulative_observed_cases
# -----------------------------
def build_origins_df(monthly_cases: pd.Series) -> pd.DataFrame:
    rows = []
    for Y in YEARS_TO_EXPLORE:
        # origins: Sep (Y-1) -> Dec (Y), step 1 month -- 16 origins
        origin_start = pd.Timestamp(f"{Y-1}-09-01")
        origin_end   = pd.Timestamp(f"{Y}-12-01")
        origins      = pd.date_range(origin_start, origin_end, freq="MS")

        # Jan Y timestamp -- used to identify observed months within year Y
        jan_y = pd.Timestamp(f"{Y}-01-01")

        for origin_ts in origins:
            # months_to_forecast: from origin to Dec Y (inclusive of Dec Y)
            # = number of months between origin and Jan (Y+1)
            jan_next = pd.Timestamp(f"{Y+1}-01-01")
            months_to_forecast = (
                (jan_next.year - origin_ts.year) * 12
                + (jan_next.month - origin_ts.month)
            )

            if months_to_forecast <= 0:
                continue

            # lag_cumulative_observed_cases:
            # sum of observed cases in year Y from Jan up to (not including) origin
            # mirrors R: lag(cumsum(cases * flag_last_12_rows))
            if origin_ts <= jan_y:
                # origin is before or at Jan Y -- no observed year-Y data yet
                lag_cumulative = 0
            else:
                # observed months are Jan Y up to (but not including) origin_ts
                obs_months = pd.date_range(jan_y, origin_ts - pd.DateOffset(months=1), freq="MS")
                lag_cumulative = float(
                    sum(monthly_cases.get(m, 0.0) for m in obs_months)
                )

            rows.append({
                "month_forecasting_from":             origin_ts,
                "year_of_interest":                   Y,
                "months_to_forecast":                 months_to_forecast,
                "lag_cumulative_observed_cases":      lag_cumulative,
            })

    df = pd.DataFrame(rows)
    logging.info(f"Origins dataframe: {len(df)} rows across {len(YEARS_TO_EXPLORE)} years")
    return df

# -----------------------------
# Core forecast function -- mirrors R's monthly_forecasts_function
#
# For each origin row:
#   1. Train on all data BEFORE month_forecasting_from  (R: filter(year_month_date < month_forecasting_from))
#   2. Apply transformation to training series
#   3. Fit model on transformed series
#   4. Forecast months_to_forecast steps ahead (in transformed space)
#   5. Invert transformation on the full probabilistic TimeSeries
#      (BEFORE sample extraction -- keeps all downstream math on original scale)
#   6. Extract the NUM_SAMPLES sample paths
#   7. For each sample path, sum only the months falling in year_of_interest (Jan-Dec Y)
#   8. Add lag_cumulative_observed_cases to EACH sample path  <-- key R logic
#   9. Compute quantiles + summary stats across the adjusted sample paths
# -----------------------------
def run_annual_forecast(
    monthly_cases:  pd.Series,
    model_name:     str,
    origins_df:     pd.DataFrame,
    transformation: str = "none",
) -> pd.DataFrame:

    all_rows = []

    for _, row in origins_df.iterrows():
        origin_ts          = row["month_forecasting_from"]
        Y                  = int(row["year_of_interest"])
        h                  = int(row["months_to_forecast"])
        lag_cumulative     = float(row["lag_cumulative_observed_cases"])

        # ------------------------------------------------------------------
        # 1. Training slice: strictly BEFORE origin_ts  (mirrors R's <)
        # ------------------------------------------------------------------
        train_slice = monthly_cases.loc[monthly_cases.index < origin_ts]

        # Deep/regression models need at least INPUT_CHUNK_LENGTH months of history.
        # Stats models (ARIMA, ETS, Prophet) only need a small minimum.
        stats_models = {"ARIMA", "ETS", "Prophet"}
        min_train = 2 if model_name in stats_models else max(2, INPUT_CHUNK_LENGTH + 1)

        if len(train_slice) < min_train:
            logging.warning(f"Skipping origin {origin_ts.date()} Y={Y}: only {len(train_slice)} training months (need {min_train})")
            continue

        train_ts = TimeSeries.from_series(train_slice)

        # ------------------------------------------------------------------
        # 2. Apply transformation, fit, forecast, then invert transformation
        #    Inversion happens on the full TimeSeries BEFORE sample extraction
        #    so all downstream math (annual sum, lag addition, quantiles)
        #    operates on the original case-count scale.
        # ------------------------------------------------------------------
        train_t, transformer = apply_transformation(train_ts, transformation)

        model = get_model(model_name=model_name, forecast_length=h)
        model.fit(train_t)
        fc = predict_probabilistic(model, h)  # shape: (h, 1, NUM_SAMPLES) in transformed space

        # Invert BEFORE extracting the sample matrix
        fc = invert_transformation(fc, transformer)

        # ------------------------------------------------------------------
        # 3. Build a DataFrame of sample paths: rows=months, cols=samples
        #    Attach the actual forecast timestamps
        # ------------------------------------------------------------------
        # fc.all_values() shape: (h, 1, NUM_SAMPLES)
        samples_matrix = fc.all_values()[:, 0, :]  # shape: (h, NUM_SAMPLES)

        # Forecast timestamps (start of each forecast month)
        forecast_timestamps = pd.date_range(origin_ts, periods=h, freq="MS")

        fc_df = pd.DataFrame(
            samples_matrix,
            index=forecast_timestamps,
            columns=[f"s{i}" for i in range(NUM_SAMPLES)]
        )

        # ------------------------------------------------------------------
        # 4. Filter to year_of_interest only (Jan-Dec Y)
        #    Mirrors R: filter(year == year_of_interest_only)
        # ------------------------------------------------------------------
        fc_year = fc_df[fc_df.index.year == Y]

        if fc_year.empty:
            logging.warning(f"No forecast months fall in year {Y} for origin {origin_ts.date()}")
            continue

        # ------------------------------------------------------------------
        # 5. Sum across the year for each sample path -> shape: (NUM_SAMPLES,)
        #    Then add lag_cumulative_observed_cases to EACH sample path
        #    Mirrors R: summarise(.sim = sum(.sim)) then
        #               mutate(.sim = .sim + lag_cumulative_observed_cases)
        # ------------------------------------------------------------------
        annual_samples = fc_year.sum(axis=0).values  # (NUM_SAMPLES,)
        annual_samples_adjusted = annual_samples + lag_cumulative  # key step

        # ------------------------------------------------------------------
        # 6. Compute quantiles from the adjusted sample paths
        #    (intervals now correctly reflect uncertainty around annual total)
        # ------------------------------------------------------------------
        quantile_vals = {qcol(q): float(np.quantile(annual_samples_adjusted, q)) for q in QUANTILES}

        mean_val   = float(np.mean(annual_samples_adjusted))
        median_val = float(np.median(annual_samples_adjusted))

        result_row = {
            "mean":                               mean_val,
            "median":                             median_val,
            "month_forecasting_from":             origin_ts,
            "year_of_interest_plus_prev_months":  Y,
            "model":                              model_name,
            "transformation":                     transformation,
            **quantile_vals,
            "months_to_forecast":                 h,
            "number_of_months_to_forecast":       f"{h} months",
        }

        all_rows.append(result_row)
        logging.info(
            f"Origin {origin_ts.date()} -> Y={Y}: h={h}, "
            f"transform={transformation}, "
            f"obs_carried={lag_cumulative:,.0f}, "
            f"mean={mean_val:,.0f}, "
            f"train_n={len(train_slice)}"
        )

    df_out = pd.DataFrame(all_rows)
    if df_out.empty:
        raise RuntimeError("No forecasts generated.")

    return df_out

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Annual total forecast aligned to R fpp3 methodology. "
            "Origins: Sep(Y-1) to Dec(Y). "
            "Observed months Jan-to-origin are added to each sample path before quantiles."
        )
    )
    parser.add_argument(
        "model",
        type=str,
        help='Model: TiDE, NBEATS, NHiTS, DLinear, NLinear, Prophet, ARIMA, ETS, "Linear Regress"',
    )
    parser.add_argument(
        "season_end_year",
        type=int,
        nargs="?",
        default=None,
        help=(
            "Optional: run a single year only (e.g. 2016). "
            "If omitted, runs all years in YEARS_TO_EXPLORE."
        ),
    )
    parser.add_argument(
        "--transformation",
        type=str,
        default="none",
        choices=["none", "log1p", "scaler", "boxcox"],
        help=(
            "Transformation applied to training data before fitting, "
            "inverted on the forecast before sample extraction. "
            "Options: none (default), log1p, scaler (min-max), boxcox. "
            "Example: --transformation log1p"
        ),
    )
    args = parser.parse_args()

    # Optionally restrict to a single year
    if args.season_end_year is not None:
        YEARS_TO_EXPLORE[:] = [args.season_end_year]

    # Load data
    monthly_cases = load_cases_monthly(RDS_PATH, DATE_COL)
    logging.info(f"Series: {monthly_cases.index.min().date()} to {monthly_cases.index.max().date()} ({len(monthly_cases)} months)")
    logging.info(f"INPUT_CHUNK_LENGTH={INPUT_CHUNK_LENGTH}, NUM_SAMPLES={NUM_SAMPLES}, transformation={args.transformation}")

    # Build origins (mirrors every_month_to_map_through_df in R)
    origins_df = build_origins_df(monthly_cases)

    # Run forecasts
    df_out = run_annual_forecast(
        monthly_cases=monthly_cases,
        model_name=args.model,
        origins_df=origins_df,
        transformation=args.transformation,
    )

    # Save -- transformation name included in filename so runs don't overwrite each other
    out_dir = os.path.join(OUT_DIR, args.model.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)

    year_tag = str(args.season_end_year) if args.season_end_year else "all_years"
    out_path = os.path.join(
        out_dir,
        f"{args.model}_transform_{args.transformation}_annual_total_forecast_{year_tag}.csv"
    )
    df_out.to_csv(out_path, index=False)
    logging.info(f"Saved: {out_path}  ({len(df_out)} rows)")
    logging.info("Done.")