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
# KEY CHANGE FROM PREVIOUS v2:
#   6. Uncertainty via RESIDUAL BOOTSTRAP, not Darts' internal marginal sampling.
#      Darts draws each forecast month independently from its marginal distribution.
#      When 12 independent monthly draws are summed, errors cancel and the annual
#      total variance is severely underestimated -- producing intervals that are
#      too tight, exactly as seen when comparing to Jamie's R output.
#
#      Jamie's R method (generate(), times=1000) simulates sequential paths where
#      each step builds on the previous simulated value -- errors accumulate and
#      compound, giving wider, correctly-calibrated annual total intervals.
#
#      We replicate this by:
#        a) Getting the model's point forecast (num_samples=1)
#        b) Computing in-sample 1-step-ahead residuals from the training data
#        c) For each of NUM_SAMPLES bootstrap paths:
#             - draw h residuals with replacement from in-sample residuals
#             - simulated_path = point_forecast + drawn_residuals
#        d) Filter each path to year Y, sum, add lag_cumulative_observed_cases
#        e) Take quantiles of the NUM_SAMPLES annual totals
#
#      This matches Jamie's steps 1-6 exactly and applies to ALL model types.
#
# TRANSFORMATIONS (added):
#   Optional transformation applied to training data before fitting.
#   The point forecast and historical forecasts (used for residuals) are both
#   inverted back to the original scale BEFORE residuals are computed and
#   before the bootstrap runs -- so all annual summing, lag-cumulative
#   addition, and quantile computation remain on the original case-count scale.
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
RDS_PATH             = "all_cases_20_Jan_2026.rds"
OUT_DIR              = "results_sam"
DATE_COL             = "merged_date_min"

NUM_SAMPLES          = 1000  # bootstrap paths -- matches R's times=1000
MIN_TRAIN_MONTHS     = 1     # R has no minimum; set to 1 to match
INPUT_CHUNK_LENGTH   = 36    # input window for deep/regression models (months)

# Minimum number of in-sample residuals required to bootstrap from.
MIN_RESIDUALS        = 12

# Quantiles to save
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
# Bootstrap only needs a point forecast (num_samples=1) so deep learning
# models no longer require QuantileRegression -- we use default MSE loss.
# -----------------------------
def get_model(model_name: str, forecast_length: int):
    if model_name == "TiDE":
        return TiDEModel(
            input_chunk_length=INPUT_CHUNK_LENGTH,
            output_chunk_length=forecast_length,
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
    elif model_name == "NBEATS":
        return NBEATSModel(
            input_chunk_length=INPUT_CHUNK_LENGTH,
            output_chunk_length=forecast_length,
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
    elif model_name == "NHiTS":
        return NHiTSModel(
            input_chunk_length=INPUT_CHUNK_LENGTH,
            output_chunk_length=forecast_length,
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
    elif model_name == "DLinear":
        return DLinearModel(
            input_chunk_length=INPUT_CHUNK_LENGTH,
            output_chunk_length=forecast_length,
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
    elif model_name == "NLinear":
        return NLinearModel(
            input_chunk_length=INPUT_CHUNK_LENGTH,
            output_chunk_length=forecast_length,
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
    elif model_name == "Linear Regress":
        return LinearRegressionModel(
            lags=INPUT_CHUNK_LENGTH,
            output_chunk_length=forecast_length,
        )
    elif model_name == "Prophet":
        return Prophet(yearly_seasonality=True)
    elif model_name == "ARIMA":
        return AutoARIMA(season_length=12)
    elif model_name == "ETS":
        return AutoETS(season_length=12)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# Maps each quantile to its PI column name.
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
#
# Transformation is applied to training data before fitting.
# Both the point forecast AND the historical forecasts used for residual
# computation are inverted back to the original scale before any arithmetic,
# so residuals, bootstrap paths, annual sums, and quantiles all stay on the
# original case-count scale.
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
    Invert transformation on a TimeSeries (point forecast or historical forecasts).

    Uses all_values() for log1p so every sample is back-transformed correctly.
    For scaler/boxcox, delegates to the fitted transformer's inverse_transform.
    """
    if transformer is None:
        return forecast_series

    if transformer == "log1p":
        all_vals = forecast_series.all_values(copy=False)  # (h, 1, n_samples)
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
# Compute in-sample residuals
#
# Uses Darts historical_forecasts() with retrain=False to get 1-step-ahead
# point predictions across the training series after the model has been fitted.
#
# IMPORTANT: historical_forecasts are run on the TRANSFORMED series (train_t),
# then inverted back to the original scale before computing residuals.
# Actual values are taken from the ORIGINAL train_ts (not transformed).
# This means residuals are always on the original case-count scale, which is
# what the bootstrap needs.
#
# residual = actual_original - predicted_original at each in-sample step.
#
# For stats models (ARIMA, ETS, Prophet): burn-in of 12 months.
# For deep learning models: burn-in of INPUT_CHUNK_LENGTH months so the model
# always has a full input window available.
#
# Returns a 1D numpy array of residuals, or None if insufficient history.
# -----------------------------
def compute_residuals(
    model,
    train_ts:    TimeSeries,
    train_t:     TimeSeries,
    transformer,
    model_name:  str,
) -> np.ndarray | None:

    stats_models = {"ARIMA", "ETS", "Prophet"}
    burn_in = 12 if model_name in stats_models else INPUT_CHUNK_LENGTH

    if len(train_ts) <= burn_in + MIN_RESIDUALS:
        return None

    try:
        # Historical forecasts on the transformed series
        hf_t = model.historical_forecasts(
            series=train_t,
            start=burn_in,
            start_format="position",
            forecast_horizon=1,
            stride=1,
            retrain=False,
            overlap_end=False,
            last_points_only=True,
            num_samples=1,
            verbose=False,
        )

        # Invert to original scale before computing residuals
        hf = invert_transformation(hf_t, transformer)

        # Actual values on original scale, sliced to match the forecast index
        actual_ts = train_ts.slice_intersect(hf)
        actual    = actual_ts.values().flatten()
        predicted = hf.values().flatten()
        residuals = actual - predicted

        if len(residuals) < MIN_RESIDUALS:
            return None

        return residuals

    except Exception as e:
        logging.warning(f"historical_forecasts failed: {e}. Trying fallback.")
        # Fallback: re-predict the full training series for approximate residuals.
        try:
            fitted_t  = model.predict(len(train_ts), series=train_t, num_samples=1)
            fitted    = invert_transformation(fitted_t, transformer)
            predicted = fitted.values().flatten()
            actual    = train_ts.values().flatten()
            residuals = actual - predicted
            if len(residuals) < MIN_RESIDUALS:
                return None
            return residuals
        except Exception as e2:
            logging.warning(f"Fallback residual computation also failed: {e2}")
            return None


# -----------------------------
# Residual bootstrap
#
# Replicates Jamie's R approach (generate(times=1000)).
# point_forecast and residuals are both on the ORIGINAL scale, so the
# bootstrap paths are also on the original scale -- no inversion needed here.
#
# Returns array of shape (n_samples,) -- one annual total per bootstrap path.
# -----------------------------
def residual_bootstrap_annual_samples(
    point_forecast:      np.ndarray,
    forecast_timestamps: pd.DatetimeIndex,
    residuals:           np.ndarray,
    year_of_interest:    int,
    lag_cumulative:      float,
    n_samples:           int,
    rng:                 np.random.Generator,
) -> np.ndarray:

    h         = len(point_forecast)
    year_mask = np.array([ts.year == year_of_interest for ts in forecast_timestamps])

    # Draw all residuals at once for efficiency: shape (n_samples, h)
    drawn_residuals = rng.choice(residuals, size=(n_samples, h), replace=True)

    # Each row is one simulated path
    sim_paths = point_forecast[np.newaxis, :] + drawn_residuals  # (n_samples, h)

    # Sum months in year of interest, add observed cases already elapsed
    annual_samples = sim_paths[:, year_mask].sum(axis=1) + lag_cumulative

    return annual_samples


# -----------------------------
# Build the origins dataframe -- mirrors R's every_month_to_map_through_df
# -----------------------------
def build_origins_df(monthly_cases: pd.Series) -> pd.DataFrame:
    rows = []
    for Y in YEARS_TO_EXPLORE:
        origin_start = pd.Timestamp(f"{Y-1}-09-01")
        origin_end   = pd.Timestamp(f"{Y}-12-01")
        origins      = pd.date_range(origin_start, origin_end, freq="MS")
        jan_y        = pd.Timestamp(f"{Y}-01-01")

        for origin_ts in origins:
            jan_next = pd.Timestamp(f"{Y+1}-01-01")
            months_to_forecast = (
                (jan_next.year  - origin_ts.year)  * 12
                + (jan_next.month - origin_ts.month)
            )

            if months_to_forecast <= 0:
                continue

            if origin_ts <= jan_y:
                lag_cumulative = 0
            else:
                obs_months     = pd.date_range(
                    jan_y, origin_ts - pd.DateOffset(months=1), freq="MS"
                )
                lag_cumulative = float(
                    sum(monthly_cases.get(m, 0.0) for m in obs_months)
                )

            rows.append({
                "month_forecasting_from":        origin_ts,
                "year_of_interest":              Y,
                "months_to_forecast":            months_to_forecast,
                "lag_cumulative_observed_cases": lag_cumulative,
            })

    df = pd.DataFrame(rows)
    logging.info(f"Origins dataframe: {len(df)} rows across {len(YEARS_TO_EXPLORE)} years")
    return df


# -----------------------------
# Core forecast function
# -----------------------------
def run_annual_forecast(
    monthly_cases:  pd.Series,
    model_name:     str,
    origins_df:     pd.DataFrame,
    transformation: str = "none",
) -> pd.DataFrame:

    all_rows     = []
    rng          = np.random.default_rng(seed=42)
    stats_models = {"ARIMA", "ETS", "Prophet"}

    for _, row in origins_df.iterrows():
        origin_ts      = row["month_forecasting_from"]
        Y              = int(row["year_of_interest"])
        h              = int(row["months_to_forecast"])
        lag_cumulative = float(row["lag_cumulative_observed_cases"])

        # ------------------------------------------------------------------
        # 1. Training slice: strictly BEFORE origin_ts  (mirrors R's <)
        # ------------------------------------------------------------------
        train_slice = monthly_cases.loc[monthly_cases.index < origin_ts]

        min_train = 2 if model_name in stats_models else max(2, INPUT_CHUNK_LENGTH + 1)

        if len(train_slice) < min_train:
            logging.warning(
                f"Skipping origin {origin_ts.date()} Y={Y}: "
                f"only {len(train_slice)} training months (need {min_train})"
            )
            continue

        train_ts = TimeSeries.from_series(train_slice)

        # ------------------------------------------------------------------
        # 2. Apply transformation, fit model
        # ------------------------------------------------------------------
        train_t, transformer = apply_transformation(train_ts, transformation)

        model = get_model(model_name=model_name, forecast_length=h)
        model.fit(train_t)

        # ------------------------------------------------------------------
        # 3. Point forecast in transformed space, then invert to original scale
        # ------------------------------------------------------------------
        fc_t       = model.predict(h, num_samples=1)
        fc_point   = invert_transformation(fc_t, transformer)
        point_vals = fc_point.all_values()[:, 0, 0]  # shape (h,), original scale

        forecast_timestamps = pd.date_range(origin_ts, periods=h, freq="MS")

        # ------------------------------------------------------------------
        # 4. In-sample residuals on the original scale
        #    historical_forecasts run on transformed series, then inverted;
        #    actuals taken from original train_ts -- residuals on original scale
        # ------------------------------------------------------------------
        residuals = compute_residuals(
            model       = model,
            train_ts    = train_ts,
            train_t     = train_t,
            transformer = transformer,
            model_name  = model_name,
        )

        if residuals is None:
            logging.warning(
                f"Skipping origin {origin_ts.date()} Y={Y}: "
                f"could not compute residuals (insufficient history)"
            )
            continue

        # ------------------------------------------------------------------
        # 5. Residual bootstrap -- replicate Jamie's generate(times=1000)
        #    point_forecast and residuals are both on original scale
        # ------------------------------------------------------------------
        annual_samples = residual_bootstrap_annual_samples(
            point_forecast      = point_vals,
            forecast_timestamps = forecast_timestamps,
            residuals           = residuals,
            year_of_interest    = Y,
            lag_cumulative      = lag_cumulative,
            n_samples           = NUM_SAMPLES,
            rng                 = rng,
        )

        # ------------------------------------------------------------------
        # 6. Quantiles of the NUM_SAMPLES annual totals
        # ------------------------------------------------------------------
        quantile_vals = {
            qcol(q): float(np.quantile(annual_samples, q)) for q in QUANTILES
        }

        mean_val   = float(np.mean(annual_samples))
        median_val = float(np.median(annual_samples))

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
            f"n_residuals={len(residuals)}, "
            f"obs_carried={lag_cumulative:,.0f}, "
            f"mean={mean_val:,.0f}, "
            f"pi_95=({quantile_vals[qcol(0.025)]:,.0f}"
            f"-{quantile_vals[qcol(0.975)]:,.0f}), "
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
            "Residual bootstrap replicates R's generate(times=1000) for all models."
        )
    )
    parser.add_argument(
        "model",
        type=str,
        help=(
            'Model: TiDE, NBEATS, NHiTS, DLinear, NLinear, '
            'Prophet, ARIMA, ETS, "Linear Regress"'
        ),
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
            "Transformation applied to training data before fitting. "
            "Point forecast and historical forecasts are inverted to the original "
            "scale before residuals are computed, so the bootstrap and all "
            "downstream quantiles remain on the original case-count scale. "
            "Options: none (default), log1p, scaler (min-max), boxcox. "
            "Example: --transformation log1p"
        ),
    )
    args = parser.parse_args()

    if args.season_end_year is not None:
        YEARS_TO_EXPLORE[:] = [args.season_end_year]

    monthly_cases = load_cases_monthly(RDS_PATH, DATE_COL)
    logging.info(
        f"Series: {monthly_cases.index.min().date()} to "
        f"{monthly_cases.index.max().date()} ({len(monthly_cases)} months)"
    )
    logging.info(
        f"INPUT_CHUNK_LENGTH={INPUT_CHUNK_LENGTH}, "
        f"NUM_SAMPLES={NUM_SAMPLES}, "
        f"transformation={args.transformation}"
    )

    origins_df = build_origins_df(monthly_cases)

    df_out = run_annual_forecast(
        monthly_cases  = monthly_cases,
        model_name     = args.model,
        origins_df     = origins_df,
        transformation = args.transformation,
    )

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