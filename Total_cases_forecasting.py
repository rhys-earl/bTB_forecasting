# Total_cases_forecasting_v2_darts.py
# ------------------------------------------------------------
# Rolling seasonal forecast aligned to R (fpp3) methodology.
# Uncertainty from Darts' internal probabilistic sampling.
#
# CLI usage:
#   python Total_cases_forecasting_v2_darts.py ARIMA
#   python Total_cases_forecasting_v2_darts.py ARIMA 2016
#   python Total_cases_forecasting_v2_darts.py ARIMA --transformation log1p scaler none
#   python Total_cases_forecasting_v2_darts.py TiDE --gpu 1 --transformation log1p none
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
    LightGBMModel,
    NLinearModel,
    AutoTheta,
    AutoTBATS,
    AutoCES,
    TiDEModel,
    NHiTSModel,
    NBEATSModel,
    LinearRegressionModel,
    AutoETS,
    TransformerModel,
    Chronos2Model

)

# -----------------------------
# USER-EDITABLE SETTINGS
# -----------------------------
RDS_PATH             = "all_cases_20_Jan_2026.rds"
OUT_DIR              = "results"
DATE_COL             = "merged_date_min"

NUM_SAMPLES          = 1000
INPUT_CHUNK_LENGTH   = 36

QUANTILES = [
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99
]

YEARS_TO_EXPLORE = list(range(2015, 2026))

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
torch.set_num_threads(4)

# pl_trainer_kwargs is set after args are parsed (so --gpu can be applied)
pl_trainer_kwargs = {}

# -----------------------------
# Model factory
# -----------------------------
def get_model(model_name: str, forecast_length: int):
    ql = QuantileRegression(quantiles=QUANTILES)
    if model_name == "TiDE":
        return TiDEModel(input_chunk_length=INPUT_CHUNK_LENGTH, output_chunk_length=forecast_length,
                         likelihood=ql, pl_trainer_kwargs=pl_trainer_kwargs)
    elif model_name == "NBEATS":
        return NBEATSModel(input_chunk_length=INPUT_CHUNK_LENGTH, output_chunk_length=forecast_length,
                           likelihood=ql, pl_trainer_kwargs=pl_trainer_kwargs)
    elif model_name == "NHiTS":
        return NHiTSModel(input_chunk_length=INPUT_CHUNK_LENGTH, output_chunk_length=forecast_length,
                          likelihood=ql, pl_trainer_kwargs=pl_trainer_kwargs)
    elif model_name == "DLinear":
        return DLinearModel(input_chunk_length=INPUT_CHUNK_LENGTH, output_chunk_length=forecast_length,
                            likelihood=ql, pl_trainer_kwargs=pl_trainer_kwargs)
    elif model_name == "NLinear":
        return NLinearModel(input_chunk_length=INPUT_CHUNK_LENGTH, output_chunk_length=forecast_length,
                            likelihood=ql, pl_trainer_kwargs=pl_trainer_kwargs)
    elif model_name == "Chronos":
        return Chronos2Model(
            input_chunk_length=INPUT_CHUNK_LENGTH, output_chunk_length=forecast_length,
            likelihood=ql)
    elif model_name == "Transformer":
        return TransformerModel(input_chunk_length=INPUT_CHUNK_LENGTH, output_chunk_length=forecast_length,
                            likelihood=ql, pl_trainer_kwargs=pl_trainer_kwargs)
    elif model_name == "Linear Regress":
        return LinearRegressionModel(lags=INPUT_CHUNK_LENGTH, output_chunk_length=forecast_length,
                                     likelihood="quantile", quantiles=QUANTILES)
    elif model_name == "LGBM":
        return LightGBMModel(lags=INPUT_CHUNK_LENGTH, output_chunk_length=forecast_length,
                                     likelihood="quantile", quantiles=QUANTILES)
    elif model_name == "Theta":
        return AutoTheta(season_length=12)
    elif model_name == "CES":
        return AutoCES(season_length=12)
    elif model_name == "TBATS":
        return AutoTBATS(season_length=12)
    elif model_name == "Prophet":
        return Prophet(yearly_seasonality=True)
    elif model_name == "ARIMA":
        return AutoARIMA(season_length=12)
    elif model_name == "ETS":
        return AutoETS(season_length=12)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


QUANTILE_TO_COL = {
    0.01:  "pi_98_lower", 0.025: "pi_95_lower", 0.05:  "pi_90_lower",
    0.10:  "pi_80_lower", 0.15:  "pi_70_lower", 0.20:  "pi_60_lower",
    0.25:  "pi_50_lower", 0.30:  "pi_40_lower", 0.35:  "pi_30_lower",
    0.40:  "pi_20_lower", 0.45:  "pi_10_lower", 0.50:  "pi_50",
    0.55:  "pi_10_upper", 0.60:  "pi_20_upper", 0.65:  "pi_30_upper",
    0.70:  "pi_40_upper", 0.75:  "pi_50_upper", 0.80:  "pi_60_upper",
    0.85:  "pi_70_upper", 0.90:  "pi_80_upper", 0.95:  "pi_90_upper",
    0.975: "pi_95_upper", 0.99:  "pi_98_upper",
}

def qcol(q: float) -> str:
    return QUANTILE_TO_COL[q]


# -----------------------------
# Transformation helpers
# -----------------------------
def apply_transformation(train_series: TimeSeries, transformation: str):
    """
    Transform training series before fitting.
    Returns (transformed_series, transformer).
    transformer is None for 'none', a string 'log1p', or a fitted Darts transformer.
    """
    if transformation == "none":
        return train_series, None

    if transformation == "log1p":
        vals = train_series.values(copy=False).reshape(-1)
        if np.any(vals <= -1):
            raise ValueError("log1p requires all values > -1")
        df = train_series.to_dataframe().apply(np.log1p)
        return TimeSeries.from_dataframe(df), "log1p"

    if transformation == "scaler":
        t = Scaler()
        return t.fit_transform(train_series), t

    if transformation == "boxcox":
        vals = train_series.values(copy=False).reshape(-1)
        if np.any(vals <= 0):
            raise ValueError("BoxCox requires strictly positive values")
        t = BoxCox()
        return t.fit_transform(train_series), t

    raise ValueError(f"Unknown transformation: '{transformation}'. Choose from: none, log1p, scaler, boxcox")


def invert_transformation(forecast_series: TimeSeries, transformer) -> TimeSeries:
    """
    Invert transformation on a probabilistic TimeSeries.
    Inversion happens BEFORE sample extraction so all downstream math
    (annual sum, lag addition, quantiles) stays on the original scale.
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
# Load RDS
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

    monthly       = monthly.sort_index()
    full_idx      = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly       = monthly.reindex(full_idx, fill_value=0.0)
    monthly.index = pd.DatetimeIndex(monthly.index, freq="MS")
    monthly.name  = "cases"
    return monthly


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
# Build origins dataframe
# -----------------------------
def build_origins_df(monthly_cases: pd.Series) -> pd.DataFrame:
    rows = []
    for Y in YEARS_TO_EXPLORE:
        origin_start = pd.Timestamp(f"{Y-1}-09-01")
        origin_end   = pd.Timestamp(f"{Y}-12-01")
        origins      = pd.date_range(origin_start, origin_end, freq="MS")
        jan_y        = pd.Timestamp(f"{Y}-01-01")

        for origin_ts in origins:
            jan_next           = pd.Timestamp(f"{Y+1}-01-01")
            months_to_forecast = (
                (jan_next.year - origin_ts.year) * 12
                + (jan_next.month - origin_ts.month)
            )
            if months_to_forecast <= 0:
                continue

            if origin_ts <= jan_y:
                lag_cumulative = 0
            else:
                obs_months     = pd.date_range(jan_y, origin_ts - pd.DateOffset(months=1), freq="MS")
                lag_cumulative = float(sum(monthly_cases.get(m, 0.0) for m in obs_months))

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
# Core forecast function (single transformation)
# -----------------------------
def run_annual_forecast(
    monthly_cases:  pd.Series,
    model_name:     str,
    origins_df:     pd.DataFrame,
    transformation: str = "none",
) -> pd.DataFrame:

    all_rows     = []
    stats_models = {"ARIMA", "ETS", "Prophet"}

    for _, row in origins_df.iterrows():
        origin_ts      = row["month_forecasting_from"]
        Y              = int(row["year_of_interest"])
        h              = int(row["months_to_forecast"])
        lag_cumulative = float(row["lag_cumulative_observed_cases"])

        train_slice = monthly_cases.loc[monthly_cases.index < origin_ts]
        min_train   = 2 if model_name in stats_models else max(2, INPUT_CHUNK_LENGTH + 1)

        if len(train_slice) < min_train:
            logging.warning(
                f"Skipping origin {origin_ts.date()} Y={Y}: "
                f"only {len(train_slice)} training months (need {min_train})"
            )
            continue

        train_ts = TimeSeries.from_series(train_slice)

        # Apply transformation, fit, forecast, invert
        train_t, transformer = apply_transformation(train_ts, transformation)
        model                = get_model(model_name=model_name, forecast_length=h)
        model.fit(train_t)
        fc = predict_probabilistic(model, h)

        # Invert BEFORE extracting sample matrix -- keeps all math on original scale
        fc = invert_transformation(fc, transformer)

        samples_matrix      = fc.all_values()[:, 0, :]  # (h, NUM_SAMPLES)
        forecast_timestamps = pd.date_range(origin_ts, periods=h, freq="MS")

        fc_df   = pd.DataFrame(samples_matrix, index=forecast_timestamps,
                               columns=[f"s{i}" for i in range(NUM_SAMPLES)])
        fc_year = fc_df[fc_df.index.year == Y]

        if fc_year.empty:
            logging.warning(f"No forecast months fall in year {Y} for origin {origin_ts.date()}")
            continue

        annual_samples          = fc_year.sum(axis=0).values
        annual_samples_adjusted = annual_samples + lag_cumulative

        quantile_vals = {qcol(q): float(np.quantile(annual_samples_adjusted, q)) for q in QUANTILES}
        mean_val      = float(np.mean(annual_samples_adjusted))
        median_val    = float(np.median(annual_samples_adjusted))

        all_rows.append({
            "mean":                              mean_val,
            "median":                            median_val,
            "month_forecasting_from":            origin_ts,
            "year_of_interest_plus_prev_months": Y,
            "model":                             model_name,
            "transformation":                    transformation,
            **quantile_vals,
            "months_to_forecast":                h,
            "number_of_months_to_forecast":      f"{h} months",
        })
        logging.info(
            f"Origin {origin_ts.date()} -> Y={Y}: h={h}, transform={transformation}, "
            f"obs_carried={lag_cumulative:,.0f}, mean={mean_val:,.0f}, train_n={len(train_slice)}"
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
        description="Annual total forecast (Darts probabilistic sampling). Origins: Sep(Y-1) to Dec(Y)."
    )
    parser.add_argument(
        "model", type=str,
        help='Model: TiDE, NBEATS, NHiTS, DLinear, NLinear, Prophet, ARIMA, ETS, "Linear Regress"',
    )
    parser.add_argument(
        "season_end_year", type=int, nargs="?", default=None,
        help="Optional: single year only (e.g. 2016). Omit to run all years.",
    )
    parser.add_argument(
        "--transformation", type=str, nargs="+",
        default=["none"],
        choices=["none", "log1p", "scaler", "boxcox"],
        help=(
            "One or more transformations to run sequentially. "
            "Each saves to its own CSV. "
            "Example: --transformation log1p scaler none"
        ),
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU device index to use (e.g. --gpu 0, --gpu 2). Defaults to CPU.",
    )
    args = parser.parse_args()

    # Configure compute device
    if args.gpu is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(f"--gpu {args.gpu} specified but no CUDA GPUs are available.")
        pl_trainer_kwargs.update({"accelerator": "gpu", "devices": [args.gpu]})
        logging.info(f"Using GPU {args.gpu}")
    else:
        pl_trainer_kwargs.update({"accelerator": "cpu", "devices": 1})
        logging.info("Using CPU")

    if args.season_end_year is not None:
        YEARS_TO_EXPLORE[:] = [args.season_end_year]

    monthly_cases = load_cases_monthly(RDS_PATH, DATE_COL)
    logging.info(
        f"Series: {monthly_cases.index.min().date()} to "
        f"{monthly_cases.index.max().date()} ({len(monthly_cases)} months)"
    )
    logging.info(
        f"Model={args.model} | INPUT_CHUNK_LENGTH={INPUT_CHUNK_LENGTH} | "
        f"NUM_SAMPLES={NUM_SAMPLES} | transformations={args.transformation}"
    )

    origins_df = build_origins_df(monthly_cases)

    out_dir  = os.path.join(OUT_DIR, args.model.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)
    year_tag = str(args.season_end_year) if args.season_end_year else "all_years"

    for transformation in args.transformation:
        logging.info(f"--- Running transformation: {transformation} ---")
        df_out   = run_annual_forecast(
            monthly_cases  = monthly_cases,
            model_name     = args.model,
            origins_df     = origins_df,
            transformation = transformation,
        )
        out_path = os.path.join(
            out_dir,
            f"{args.model}_transform_{transformation}_annual_total_forecast_{year_tag}.csv"
        )
        df_out.to_csv(out_path, index=False)
        logging.info(f"Saved: {out_path}  ({len(df_out)} rows)")

    logging.info("Done.")