import math

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


WEATHER_VARIABLES = ["air_temperature", "wind_speed", "precipitation_mm"]


def _two_sided_pvalue_from_z(z_value: float) -> float:
    """Compute two-sided p-value from standard normal z-score."""
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def _fit_single_period_weather_model(
    period_df: pd.DataFrame,
    weather_variables: list[str],
    dependent_variable: str,
):
    """Fit a smaller panel model for a single period and return results plus weather p-values."""
    period_df = period_df.copy()
    period_df["timestamp"] = pd.to_datetime(period_df["timestamp"])
    period_df["metering_point_anonymous"] = period_df["metering_point_anonymous"].astype(str)
    period_df["is_weekend"] = period_df["is_weekend"].astype(int)
    period_df["is_holiday"] = period_df["is_holiday"].astype(int)
    period_df["hour"] = period_df["hour"].astype(int)
    period_df["month"] = period_df["month"].astype(int)
    period_df["log_value_kwh"] = np.log1p(period_df[dependent_variable])

    period_df = period_df.dropna(
        subset=[
            dependent_variable,
            "hour",
            "is_weekend",
            "month",
            "is_holiday",
            "metering_point_anonymous",
            *weather_variables,
        ]
    ).copy()

    panel_df = period_df.set_index(["metering_point_anonymous", "timestamp"]).sort_index()

    weather_main = " + ".join(weather_variables)
    formula = (
        f"log_value_kwh ~ 1 + {weather_main} + is_weekend + is_holiday + C(hour) + C(month) + EntityEffects"
    )

    model = PanelOLS.from_formula(formula, data=panel_df, drop_absorbed=True)
    results = model.fit(
        cov_type="clustered",
        cluster_entity=True,
        low_memory=True,
    )

    weather_rows = []
    for var in weather_variables:
        weather_rows.append(
            {
                "variabel": var,
                "beta": float(results.params.get(var, np.nan)),
                "p_verdi": float(results.pvalues.get(var, np.nan)),
            }
        )

    return results, pd.DataFrame(weather_rows)


def fit_weather_before_after_model(
    regression_df: pd.DataFrame,
    dependent_variable: str = "value_kwh",
    weather_variables: list[str] | None = None,
):
    """
    Estimate separate panel models for the pre and post Norgespris periods.

    Returns
    -------
    pre_results : PanelEffectsResults
        Fitted PanelOLS results for the pre-period model.
    post_results : PanelEffectsResults
        Fitted PanelOLS results for the post-period model.
    weather_pvalues_df : pd.DataFrame
        Table with weather coefficients and p-values before and after Norgespris.
    """
    weather_variables = weather_variables or WEATHER_VARIABLES

    required_columns = {
        "metering_point_anonymous",
        "timestamp",
        dependent_variable,
        "norgespris",
        "norgespris_share",
        "hour",
        "is_weekend",
        "month",
        "is_holiday",
        *weather_variables,
    }

    missing_columns = required_columns.difference(regression_df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Datasettet mangler nødvendige kolonner: {missing_text}")

    period_df = regression_df.copy()
    period_df["timestamp"] = pd.to_datetime(period_df["timestamp"])
    period_df["norgespris"] = period_df["norgespris"].astype(int)

    pre_df = period_df.loc[period_df["norgespris"] == 0].copy()
    post_df = period_df.loc[period_df["norgespris"] == 1].copy()

    pre_results, pre_weather_df = _fit_single_period_weather_model(
        pre_df,
        weather_variables=weather_variables,
        dependent_variable=dependent_variable,
    )
    post_results, post_weather_df = _fit_single_period_weather_model(
        post_df,
        weather_variables=weather_variables,
        dependent_variable=dependent_variable,
    )

    weather_pvalues_df = pre_weather_df.rename(
        columns={"beta": "beta_før", "p_verdi": "p_verdi_før"}
    ).merge(
        post_weather_df.rename(columns={"beta": "beta_etter", "p_verdi": "p_verdi_etter"}),
        on="variabel",
        how="outer",
    )

    return pre_results, post_results, weather_pvalues_df
