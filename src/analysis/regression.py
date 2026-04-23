import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


def prepare_norgespris_regression_data(
    df_consumption,
    df_weather=None,
    df_norgespris=None,
    station_customers_total=None,
    exclude_consumption_codes=(26,),
):
    df = df_consumption.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.normalize()

    if "norgespris" not in df.columns:
        raise ValueError("df_consumption må inneholde kolonnen 'norgespris'.")

    if station_customers_total is None:
        station_customers_total = df["metering_point_anonymous"].nunique()
    if float(station_customers_total) <= 0:
        raise ValueError("station_customers_total må være et positivt tall.")

    # Fjern kundetyper som ikke kan ha Norgespris (f.eks. 26 = næring).
    if "consumption_code" in df.columns and exclude_consumption_codes:
        df = df[~df["consumption_code"].isin(exclude_consumption_codes)].copy()

    # Periodemarkør 0/1 fra forbruksfilen
    df = df[df["norgespris"].notna()].copy()
    df["norgespris"] = df["norgespris"].astype(int)

    if df_weather is not None:
        weather_df = df_weather.copy()
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
        df = df.merge(weather_df, on="timestamp", how="left")

    if df_norgespris is not None and {"transformer_station", "count_total", "timestamp"}.issubset(df_norgespris.columns):
        station_daily = df_norgespris.copy()
        station_daily["timestamp"] = pd.to_datetime(station_daily["timestamp"])
        station_daily["date"] = station_daily["timestamp"].dt.normalize()

        # Ett daglig nivå per stasjon
        station_daily = (
            station_daily.groupby(["date", "transformer_station"], as_index=False)["count_total"]
            .max()
        )

        # Bruk kjent total kundebase i denne trafostasjonen
        station_daily["station_customers_total"] = float(station_customers_total)

        # Andel med Norgespris = antall med avtale / total kundebase
        station_daily["norgespris_share"] = station_daily["count_total"] / station_daily["station_customers_total"]

        # Slå på stasjonsnivå + dato
        df = df.merge(
            station_daily[["date", "transformer_station", "count_total", "station_customers_total", "norgespris_share"]],
            on=["date", "transformer_station"],
            how="left",
        )

    else:
        df["count_total"] = np.nan
        df["station_customers_total"] = float(station_customers_total)
        df["norgespris_share"] = np.nan

    # df_norgespris finnes kun på dager med norgespris=1. Sett 0 i før-periode.
    df.loc[df["norgespris"] == 0, ["count_total", "norgespris_share"]] = 0.0

    # Robust fallback om det mangler noen match i etter-periode
    df["count_total"] = df["count_total"].fillna(0.0)
    df["norgespris_share"] = df["norgespris_share"].fillna(0.0)

    # Klipp andel til [0, 1] for å unngå ekstreme verdier ved eventuelle datamismatch
    df["norgespris_share"] = df["norgespris_share"].clip(lower=0.0, upper=1.0)

    return df


def fit_best_norgespris_model(df, dependent_variable="value_kwh"):
    required_columns = {
        "metering_point_anonymous",
        "timestamp",
        "transformer_station",
        dependent_variable,
        "norgespris",
        "norgespris_share",
        "air_temperature",
        "wind_speed",
        "precipitation_mm",
        "hour",
        "is_weekend",
        "month",
        "is_holiday",
    }

    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Datasettet mangler nødvendige kolonner: {missing_text}")

    modeling_df = df.copy()
    modeling_df["timestamp"] = pd.to_datetime(modeling_df["timestamp"])
    modeling_df["metering_point_anonymous"] = modeling_df["metering_point_anonymous"].astype(str)
    # Sikrer at indikatorer og tidskomponenter behandles som diskrete numeriske variabler.
    modeling_df["is_weekend"] = modeling_df["is_weekend"].astype(int)
    modeling_df["is_holiday"] = modeling_df["is_holiday"].astype(int)
    modeling_df["hour"] = modeling_df["hour"].astype(int)
    modeling_df["month"] = modeling_df["month"].astype(int)
    modeling_df["norgespris"] = modeling_df["norgespris"].astype(int)
    modeling_df["norgespris_share"] = modeling_df["norgespris_share"].astype(float)
    # Viktig: log1p betyr log(1 + forbruk), ikke vanlig log(forbruk).
    modeling_df["log_value_kwh"] = np.log1p(modeling_df[dependent_variable])

    modeling_df = modeling_df.dropna(
        subset=[
            dependent_variable,
            "norgespris",
            "norgespris_share",
            "air_temperature",
            "wind_speed",
            "precipitation_mm",
            "hour",
            "is_weekend",
            "month",
            "is_holiday",
            "metering_point_anonymous",
        ]
    ).copy()

    # Dato for time fixed effects
    modeling_df["date"] = modeling_df["timestamp"].dt.normalize()

    panel_df = modeling_df.set_index(["metering_point_anonymous", "timestamp"]).sort_index()

    # Modell
    formula = (
    "log_value_kwh ~ 1 + norgespris_share + air_temperature + wind_speed + precipitation_mm "
    "+ is_weekend + is_holiday + C(hour) + C(month) + EntityEffects"
    )

    model = PanelOLS.from_formula(formula, data=panel_df, drop_absorbed=True)

    # Clustered standard errors
    results = model.fit(
        cov_type="clustered",
        cluster_entity=True,
        low_memory=True,
        use_lsmr=True
    )

    beta_share = float(results.params.get("norgespris_share", 0.0))

    # Kontrafaktisk uten Norgespris-andel: hold alt annet likt, fjern kun beta*share-leddet.
    observed_kwh = modeling_df[dependent_variable].astype(float)
    share_effect_log = beta_share * modeling_df["norgespris_share"]
    counterfactual_no_np_kwh = np.expm1(np.log1p(observed_kwh) - share_effect_log)
    attributable_kwh = observed_kwh - counterfactual_no_np_kwh

    post_mask = modeling_df["norgespris"] == 1
    total_observed_post_kwh = float(observed_kwh.loc[post_mask].sum())
    total_attributable_post_kwh = float(attributable_kwh.loc[post_mask].sum())
    attributable_share_of_post_kwh_pct = (
        float(total_attributable_post_kwh / total_observed_post_kwh * 100)
        if total_observed_post_kwh > 0
        else 0.0
    )

    unique_customers = int(modeling_df["metering_point_anonymous"].nunique())
    post_days = int(modeling_df.loc[post_mask, "timestamp"].dt.normalize().nunique()) if post_mask.any() else 0
    attributable_per_customer_post_kwh = (
        float(total_attributable_post_kwh / unique_customers)
        if unique_customers > 0
        else 0.0
    )
    attributable_per_customer_post_day_kwh = (
        float(total_attributable_post_kwh / (unique_customers * post_days))
        if unique_customers > 0 and post_days > 0
        else 0.0
    )

    metrics = {
        "nobs": int(results.nobs),
        "r2_within": float(results.rsquared_within),
        "beta_share": beta_share,
        # Effekt hvis andelen går fra 0% til 100%
        "effect_full_share_pct": float((np.exp(beta_share) - 1) * 100),
        # Mer praktisk tolkning: effekt ved +10 prosentpoeng
        "effect_10pp_pct": float((np.exp(beta_share * 0.10) - 1) * 100),
        "mean_share_post": float(
            modeling_df.loc[post_mask, "norgespris_share"].mean()
        ) if post_mask.any() else 0.0,
        # Total effekt i kWh i etterperioden
        "total_observed_post_kwh": total_observed_post_kwh,
        "total_attributable_post_kwh": total_attributable_post_kwh,
        "attributable_share_of_post_kwh_pct": attributable_share_of_post_kwh_pct,
        "attributable_per_customer_post_kwh": attributable_per_customer_post_kwh,
        "attributable_per_customer_post_day_kwh": attributable_per_customer_post_day_kwh,
    }

    return results, modeling_df, metrics