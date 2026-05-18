import pandas as pd
import numpy as np
import streamlit as st

from src.dashboard.config import (
    SEASON_1_START,
    SEASON_1_END,
    SEASON_2_START,
    SEASON_2_END,
)
from src.database.duckdb_utils import run_query


def _build_code_filter(consumption_codes):
    if consumption_codes is None:
        return ""
    codes_str = ", ".join(str(code) for code in consumption_codes)
    return f"AND consumption_code IN ({codes_str})"


def _build_day_type_filter(day_type):
    if day_type == "Alle":
        return ""
    if day_type == "Helligdag":
        return "AND is_holiday = TRUE"
    if day_type == "Helg":
        return "AND is_weekend = TRUE AND is_holiday = FALSE"
    return "AND is_weekend = FALSE AND is_holiday = FALSE"


def _build_month_filter(month_filter, timestamp_expr):
    month_map = {"November": 11, "Desember": 12, "Januar": 1}
    if month_filter in month_map:
        return f"AND MONTH({timestamp_expr}) = {month_map[month_filter]}"
    return ""


@st.cache_data
def get_forbruksdata(område="breive", limit=10000):
    table_name = f"forbruksdata_{område.lower()}"
    try:
        query = f"""
        SELECT
            timestamp,
            value_kwh,
            hour,
            weekday,
            month,
            is_weekend,
            is_holiday
        FROM {table_name}
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        df = run_query(query)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        st.error(f"Kunne ikke hente data for {område}: {e}")
        return pd.DataFrame()


@st.cache_data
def get_season_comparison(område="breive", day_type="Hverdag", month_filter="Alle", consumption_codes=None):
    table_name = f"forbruksdata_{område.lower()}"
    timestamp_expr = "CAST(timestamp AS TIMESTAMP)"
    code_filter = _build_code_filter(consumption_codes)
    day_type_filter = _build_day_type_filter(day_type)
    month_sql_filter = _build_month_filter(month_filter, timestamp_expr)

    query = f"""
    SELECT hour, season_label, AVG(value_kwh) AS avg_forbruk
    FROM (
        SELECT
            HOUR({timestamp_expr}) AS hour,
            'Før Norgespris' AS season_label,
            value_kwh
        FROM {table_name}
        WHERE DATE({timestamp_expr}) BETWEEN '{SEASON_1_START}' AND '{SEASON_1_END}'
          {day_type_filter}
          {month_sql_filter}
          {code_filter}
        UNION ALL
        SELECT
            HOUR({timestamp_expr}) AS hour,
            'Etter Norgespris' AS season_label,
            value_kwh
        FROM {table_name}
        WHERE DATE({timestamp_expr}) BETWEEN '{SEASON_2_START}' AND '{SEASON_2_END}'
          {day_type_filter}
          {month_sql_filter}
          {code_filter}
    )
    GROUP BY hour, season_label
    ORDER BY hour, season_label
    """

    try:
        return run_query(query)
    except Exception as e:
        st.error(f"Kunne ikke hente sesong-sammenligning: {e}")
        return pd.DataFrame()


@st.cache_data
def get_norgespris_user_stats(område="breive", month_filter="Alle"):
    table_name = f"norgespris_{område.lower()}"
    timestamp_expr = "CAST(timestamp AS TIMESTAMP)"
    month_sql_filter = _build_month_filter(month_filter, timestamp_expr)

    query = f"""
    SELECT count_total
    FROM {table_name}
    WHERE DATE({timestamp_expr}) BETWEEN '{SEASON_2_START}' AND '{SEASON_2_END}'
      {month_sql_filter}
    ORDER BY {timestamp_expr} DESC
    LIMIT 1
    """

    try:
        df = run_query(query)
        if not df.empty:
            return int(df.iloc[0]["count_total"])
        return 0
    except Exception as e:
        st.error(f"Kunne ikke hente Norgespris-statistikk: {e}")
        return 0


@st.cache_data
def get_total_users(område="breive", consumption_codes=None):
    table_name = f"forbruksdata_{område.lower()}"
    code_filter = _build_code_filter(consumption_codes)

    query = f"""
    SELECT COUNT(DISTINCT metering_point_anonymous) AS total_users
    FROM {table_name}
    WHERE 1=1
      {code_filter}
    """

    try:
        df = run_query(query)
        if not df.empty:
            return int(df.iloc[0]["total_users"])
        return 0
    except Exception as e:
        st.error(f"Kunne ikke hente totalt antall brukere: {e}")
        return 0


@st.cache_data
def get_weather_season_covariates(område="breive", day_type="Hverdag", month_filter="Alle", consumption_codes=None):
    consumption_table = f"forbruksdata_{område.lower()}"
    weather_table = f"værdata_{område.lower()}"
    timestamp_expr = "CAST(timestamp AS TIMESTAMP)"
    code_filter = _build_code_filter(consumption_codes)
    day_type_filter = _build_day_type_filter(day_type)
    month_sql_filter = _build_month_filter(month_filter, timestamp_expr)

    query = f"""
    WITH filtered_hours AS (
        SELECT DISTINCT
            DATE_TRUNC('hour', {timestamp_expr}) AS ts_hour,
            HOUR({timestamp_expr}) AS hour,
            'Før Norgespris' AS season_label
        FROM {consumption_table}
        WHERE DATE({timestamp_expr}) BETWEEN '{SEASON_1_START}' AND '{SEASON_1_END}'
          {day_type_filter}
          {month_sql_filter}
          {code_filter}
        UNION ALL
        SELECT DISTINCT
            DATE_TRUNC('hour', {timestamp_expr}) AS ts_hour,
            HOUR({timestamp_expr}) AS hour,
            'Etter Norgespris' AS season_label
        FROM {consumption_table}
        WHERE DATE({timestamp_expr}) BETWEEN '{SEASON_2_START}' AND '{SEASON_2_END}'
          {day_type_filter}
          {month_sql_filter}
          {code_filter}
    )
    SELECT
        fh.hour,
        fh.season_label,
        AVG(w.air_temperature) AS avg_temperature,
        AVG(w.wind_speed) AS avg_wind_speed,
        AVG(w.precipitation_mm) AS avg_precipitation_mm
    FROM filtered_hours fh
    JOIN {weather_table} w
      ON DATE_TRUNC('hour', CAST(w.timestamp AS TIMESTAMP)) = fh.ts_hour
    GROUP BY fh.hour, fh.season_label
    ORDER BY fh.hour, fh.season_label
    """

    try:
        return run_query(query)
    except Exception as e:
        st.error(f"Kunne ikke hente vær-sammenligning: {e}")
        return pd.DataFrame()


@st.cache_data
def get_weather_season_summary(område="breive", day_type="Hverdag", month_filter="Alle", consumption_codes=None):
    """
    Beregner gjennomsnittlig temperatur/vind/nedbør direkte fra rådata,
    ikke aggregert per time. Slik blir statistikken konsistent uavhengig av måned.
    """
    consumption_table = f"forbruksdata_{område.lower()}"
    weather_table = f"værdata_{område.lower()}"
    timestamp_expr = "CAST(timestamp AS TIMESTAMP)"
    code_filter = _build_code_filter(consumption_codes)
    day_type_filter = _build_day_type_filter(day_type)
    month_sql_filter = _build_month_filter(month_filter, timestamp_expr)

    query = f"""
    WITH filtered_days AS (
        SELECT DISTINCT
            DATE_TRUNC('day', {timestamp_expr}) AS ts_day,
            'Før Norgespris' AS season_label
        FROM {consumption_table}
        WHERE DATE({timestamp_expr}) BETWEEN '{SEASON_1_START}' AND '{SEASON_1_END}'
          {day_type_filter}
          {month_sql_filter}
          {code_filter}
        UNION ALL
        SELECT DISTINCT
            DATE_TRUNC('day', {timestamp_expr}) AS ts_day,
            'Etter Norgespris' AS season_label
        FROM {consumption_table}
        WHERE DATE({timestamp_expr}) BETWEEN '{SEASON_2_START}' AND '{SEASON_2_END}'
          {day_type_filter}
          {month_sql_filter}
          {code_filter}
    ),
    daily_weather AS (
        SELECT
            fd.season_label,
            fd.ts_day,
            AVG(w.air_temperature) AS avg_temperature,
            AVG(w.wind_speed) AS avg_wind_speed,
            AVG(w.precipitation_mm) AS avg_precipitation_mm
        FROM filtered_days fd
        JOIN {weather_table} w
          ON DATE_TRUNC('day', CAST(w.timestamp AS TIMESTAMP)) = fd.ts_day
        GROUP BY fd.season_label, fd.ts_day
    )
    SELECT
        season_label,
        AVG(avg_temperature) AS avg_temperature,
        AVG(avg_wind_speed) AS avg_wind_speed,
        AVG(avg_precipitation_mm) AS avg_precipitation_mm
    FROM daily_weather
    GROUP BY season_label
    """

    try:
        return run_query(query)
    except Exception as e:
        st.error(f"Kunne ikke hente værsammendrag: {e}")
        return pd.DataFrame()


@st.cache_data
def get_consumption_season_summary(område="breive", day_type="Hverdag", month_filter="Alle", consumption_codes=None):
    """
    Beregner gjennomsnittlig forbruk direkte fra rådata,
    ikke aggregert per time. Slik blir statistikken konsistent uavhengig av måned.
    """
    table_name = f"forbruksdata_{område.lower()}"
    timestamp_expr = "CAST(timestamp AS TIMESTAMP)"
    code_filter = _build_code_filter(consumption_codes)
    day_type_filter = _build_day_type_filter(day_type)
    month_sql_filter = _build_month_filter(month_filter, timestamp_expr)

    query = f"""
    SELECT
        'Før Norgespris' AS season_label,
        AVG(value_kwh) AS avg_forbruk,
        SUM(value_kwh) AS total_forbruk
    FROM {table_name}
    WHERE DATE({timestamp_expr}) BETWEEN '{SEASON_1_START}' AND '{SEASON_1_END}'
      {day_type_filter}
      {month_sql_filter}
      {code_filter}
    
    UNION ALL
    
    SELECT
        'Etter Norgespris' AS season_label,
        AVG(value_kwh) AS avg_forbruk,
        SUM(value_kwh) AS total_forbruk
    FROM {table_name}
    WHERE DATE({timestamp_expr}) BETWEEN '{SEASON_2_START}' AND '{SEASON_2_END}'
      {day_type_filter}
      {month_sql_filter}
      {code_filter}
    """

    try:
        return run_query(query)
    except Exception as e:
        st.error(f"Kunne ikke hente forbrukssammendrag: {e}")
        return pd.DataFrame()


def apply_weather_control(profile_df, weather_df, before_betas, after_betas, controls):
    if profile_df.empty or weather_df.empty or not controls:
        return profile_df

    weather_cols = {
        "Temperatur": "avg_temperature",
        "Vind": "avg_wind_speed",
        "Nedbør": "avg_precipitation_mm",
    }

    merged = profile_df.merge(weather_df, on=["hour", "season_label"], how="left")
    ref_weather = weather_df.groupby("hour", as_index=False)[
        ["avg_temperature", "avg_wind_speed", "avg_precipitation_mm"]
    ].mean().rename(columns={
        "avg_temperature": "ref_temperature",
        "avg_wind_speed": "ref_wind_speed",
        "avg_precipitation_mm": "ref_precipitation_mm",
    })

    merged = merged.merge(ref_weather, on="hour", how="left")
    merged["weather_effect_log"] = 0.0

    for control in controls:
        source_col = weather_cols.get(control)
        if source_col is None:
            continue

        beta_per_season = merged["season_label"].map({
            "Før Norgespris": float(before_betas.get(control, 0.0)),
            "Etter Norgespris": float(after_betas.get(control, 0.0)),
        }).fillna(0.0)

        ref_col = source_col.replace("avg_", "ref_")
        merged["weather_effect_log"] += beta_per_season * (merged[source_col] - merged[ref_col])

    merged["avg_forbruk"] = np.expm1(np.log1p(merged["avg_forbruk"].clip(lower=0)) - merged["weather_effect_log"])
    merged["avg_forbruk"] = merged["avg_forbruk"].clip(lower=0)

    return merged[["hour", "season_label", "avg_forbruk"]]
