from pathlib import Path
import sys
import os
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# Sørg for at prosjektroten er på Python import-stien, uansett hvor appen kjøres fra.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Gjør Streamlit secrets tilgjengelig som miljøvariabler i cloud,
# men uten å feile lokalt når secrets.toml mangler.
def _safe_get_secret(key: str):
    try:
        return st.secrets[key]
    except Exception:
        return None


azure_conn_secret = _safe_get_secret("AZURE_STORAGE_CONNECTION_STRING")
frost_client_secret = _safe_get_secret("FROST_CLIENT_ID")
duckdb_container_secret = _safe_get_secret("DUCKDB_CONTAINER")
duckdb_blob_secret = _safe_get_secret("DUCKDB_BLOB_NAME")

if azure_conn_secret and not os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = azure_conn_secret
if frost_client_secret and not os.getenv("FROST_CLIENT_ID"):
    os.environ["FROST_CLIENT_ID"] = frost_client_secret
if duckdb_container_secret and not os.getenv("DUCKDB_CONTAINER"):
    os.environ["DUCKDB_CONTAINER"] = duckdb_container_secret
if duckdb_blob_secret and not os.getenv("DUCKDB_BLOB_NAME"):
    os.environ["DUCKDB_BLOB_NAME"] = duckdb_blob_secret

# Importer database funksjoner
from src.database.duckdb_utils import run_query


BASE_DIR = Path(__file__).resolve().parent

def get_asset_path(path):
    return (BASE_DIR / "../assets" / path).resolve()

SEASON_1_START = "2024-11-01"
SEASON_1_END = "2025-01-31"
SEASON_2_START = "2025-11-01"
SEASON_2_END = "2026-01-31"

# Koeffisienter fra regresjon per stasjon.
STATION_WEATHER_BETAS = {
    "Breive": {"Temperatur": -0.0170, "Vind": 0.0073, "Nedbør": -0.0016},
    "Frikstad": {"Temperatur": -0.0208, "Vind": 0.0084, "Nedbør": 0.0049},
    "Hartevatn": {"Temperatur": -0.0164, "Vind": 0.0067, "Nedbør": -0.0022},
    "Timenes": {"Temperatur": -0.0179, "Vind": 0.0101, "Nedbør": 0.0038},
}

capgemini_logo = get_asset_path("images/Capgemini_201x_logo.svg")
a_energi_logo = get_asset_path("images/file.svg")

st.set_page_config(layout="wide")


# --- HJELPEFUNKSJONER FOR DATA ---

@st.cache_data
def get_forbruksdata(område="breive", limit=10000):
    """
    Henter forbruksdata fra DuckDB for valgt område.
    
    Args:
        område: "Breive", "Frikstad", "Hartevatn", eller "Timenes"
        limit: Maks antall rader å hente (for ytelse)
    
    Returns:
        pandas.DataFrame med forbruksdata
    """
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
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Kunne ikke hente data for {område}: {e}")
        return pd.DataFrame()


@st.cache_data
def get_time_series_data(område="breive", start_date=None, end_date=None, group_by="hour"):
    """
    Henter aggregert data for tidsserier.
    
    Args:
        område: Valgt område
        start_date, end_date: Dato-filter
        group_by: "hour", "day", "month"
    
    Returns:
        DataFrame med aggregert data
    """
    table_name = f"forbruksdata_{område.lower()}"
    timestamp_expr = "CAST(timestamp AS TIMESTAMP)"
    
    # Bygg WHERE klausul
    where_clauses = []
    if start_date:
        where_clauses.append(f"DATE({timestamp_expr}) >= '{start_date}'")
    if end_date:
        where_clauses.append(f"DATE({timestamp_expr}) <= '{end_date}'")
    
    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
    
    # Bygg GROUP BY basert på ønsket granularitet
    if group_by == "hour":
        group_sql = f"DATE({timestamp_expr}), HOUR({timestamp_expr}), norgespris"
        select_sql = f"DATE({timestamp_expr}) as date, HOUR({timestamp_expr}) as hour, norgespris"
    elif group_by == "day":
        group_sql = f"DATE({timestamp_expr}), norgespris"
        select_sql = f"DATE({timestamp_expr}) as date, norgespris"
    elif group_by == "month":
        group_sql = f"YEAR({timestamp_expr}), MONTH({timestamp_expr}), norgespris"
        select_sql = f"YEAR({timestamp_expr}) as year, MONTH({timestamp_expr}) as month, norgespris"
    
    query = f"""
    SELECT 
        {select_sql},
        AVG(value_kwh) as avg_forbruk,
        SUM(value_kwh) as total_forbruk,
        COUNT(*) as antall_målinger
    FROM {table_name}
    WHERE {where_sql}
    GROUP BY {group_sql}
    ORDER BY {group_sql}
    """
    
    try:
        df = run_query(query)
        return df
    except Exception as e:
        st.error(f"Kunne ikke hente tidsserie data: {e}")
        return pd.DataFrame()


@st.cache_data
def get_season_comparison(område="breive", day_type="Hverdag", month_filter="Alle", consumption_codes=None):
    """
    Henter gjennomsnittlig døgnprofil i to sesonger filtrert på helg/hverdag og helligdag.
    """
    table_name = f"forbruksdata_{område.lower()}"
    timestamp_expr = "CAST(timestamp AS TIMESTAMP)"

    code_filter = ""
    if consumption_codes is not None:
        codes_str = ", ".join(str(code) for code in consumption_codes)
        code_filter = f"AND consumption_code IN ({codes_str})"

    if day_type == "Alle":
        day_type_filter = ""
    elif day_type == "Helligdag":
        day_type_filter = "AND is_holiday = TRUE"
    elif day_type == "Helg":
        day_type_filter = "AND is_weekend = TRUE AND is_holiday = FALSE"
    else:  # Hverdag
        day_type_filter = "AND is_weekend = FALSE AND is_holiday = FALSE"

    month_map = {"November": 11, "Desember": 12, "Januar": 1}
    if month_filter in month_map:
        month_sql_filter = f"AND MONTH({timestamp_expr}) = {month_map[month_filter]}"
    else:
        month_sql_filter = ""

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
        df = run_query(query)
        return df
    except Exception as e:
        st.error(f"Kunne ikke hente sesong-sammenligning: {e}")
        return pd.DataFrame()
    
@st.cache_data
def get_norgespris_user_stats(område="breive", month_filter="Alle"):
    """
    Henter antall brukere med Norgespris fra ferdig aggregert tabell.
    """

    table_name = f"norgespris_{område.lower()}"
    timestamp_expr = "CAST(timestamp AS TIMESTAMP)"

    month_map = {
        "November": 11,
        "Desember": 12,
        "Januar": 1
    }

    month_sql_filter = ""
    if month_filter in month_map:
        month_sql_filter = f"AND MONTH({timestamp_expr}) = {month_map[month_filter]}"

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

    code_filter = ""
    if consumption_codes is not None:
        codes_str = ", ".join(str(code) for code in consumption_codes)
        code_filter = f"AND consumption_code IN ({codes_str})"

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
def get_temperature_season_comparison(område="breive", day_type="Hverdag", month_filter="Alle", consumption_codes=None):
    """
    Henter gjennomsnittlig temperaturprofil per time i to sesonger basert på samme filtre som forbruksgrafen.
    """
    consumption_table = f"forbruksdata_{område.lower()}"
    weather_table = f"værdata_{område.lower()}"
    timestamp_expr = "CAST(timestamp AS TIMESTAMP)"

    code_filter = ""
    if consumption_codes is not None:
        codes_str = ", ".join(str(code) for code in consumption_codes)
        code_filter = f"AND consumption_code IN ({codes_str})"

    if day_type == "Alle":
        day_type_filter = ""
    elif day_type == "Helligdag":
        day_type_filter = "AND is_holiday = TRUE"
    elif day_type == "Helg":
        day_type_filter = "AND is_weekend = TRUE AND is_holiday = FALSE"
    else:
        day_type_filter = "AND is_weekend = FALSE AND is_holiday = FALSE"

    month_map = {"November": 11, "Desember": 12, "Januar": 1}
    if month_filter in month_map:
        month_sql_filter = f"AND MONTH({timestamp_expr}) = {month_map[month_filter]}"
    else:
        month_sql_filter = ""

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
        AVG(w.air_temperature) AS avg_temperature
    FROM filtered_hours fh
    JOIN {weather_table} w
      ON DATE_TRUNC('hour', CAST(w.timestamp AS TIMESTAMP)) = fh.ts_hour
    GROUP BY fh.hour, fh.season_label
    ORDER BY fh.hour, fh.season_label
    """

    try:
        df = run_query(query)
        return df
    except Exception as e:
        st.error(f"Kunne ikke hente temperatur-sammenligning: {e}")
        return pd.DataFrame()


@st.cache_data
def get_weather_season_covariates(område="breive", day_type="Hverdag", month_filter="Alle", consumption_codes=None):
    """
    Henter gjennomsnittlig værprofil per time i to sesonger (temperatur, vind og nedbør)
    basert på samme filtre som forbruksgrafen.
    """
    consumption_table = f"forbruksdata_{område.lower()}"
    weather_table = f"værdata_{område.lower()}"
    timestamp_expr = "CAST(timestamp AS TIMESTAMP)"

    code_filter = ""
    if consumption_codes is not None:
        codes_str = ", ".join(str(code) for code in consumption_codes)
        code_filter = f"AND consumption_code IN ({codes_str})"

    if day_type == "Alle":
        day_type_filter = ""
    elif day_type == "Helligdag":
        day_type_filter = "AND is_holiday = TRUE"
    elif day_type == "Helg":
        day_type_filter = "AND is_weekend = TRUE AND is_holiday = FALSE"
    else:
        day_type_filter = "AND is_weekend = FALSE AND is_holiday = FALSE"

    month_map = {"November": 11, "Desember": 12, "Januar": 1}
    if month_filter in month_map:
        month_sql_filter = f"AND MONTH({timestamp_expr}) = {month_map[month_filter]}"
    else:
        month_sql_filter = ""

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


def apply_weather_control(profile_df, weather_df, betas, controls):
    """
    Justerer timeprofilen til en felles værreferanse per time.
    Dette gjør før/etter-sammenligningen mindre følsom for ulike værforhold.
    """
    if profile_df.empty or weather_df.empty or not controls:
        return profile_df

    weather_cols = {
        "Temperatur": ("avg_temperature", float(betas.get("Temperatur", 0.0))),
        "Vind": ("avg_wind_speed", float(betas.get("Vind", 0.0))),
        "Nedbør": ("avg_precipitation_mm", float(betas.get("Nedbør", 0.0))),
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
        source_col, beta = weather_cols.get(control, (None, 0.0))
        if source_col is None:
            continue
        ref_col = source_col.replace("avg_", "ref_")
        merged["weather_effect_log"] += beta * (merged[source_col] - merged[ref_col])

    merged["avg_forbruk"] = np.expm1(np.log1p(merged["avg_forbruk"].clip(lower=0)) - merged["weather_effect_log"])
    merged["avg_forbruk"] = merged["avg_forbruk"].clip(lower=0)

    return merged[["hour", "season_label", "avg_forbruk"]]


@st.cache_data
def get_before_after_norgespris(område="breive", start_date=None, end_date=None, aggregate="day"):
    """
    Henter forbruk før og etter norgespris-flagget.
    """
    table_name = f"forbruksdata_{område.lower()}"
    timestamp_expr = "CAST(timestamp AS TIMESTAMP)"

    where_clauses = []
    if start_date:
        where_clauses.append(f"DATE({timestamp_expr}) >= '{start_date}'")
    if end_date:
        where_clauses.append(f"DATE({timestamp_expr}) <= '{end_date}'")

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    if aggregate == "hour":
        group_sql = f"DATE({timestamp_expr}), HOUR({timestamp_expr}), norgespris"
        select_sql = f"DATE({timestamp_expr}) as date, HOUR({timestamp_expr}) as hour, norgespris"
    elif aggregate == "month":
        group_sql = f"YEAR({timestamp_expr}), MONTH({timestamp_expr}), norgespris"
        select_sql = f"YEAR({timestamp_expr}) as year, MONTH({timestamp_expr}) as month, norgespris"
    else:
        group_sql = f"DATE({timestamp_expr}), norgespris"
        select_sql = f"DATE({timestamp_expr}) as date, norgespris"

    query = f"""
    SELECT
        {select_sql},
        AVG(value_kwh) as avg_forbruk,
        SUM(value_kwh) as total_forbruk
    FROM {table_name}
    WHERE {where_sql}
    GROUP BY {group_sql}
    ORDER BY {group_sql}
    """

    try:
        df = run_query(query)
        return df
    except Exception as e:
        st.error(f"Kunne ikke hente før/etter Norgespris data: {e}")
        return pd.DataFrame()


# --- CSS for smooth scroll + styling ---
st.markdown("""
<style>
html {
    scroll-behavior: smooth;
}

.logo-img {
    max-height: 50px;
}

.header-btn {
    padding: 10px 20px;
    border-radius: 10px;
    border: 1px solid #ccc;
    background-color: #f5f5f5;
}
</style>
""", unsafe_allow_html=True)


# --- HEADER ---
col1, col2 = st.columns([6, 1])

with col1:
    st.markdown("## Norgespris – Analyse av strømforbruk")

with col2:
    st.markdown("""
    <a href="#prosjektet">
        <button style="
            padding:10px 20px;
            border-radius:10px;
            border:1px solid #ccc;
            background-color:#f5f5f5;
        ">
            Prosjektet
        </button>
    </a>
    """, unsafe_allow_html=True)

st.markdown("---")


# --- FORKLAR GRAF ---
st.markdown("## Hva viser grafen?")
st.write("""
Denne grafen viser gjennomsnittlig døgnprofil for valgt trafostasjon.
""")


# --- FILTER + GRAF ---
col_filter, col_graph = st.columns([1, 4])

with col_filter:
    st.markdown("### Filter")
    
    # Område-valg
    område_options = ["Breive", "Frikstad", "Hartevatn", "Timenes"]
    selected_område = st.selectbox("Område", område_options, key="area")
    
    selected_day_type = st.selectbox(
        "Dagtype",
        ["Alle", "Hverdag", "Helg", "Helligdag"],
        index=0,
        key="selected_day_type"
    )

    selected_month = st.selectbox(
        "Måned",
        ["Alle", "November", "Desember", "Januar"],
        index=0,
        key="selected_month"
    )

    # Visningsalternativer
    chart_type = st.selectbox("Graf-type", 
                              ["Linje", "Stolpe", "Område"],
                              key="chart_type")
    
    # Consumption code filter
    consumption_code_option = st.selectbox(
        "Forbrukstype",
        ["Boliger", "Fritidsboliger", "Begge"],
        index=2,
        key="consumption_code"
    )
    if consumption_code_option == "Boliger":
        selected_consumption_codes = [35]
    elif consumption_code_option == "Fritidsboliger":
        selected_consumption_codes = [36]
    else:
        selected_consumption_codes = [35, 36]

    weather_control_enabled = st.toggle(
        "Kontroller for vær (regresjon)",
        value=False,
        help="Justerer kurvene med beta-verdier fra regresjonsanalysen."
    )

    selected_weather_controls = []
    default_weather_betas = STATION_WEATHER_BETAS.get(
        selected_område,
        {"Temperatur": 0.0, "Vind": 0.0, "Nedbør": 0.0},
    )
    weather_betas = default_weather_betas.copy()

    if weather_control_enabled:
        st.caption(f"Stasjonskoeffisienter for {selected_område}. Du kan overstyre manuelt under.")
        selected_weather_controls = st.multiselect(
            "Aktive kontroller",
            ["Temperatur", "Vind", "Nedbør"],
            default=["Temperatur", "Vind", "Nedbør"],
            key="selected_weather_controls",
        )

        weather_betas["Temperatur"] = st.number_input(
            "Beta temperatur",
            value=float(default_weather_betas["Temperatur"]),
            format="%.6f",
            key=f"beta_temperature_{selected_område}",
        )
        weather_betas["Vind"] = st.number_input(
            "Beta vind",
            value=float(default_weather_betas["Vind"]),
            format="%.6f",
            key=f"beta_wind_{selected_område}",
        )
        weather_betas["Nedbør"] = st.number_input(
            "Beta nedbør",
            value=float(default_weather_betas["Nedbør"]),
            format="%.6f",
            key=f"beta_precipitation_{selected_område}",
        )

    selected_metric = "avg_forbruk"

with col_graph:
    # Hent data basert på filtre
    with st.spinner("Henter data..."):
        df = get_season_comparison(
            område=selected_område,
            day_type=selected_day_type,
            month_filter=selected_month,
            consumption_codes=selected_consumption_codes
        )
        weather_cov_df = get_weather_season_covariates(
            område=selected_område,
            day_type=selected_day_type,
            month_filter=selected_month,
            consumption_codes=selected_consumption_codes
        )
        temp_df = weather_cov_df[["hour", "season_label", "avg_temperature"]].copy() if not weather_cov_df.empty else pd.DataFrame()
    
    if not df.empty:
        st.markdown(f"### Gjennomsnittlig døgnprofil - {selected_område.title()}")
        st.write(
            f"Sammenligner dagtype: {selected_day_type.lower()} "
            f"og måned: {selected_month.lower()} for sesongene \"Før Norgespris\" (2024-2025) og \"Etter Norgespris\" (2025-2026)."
        )
        
        df = df.sort_values('hour')
        plot_df = df[["hour", "season_label", "avg_forbruk"]].copy()

        if weather_control_enabled:
            if selected_weather_controls and not weather_cov_df.empty:
                plot_df = apply_weather_control(
                    profile_df=plot_df,
                    weather_df=weather_cov_df,
                    betas=weather_betas,
                    controls=selected_weather_controls,
                )
            elif selected_weather_controls and weather_cov_df.empty:
                st.info("Fant ikke værdata for valgt filter, viser ukontrollert graf.")

        if weather_control_enabled and selected_weather_controls:
            st.caption("Grafen er værjustert med valgte regresjonskoeffisienter.")

        pivot_df = plot_df.pivot(index='hour', columns='season_label', values='avg_forbruk')

        if chart_type == "Linje":

            y_min = plot_df["avg_forbruk"].min()
            y_max = plot_df["avg_forbruk"].max()

            padding = (y_max - y_min) * 0.1

            chart = alt.Chart(plot_df).mark_line(point=True).encode(
                x=alt.X("hour:O", title="Time"),
                y=alt.Y(
                    "avg_forbruk:Q",
                    title="Forbruk (kWh)",
                    scale=alt.Scale(domain=[y_min - padding, y_max + padding])
                ),
                color=alt.Color("season_label:N", title="Sesong"),
                tooltip=[
                        alt.Tooltip("hour:O", title="Time"),
                        alt.Tooltip("avg_forbruk:Q", title="Forbruk (kWh)", format=".2f"),
                        alt.Tooltip("season_label:N", title="Sesong")
]
            ).properties(height=400)

            st.altair_chart(chart, use_container_width=True)

        elif chart_type == "Stolpe":
            st.bar_chart(pivot_df)

        elif chart_type == "Område":
            st.area_chart(pivot_df)

        # Vis statistikk
        norgespris_users = get_norgespris_user_stats(
            område=selected_område,
            month_filter=selected_month
        )

        total_users = get_total_users(
            område=selected_område,
            consumption_codes=selected_consumption_codes
        )
        avg_per_season = pivot_df.mean()

        before_avg = avg_per_season.get("Før Norgespris", None)
        after_avg = avg_per_season.get("Etter Norgespris", None)

        st.markdown("### Gjennomsnittlig forbruk")
        
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

        with col_stats1:
            if before_avg is not None:
                st.metric("Før Norgespris", f"{before_avg:.2f} kWh")

        with col_stats2:
            if after_avg is not None:
                st.metric("Etter Norgespris", f"{after_avg:.2f} kWh")
        with col_stats3:
            if before_avg is not None and after_avg is not None:
                if before_avg > 0:
                    change_pct = ((after_avg - before_avg) / before_avg) * 100
                    change_str = f"{change_pct:+.2f}%"
                else:
                    change_str = "N/A"
                st.metric("Prosentvis endring", change_str)
        with col_stats4:
            st.metric(
                f"Brukere med Norgespris i {selected_område}",
                f"{norgespris_users} av {total_users}"
)

        if not temp_df.empty:
            st.markdown("#### Gjennomsnittlig temperatur (°C)")
            temp_df = temp_df.sort_values("hour")
            temp_pivot_df = temp_df.pivot(index="hour", columns="season_label", values="avg_temperature")

            temp_avg_per_season = temp_pivot_df.mean()
            temp_before_avg = temp_avg_per_season.get("Før Norgespris", None)
            temp_after_avg = temp_avg_per_season.get("Etter Norgespris", None)

            temp_plot_df = temp_pivot_df.reset_index().melt(
                id_vars="hour",
                var_name="season_label",
                value_name="avg_temperature"
            )

            temp_y_min = temp_plot_df["avg_temperature"].min()
            temp_y_max = temp_plot_df["avg_temperature"].max()
            temp_padding = (temp_y_max - temp_y_min) * 0.1

            chart_temp = alt.Chart(temp_plot_df).mark_line(point=True).encode(
                x=alt.X("hour:O", title="Time"),
                y=alt.Y(
                    "avg_temperature:Q",
                    title="Temperatur (°C)",
                    scale=alt.Scale(domain=[temp_y_min - temp_padding, temp_y_max + temp_padding])
                ),
                color=alt.Color("season_label:N", title="Sesong"),
                tooltip=[
                    alt.Tooltip("hour:O", title="Time"),
                    alt.Tooltip("avg_temperature:Q", title="Temperatur (°C)", format=".2f"),
                    alt.Tooltip("season_label:N", title="Sesong")
                ]
            ).properties(height=400)

            st.altair_chart(chart_temp, use_container_width=True)


            temp_col1, temp_col2 = st.columns(2)
            with temp_col1:
                if temp_before_avg is not None:
                    st.metric("Gjennomsnittstemperatur før Norgespris", f"{temp_before_avg:.2f} °C")
            with temp_col2:
                if temp_after_avg is not None:
                    st.metric("Gjennomsnittstemperatur etter Norgespris", f"{temp_after_avg:.2f} °C")
        
            
    else:
        if selected_day_type == "Helligdag" and selected_month in ["November", "Januar"]:
            st.warning(
                "Ingen data funnet: Helligdager finnes kun i desember i dette datasettet. "
                "Velg måned Desember eller endre dagtype."
            )
        else:
            st.warning("Ingen data funnet for valgte filtre. Prøv å justere filtrene eller området.")
        
        # Vis eksempel på tilgjengelige data
        st.markdown("**Tilgjengelige områder:**")
        for område in område_options:
            try:
                test_df = get_forbruksdata(område, limit=1)
                if not test_df.empty:
                    st.write(f" {område.title()}: Data tilgjengelig")
                else:
                    st.write(f" {område.title()}: Ingen data")
            except:
                st.write(f" {område.title()}: Feil ved tilkobling")


st.markdown("---")


# --- PROSJEKTSEKSJON (ID FOR SCROLL) ---
st.markdown('<div id="prosjektet"></div>', unsafe_allow_html=True)

st.markdown("## Prosjektet")

col_img, col_text = st.columns([1, 3])

with col_img:
    st.image("https://via.placeholder.com/300")

with col_text:
    st.write("Om prosjektgruppen og samarbeid...")



