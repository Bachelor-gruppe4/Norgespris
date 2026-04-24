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

# Koeffisienter før Norgespris, brukt for værkontroll i grafen
STATION_WEATHER_BETAS_BEFORE = {
    "Breive": {"Temperatur": -0.014512, "Vind": 0.004102, "Nedbør": 0.011508},
    "Frikstad": {"Temperatur": -0.018790, "Vind": 0.005962, "Nedbør": 0.010317},
    "Hartevatn": {"Temperatur": -0.014199, "Vind": 0.004867, "Nedbør": 0.010142},
    "Timenes": {"Temperatur": -0.016093, "Vind": 0.007853, "Nedbør": 0.009971},
}

# Koeffisienter etter Norgespris, brukt for værkontroll i grafen
STATION_WEATHER_BETAS_AFTER = {
    "Breive": {"Temperatur": -0.019255, "Vind": 0.011599, "Nedbør": -0.015119},
    "Frikstad": {"Temperatur": -0.021582, "Vind": 0.008675, "Nedbør": 0.001878},
    "Hartevatn": {"Temperatur": -0.017798, "Vind": 0.008936, "Nedbør": -0.01540},
    "Timenes": {"Temperatur": -0.018753, "Vind": 0.010886, "Nedbør": 0.001263},
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


def apply_weather_control(profile_df, weather_df, before_betas, after_betas, controls):
    """
    Justerer timeprofilen til en felles værreferanse per time.
    Dette gjør før/etter-sammenligningen mindre følsom for ulike værforhold.
    """
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

:root {
    --aenergi-accent: #FFBBFC;
    --aenergi-deep: #3C000F;
    --aenergi-number: #7D283D;
    --aenergi-burgundy: #7D283D;
}

/* Global tekstfarge */
.stApp,
.stApp * {
    color: var(--aenergi-burgundy) !important;
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

/* Nullstill ekstra kort-styling på hver metric */
div[data-testid="stMetric"] {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
}

/* Rosa bakgrunn på hovedboksene */
.st-key-consumption_box,
.st-key-norgespris_box {
    background: #fff3fe;
    border: 1px solid #fff3fe;
    border-radius: 12px;
    padding: 0.55rem 0.75rem;
}

/* Label på metrikker */
div[data-testid="stMetricLabel"] {
    color: var(--aenergi-burgundy) !important;
}

/* Selve tallene */
div[data-testid="stMetricValue"] {
    color: var(--aenergi-burgundy) !important;
}

/* Toggle-bryteren - burgunder tema */
div[data-testid="stToggle"] [role="switch"] {
    background-color: #7D283D !important;
    border: 1px solid #7D283D !important;
}

div[data-testid="stToggle"] [role="switch"][aria-checked="true"] {
    background-color: #7D283D !important;
    border-color: #7D283D !important;
}

/* Eksplisitt overstyring for vær-toggle (key=weather_control_enabled) */
div[data-testid="stToggle"] input[type="checkbox"][id="weather_control_enabled"] + div,
div[data-testid="stToggle"] input[type="checkbox"][id="weather_control_enabled"]:checked + div,
div[data-testid="stToggle"] input[type="checkbox"][id="weather_control_enabled"] + div > div,
div[data-testid="stToggle"] input[type="checkbox"][id="weather_control_enabled"]:checked + div > div {
    background-color: #7D283D !important;
    border-color: #7D283D !important;
}

div[data-testid="stToggle"] input[type="checkbox"][id="weather_control_enabled"] + div > div {
    background-color: #ffffff !important;
}

/* Multiselect-tags - burgunder tema */
div[data-testid="stMultiSelect"] .stTags {
    background-color: #7D283D !important;
    color: #FFBBFC !important;
}

div[data-testid="stMultiSelect"] [role="button"][aria-selected="true"] {
    background-color: #7D283D !important;
    border-color: #7D283D !important;
    color: #FFBBFC !important;
}

/* Multiselect input-boks */
div[data-testid="stMultiSelect"] > div > div {
    border-color: #7D283D !important;
}

/* Burgunder outline ved hover/fokus (ikke rød) */
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:hover,
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
    border-color: #7D283D !important;
    box-shadow: 0 0 0 1px #7D283D !important;
}

/* Multiselect pills/tags */
span[data-baseweb="tag"] {
    background-color: #7D283D !important;
    color: #FFBBFC !important;
}

/* Tving rosa tekst inni valgte piller (tekst + x-ikon) */
span[data-baseweb="tag"],
span[data-baseweb="tag"] *,
div[data-testid="stMultiSelect"] [data-baseweb="tag"],
div[data-testid="stMultiSelect"] [data-baseweb="tag"] * {
    color: #FFBBFC !important;
    fill: #FFBBFC !important;
}
</style>
""", unsafe_allow_html=True)


# --- HEADER ---
col1, col2 = st.columns([6, 1])

with col1:
    st.markdown(
        "<h2>Norgespris – Analyse av strømforbruk</h2>",
        unsafe_allow_html=True,
    )


st.markdown("---")


# --- FORKLAR GRAF ---
st.header("Hva viser grafen?", anchor=False)
st.write("""
Denne grafen viser gjennomsnittlig døgnprofil for valgt trafostasjon.
""")


# --- FILTER + GRAF ---
col_filter, col_graph = st.columns([1, 4])

with col_filter:
    st.subheader("Filter", anchor=False)
    
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
        key="weather_control_enabled",
        help="Justerer kurvene med beta-verdier fra regresjonsanalysen."
    )

    selected_weather_controls = []
    default_weather_betas_before = STATION_WEATHER_BETAS_BEFORE.get(
        selected_område,
        {"Temperatur": 0.0, "Vind": 0.0, "Nedbør": 0.0},
    )
    default_weather_betas_after = STATION_WEATHER_BETAS_AFTER.get(
        selected_område,
        {"Temperatur": 0.0, "Vind": 0.0, "Nedbør": 0.0},
    )
    weather_betas_before = default_weather_betas_before.copy()
    weather_betas_after = default_weather_betas_after.copy()

    if weather_control_enabled:
        st.caption(f"Stasjonskoeffisienter for {selected_område}. Du kan overstyre manuelt under.")
        selected_weather_controls = st.multiselect(
            "Aktive kontroller",
            ["Temperatur", "Vind", "Nedbør"],
            default=["Temperatur", "Vind", "Nedbør"],
            key="selected_weather_controls",
        )

        beta_col_before, beta_col_after = st.columns(2)

        with beta_col_before:
            st.caption("Før Norgespris")
            weather_betas_before["Temperatur"] = st.number_input(
                "Beta temperatur",
                value=float(default_weather_betas_before["Temperatur"]),
                format="%.6f",
                key=f"beta_temperature_before_{selected_område}",
            )
            weather_betas_before["Vind"] = st.number_input(
                "Beta vind",
                value=float(default_weather_betas_before["Vind"]),
                format="%.6f",
                key=f"beta_wind_before_{selected_område}",
            )
            weather_betas_before["Nedbør"] = st.number_input(
                "Beta nedbør",
                value=float(default_weather_betas_before["Nedbør"]),
                format="%.6f",
                key=f"beta_precipitation_before_{selected_område}",
            )

        with beta_col_after:
            st.caption("Etter Norgespris")
            weather_betas_after["Temperatur"] = st.number_input(
                "Beta temperatur",
                value=float(default_weather_betas_after["Temperatur"]),
                format="%.6f",
                key=f"beta_temperature_after_{selected_område}",
            )
            weather_betas_after["Vind"] = st.number_input(
                "Beta vind",
                value=float(default_weather_betas_after["Vind"]),
                format="%.6f",
                key=f"beta_wind_after_{selected_område}",
            )
            weather_betas_after["Nedbør"] = st.number_input(
                "Beta nedbør",
                value=float(default_weather_betas_after["Nedbør"]),
                format="%.6f",
                key=f"beta_precipitation_after_{selected_område}",
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
        st.subheader(f"Gjennomsnittlig døgnprofil - {selected_område.title()}", anchor=False)
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
                    before_betas=weather_betas_before,
                    after_betas=weather_betas_after,
                    controls=selected_weather_controls,
                )
            elif selected_weather_controls and weather_cov_df.empty:
                st.info("Fant ikke værdata for valgt filter, viser ukontrollert graf.")

        pivot_df = plot_df.pivot(index='hour', columns='season_label', values='avg_forbruk')

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
            color=alt.Color(
                "season_label:N",
                title="Sesong",
                scale=alt.Scale(
                    domain=["Før Norgespris", "Etter Norgespris"],
                    range=["#FFBBFC", "#7D283D"],
                ),
            ),
            tooltip=[
                    alt.Tooltip("hour:O", title="Time"),
                    alt.Tooltip("avg_forbruk:Q", title="Forbruk (kWh)", format=".2f"),
                    alt.Tooltip("season_label:N", title="Sesong")
]
        ).properties(height=400)

        st.altair_chart(chart, use_container_width=True)

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

        stats_col_left, stats_col_right = st.columns([2.6, 1.4])

        with stats_col_left:
            with st.container(border=False, key="consumption_box"):
                st.markdown("**Gjennomsnittlig forbruk**")

                col_stats1, col_stats2, col_stats3 = st.columns(3)

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

        with stats_col_right:
            with st.container(border=False, key="norgespris_box"):
                st.markdown("**Norgespris-brukere**")
                st.metric(
                    f"Brukere i {selected_område}",
                    f"{norgespris_users} av {total_users}"
                )

        if not temp_df.empty:
            st.subheader("Gjennomsnittlig temperatur (°C)", anchor=False)
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
                color=alt.Color(
                    "season_label:N",
                    title="Sesong",
                    scale=alt.Scale(
                        domain=["Før Norgespris", "Etter Norgespris"],
                        range=["#FFBBFC", "#7D283D"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("hour:O", title="Time"),
                    alt.Tooltip("avg_temperature:Q", title="Temperatur (°C)", format=".2f"),
                    alt.Tooltip("season_label:N", title="Sesong")
                ]
            ).properties(height=400)

            st.altair_chart(chart_temp, use_container_width=True)


            temp_stats_col_left, temp_stats_col_right = st.columns([2.6, 1.4])

            with temp_stats_col_left:
                st.markdown("**Gjennomsnittlig temperatur**")
                temp_col1, temp_col2, temp_col3 = st.columns(3)
                with temp_col1:
                    if temp_before_avg is not None:
                        st.metric("Før Norgespris", f"{temp_before_avg:.2f} °C")
                with temp_col2:
                    if temp_after_avg is not None:
                        st.metric("Etter Norgespris", f"{temp_after_avg:.2f} °C")
                with temp_col3:
                    if temp_before_avg is not None and temp_after_avg is not None:
                        temp_diff = temp_after_avg - temp_before_avg
                        st.metric("Differanse (Etter - Før)", f"{temp_diff:+.2f} °C")
                    else:
                        st.metric("Differanse (Etter - Før)", "N/A")
        
            
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