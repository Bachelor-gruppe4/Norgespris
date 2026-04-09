from pathlib import Path
import sys
import os
import calendar
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

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

MONTH_NAMES = {11: "nov", 12: "des", 1: "jan"}

def format_month_day(month, day):
    return f"{day:02d}.{MONTH_NAMES[month]}"

def build_month_day_options():
    options = []
    for month in [11, 12]:
        days = calendar.monthrange(2024, month)[1]
        for day in range(1, days + 1):
            options.append((month, day))
    for day in range(1, calendar.monthrange(2025, 1)[1] + 1):
        options.append((1, day))
    return options

MONTH_DAY_OPTIONS = build_month_day_options()
MONTH_DAY_LABELS = [format_month_day(month, day) for month, day in MONTH_DAY_OPTIONS]
MONTH_STR_TO_MONTH = {v: k for k, v in MONTH_NAMES.items()}

def parse_month_day(label):
    day_str, month_str = label.split('.')
    return int(day_str), MONTH_STR_TO_MONTH[month_str]

def month_day_index(label):
    day, month = parse_month_day(label)
    if month == 11:
        return day - 1
    if month == 12:
        return 30 + day
    return 61 + day

def season_date(label, season_start_year):
    day, month = parse_month_day(label)
    year = season_start_year if month in (11, 12) else season_start_year + 1
    return datetime(year, month, day).date()

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
def get_season_comparison(område="breive", month_day="01.nov", consumption_codes=None):
    """
    Henter gjennomsnittlig døgnprofil for én utvalgt dag i to sesonger uten å bruke norgespris-kolonnen.
    """
    table_name = f"forbruksdata_{område.lower()}"
    timestamp_expr = "CAST(timestamp AS TIMESTAMP)"

    season1_date = season_date(month_day, 2024)
    season2_date = season_date(month_day, 2025)

    code_filter = ""
    if consumption_codes is not None:
        codes_str = ", ".join(str(code) for code in consumption_codes)
        code_filter = f"AND consumption_code IN ({codes_str})"

    query = f"""
    SELECT hour, season_label, AVG(value_kwh) AS avg_forbruk
    FROM (
        SELECT
            HOUR({timestamp_expr}) AS hour,
            'Før Norgespris' AS season_label,
            value_kwh
        FROM {table_name}
        WHERE DATE({timestamp_expr}) = '{season1_date}'
          {code_filter}
        UNION ALL
        SELECT
            HOUR({timestamp_expr}) AS hour,
            'Etter Norgespris' AS season_label,
            value_kwh
        FROM {table_name}
        WHERE DATE({timestamp_expr}) = '{season2_date}'
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
def get_norgespris_user_count(område="breive", month_day="01.nov"):
    """
    Henter antall brukere med Norgespris for en gitt dag i begge sesonger.
    """
    table_name = f"norgespris_{område.lower()}"
    
    season1_date = season_date(month_day, 2024)
    season2_date = season_date(month_day, 2025)
    
    query = f"""
    SELECT season_label, count_total
    FROM (
        SELECT
            'Før Norgespris' AS season_label,
            count_total
        FROM {table_name}
        WHERE timestamp = '{season1_date}'
        UNION ALL
        SELECT
            'Etter Norgespris' AS season_label,
            count_total
        FROM {table_name}
        WHERE timestamp = '{season2_date}'
    )
    """
    
    try:
        df = run_query(query)
        return df
    except Exception as e:
        st.error(f"Kunne ikke hente brukerantal: {e}")
        return pd.DataFrame()


@st.cache_data
def get_total_unique_users(område="breive", month_day="01.nov"):
    """
    Henter totalt antall unike brukere i forbruksdata for valgt dag etter Norgespris.
    """
    table_name = f"forbruksdata_{område.lower()}"
    timestamp_expr = "CAST(timestamp AS TIMESTAMP)"
    season2_date = season_date(month_day, 2025)

    query = f"""
    SELECT COUNT(DISTINCT metering_point_anonymous) AS total_unique_users
    FROM {table_name}
    WHERE DATE({timestamp_expr}) = '{season2_date}'
    """

    try:
        df = run_query(query)
        if not df.empty and "total_unique_users" in df.columns:
            return int(df.iloc[0]["total_unique_users"])
        return 0
    except Exception as e:
        st.error(f"Kunne ikke hente totalt antall unike brukere: {e}")
        return 0


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
    
    # Dato-filter uten år
    selected_day = st.selectbox(
        "Velg dag (dag og måned)",
        MONTH_DAY_LABELS,
        index=0,
        key="selected_month_day"
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

    selected_metric = "avg_forbruk"

with col_graph:
    # Hent data basert på filtre
    with st.spinner("Henter data..."):
        df = get_season_comparison(
            område=selected_område,
            month_day=selected_day,
            consumption_codes=selected_consumption_codes
        )
    
    if not df.empty:
        st.markdown(f"### Gjennomsnittlig døgnprofil - {selected_område.title()}")
        st.write(f"Sammenligner dagen {selected_day} som \"Før Norgespris\" (2024-2025) og \"Etter Norgespris\" (2025-2026).")
        
        df = df.sort_values('hour')
        pivot_df = df.pivot(index='hour', columns='season_label', values='avg_forbruk')
        
        if chart_type == "Linje":
            st.line_chart(pivot_df)
        elif chart_type == "Stolpe":
            st.bar_chart(pivot_df)
        elif chart_type == "Område":
            st.area_chart(pivot_df)
        
        # Vis statistikk
        user_count_df = get_norgespris_user_count(
            område=selected_område,
            month_day=selected_day
        )
        total_unique_users = get_total_unique_users(
            område=selected_område,
            month_day=selected_day
        )
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("Gjennomsnitt", f"{pivot_df.mean().mean():.2f} kWh")
        with col_stats2:
            st.metric("Maksimum", f"{pivot_df.max().max():.2f} kWh")
        with col_stats3:
            if not user_count_df.empty:
                etter_row = user_count_df[user_count_df["season_label"] == "Etter Norgespris"]
                if not etter_row.empty:
                    norgespris_users = int(etter_row["count_total"].iloc[0])
                else:
                    norgespris_users = int(user_count_df["count_total"].iloc[-1])

                if total_unique_users > 0:
                    st.metric(
                        f"Antall brukere med Norgespris i {selected_område}",
                        f"{norgespris_users} av {total_unique_users}"
                    )
                else:
                    st.metric(
                        f"Antall brukere med Norgespris i {selected_område}",
                        f"{norgespris_users}"
                    )
            else:
                st.metric(f"Antall brukere med Norgespris i {selected_område}", "N/A")
            
    else:
        st.warning("Ingen data funnet for valgte filtre. Prøv å justere dato-intervallet eller området.")
        
        # Vis eksempel på tilgjengelige data
        st.markdown("**Tilgjengelige områder:**")
        for område in område_options:
            try:
                test_df = get_forbruksdata(område, limit=1)
                if not test_df.empty:
                    st.write(f"✅ {område.title()}: Data tilgjengelig")
                else:
                    st.write(f"❌ {område.title()}: Ingen data")
            except:
                st.write(f"❌ {område.title()}: Feil ved tilkobling")


st.markdown("---")


# --- PROSJEKTSEKSJON (ID FOR SCROLL) ---
st.markdown('<div id="prosjektet"></div>', unsafe_allow_html=True)

st.markdown("## Prosjektet")

col_img, col_text = st.columns([1, 3])

with col_img:
    st.image("https://via.placeholder.com/300")

with col_text:
    st.write("Om prosjektgruppen og samarbeid...")

st.markdown("---")

st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px;'>Samarbeidspartnere</div>",
    unsafe_allow_html=True
)

# 5 kolonner: tom - logo - logo - logo - tom
col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])

with col2:
    st.image(get_asset_path("images/file.svg"), width=60)

with col3:
    st.image(get_asset_path("images/Capgemini_201x_logo.svg"), width=100)

with col4:
    st.image(get_asset_path("images/uia.svg"), width=60)