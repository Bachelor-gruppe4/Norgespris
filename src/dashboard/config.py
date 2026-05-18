from pathlib import Path
import sys
import os
import streamlit as st

# Sørg for at prosjektroten er på Python import-stien, uansett hvor appen kjøres fra.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


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


BASE_DIR = Path(__file__).resolve().parent


def get_asset_path(path: str) -> Path:
    return (BASE_DIR / "../assets" / path).resolve()


SEASON_1_START = "2024-11-01"
SEASON_1_END = "2025-01-31"
SEASON_2_START = "2025-11-01"
SEASON_2_END = "2026-01-31"

STATION_WEATHER_BETAS_BEFORE = {
    "Breive": {"Temperatur": -0.014512, "Vind": 0.004102, "Nedbør": 0.011508},
    "Frikstad": {"Temperatur": -0.018790, "Vind": 0.005962, "Nedbør": 0.010317},
    "Hartevatn": {"Temperatur": -0.014199, "Vind": 0.004867, "Nedbør": 0.010142},
    "Timenes": {"Temperatur": -0.016093, "Vind": 0.007853, "Nedbør": 0.009971},
}

STATION_WEATHER_BETAS_AFTER = {
    "Breive": {"Temperatur": -0.019255, "Vind": 0.011599, "Nedbør": -0.015119},
    "Frikstad": {"Temperatur": -0.021582, "Vind": 0.008675, "Nedbør": 0.001878},
    "Hartevatn": {"Temperatur": -0.017798, "Vind": 0.008936, "Nedbør": -0.01540},
    "Timenes": {"Temperatur": -0.018753, "Vind": 0.010886, "Nedbør": 0.001263},
}

CAPGEMINI_LOGO = get_asset_path("images/Capgemini_201x_logo.svg")
A_ENERGI_LOGO = get_asset_path("images/file.svg")

STATION_TEXTS = {
    "Breive": (
        "Breive har en høy andel fritidsboliger. Regresjonsmodellen for Breive indikerer at "
        "økt utbredelse av Norgespris henger sammen med høyere strømforbruk. Når det kontrolleres for "
        "temperatur, vind, nedbør og tidsmønstre, er en økning på 10 prosentpoeng i andelen kunder "
        "med Norgespris assosiert med 0,30 % høyere forbruk. Den estimerte merbruken i perioden etter "
        "at Norgespris ble innført er 166 573 kWh, tilsvarende 3,89 % av totalforbruket."
    ),
    "Frikstad": (
        "Frikstad har et større og mer sammensatt kundegrunnlag. Regresjonsmodellen for Frikstad indikerer "
        "at økt utbredelse av Norgespris henger sammen med noe høyere strømforbruk. Når det kontrolleres for "
        "temperatur, vind, nedbør og tidsmønstre, er en økning på 10 prosentpoeng i andelen kunder "
        "med Norgespris assosiert med omtrent 0,19–0,20 % høyere forbruk. Den estimerte merbruken i perioden etter "
        "at Norgespris ble innført er omtrent 277 000–285 000 kWh, tilsvarende rundt 2,15–2,21 % av totalforbruket."
    ),
    "Hartevatn": (
        "Hartevatn har en høy andel fritidsboliger. Regresjonsmodellen for Hartevatn indikerer at økt "
        "utbredelse av Norgespris henger sammen med høyere strømforbruk. Når det kontrolleres for temperatur, "
        "vind, nedbør og tidsmønstre, er en økning på 10 prosentpoeng i andelen kunder med Norgespris assosiert "
        "med 0,32 % høyere forbruk. Den estimerte merbruken i perioden etter at Norgespris ble innført er 277 891 kWh, "
        "tilsvarende 4,31 % av totalforbruket."
    ),
    "Timenes": (
        "Regresjonsmodellen for Timenes indikerer at økt utbredelse av Norgespris henger sammen med høyere strømforbruk. "
        "Når det kontrolleres for temperatur, vind, nedbør og tidsmønstre, er en økning på 10 prosentpoeng i "
        "andelen kunder med Norgespris assosiert med 0,39 % høyere forbruk. Den estimerte merbruken i perioden etter "
        "at Norgespris ble innført er 547 247 kWh, tilsvarende 3,93 % av totalforbruket."
    ),
}
