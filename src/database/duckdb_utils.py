from pathlib import Path
import io
import os
import re
import tempfile

import duckdb
import pandas as pd
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Last inn .env lokalt (cloud bruker miljøvariabler/Streamlit secrets)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
DUCKDB_CONTAINER = os.getenv("DUCKDB_CONTAINER", "duckdb")
DUCKDB_BLOB_NAME = os.getenv("DUCKDB_BLOB_NAME", "stromdata.duckdb")
ATTACHED_DB_ALIAS = "blobdb"
LOCAL_DB_PATH = Path(tempfile.gettempdir()) / "stromdata.duckdb"

# Tabeller vi bruker i appen. Disse kvalifiseres automatisk mot attached blob-db.
KNOWN_TABLES = [
    "forbruksdata_breive",
    "forbruksdata_frikstad",
    "forbruksdata_hartevatn",
    "forbruksdata_timenes",
    "norgespris_breive",
    "norgespris_frikstad",
    "norgespris_hartevatn",
    "norgespris_timenes",
    "værdata_breive",
    "værdata_frikstad",
    "værdata_hartevatn",
    "værdata_timenes",
]


def _validate_config():
    if not AZURE_CONNECTION_STRING:
        raise ValueError(
            "AZURE_STORAGE_CONNECTION_STRING mangler. Sett den i .env lokalt eller i Streamlit secrets."
        )


def _qualify_known_tables(query: str) -> str:
    """
    Kvalifiserer tabellnavn mot attached blob-db slik at eksisterende SQL i appen fortsatt virker.
    """
    transformed = query
    for table in KNOWN_TABLES:
        pattern = rf"\b{re.escape(table)}\b"
        replacement = f"{ATTACHED_DB_ALIAS}.main.{table}"
        transformed = re.sub(pattern, replacement, transformed)
    return transformed


def _download_blob_duckdb_if_missing(force: bool = False) -> Path:
    """
    Laster ned blob-lagret DuckDB til midlertidig lokal fil ved behov.
    """
    _validate_config()
    if LOCAL_DB_PATH.exists() and not force:
        return LOCAL_DB_PATH

    blob_service_client = get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(
        container=DUCKDB_CONTAINER,
        blob=DUCKDB_BLOB_NAME,
    )

    LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_DB_PATH, "wb") as f:
        f.write(blob_client.download_blob().readall())

    return LOCAL_DB_PATH


def _get_remote_connection() -> duckdb.DuckDBPyConnection:
    """
    Oppretter en in-memory DuckDB-tilkobling og attacher nedlastet db-fil read-only.
    """
    db_path = _download_blob_duckdb_if_missing()
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{db_path.as_posix()}' AS {ATTACHED_DB_ALIAS} (READ_ONLY)")
    return con


def get_connection(read_only: bool = True):
    """
    Beholdt for kompatibilitet. Returnerer alltid en read-only in-memory tilkobling mot blob-db.
    """
    _ = read_only
    return _get_remote_connection()


def read_table(table_name: str) -> pd.DataFrame:
    query = f"SELECT * FROM {table_name}"
    return run_query(query)


def run_query(query: str) -> pd.DataFrame:
    con = _get_remote_connection()
    try:
        sql = _qualify_known_tables(query)
        return con.execute(sql).df()
    finally:
        con.close()


# ==================== Azure Blob Storage helper functions ====================

def get_blob_service_client():
    """Returnerer Azure Blob Service Client."""
    _validate_config()
    return BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)


def download_from_azure(container_name: str, blob_name: str) -> pd.DataFrame:
    """
    Laster ned en fil fra Azure Blob Storage og returnerer som pandas DataFrame.
    Støtter CSV og Parquet.
    """
    blob_service_client = get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    blob_data = blob_client.download_blob().readall()

    if blob_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(blob_data))
    if blob_name.endswith(".parquet") or blob_name.endswith(".pq"):
        return pd.read_parquet(io.BytesIO(blob_data))

    raise ValueError(f"Ukjent filformat: {blob_name}. Støttes CSV og Parquet.")


def create_or_replace_table(df: pd.DataFrame, table_name: str):
    """
    Ikke støttet i blob-direct modus (read-only database).
    """
    _ = (df, table_name)
    raise NotImplementedError("create_or_replace_table er ikke støttet når databasen brukes direkte fra Blob Storage.")


def load_blob_to_duckdb(container_name: str, blob_name: str, table_name: str, replace: bool = True):
    """
    Ikke støttet i blob-direct modus (read-only database).
    """
    _ = (container_name, blob_name, table_name, replace)
    raise NotImplementedError("load_blob_to_duckdb er ikke støttet når databasen brukes direkte fra Blob Storage.")


def load_multiple_blobs_to_duckdb(blob_config: dict):
    """
    Ikke støttet i blob-direct modus (read-only database).
    """
    _ = blob_config
    raise NotImplementedError("load_multiple_blobs_to_duckdb er ikke støttet når databasen brukes direkte fra Blob Storage.")


def list_blobs(container_name: str) -> list:
    """Lister alle filer i en Azure Blob Storage container."""
    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client(container_name)
    return [blob.name for blob in container_client.list_blobs()]
