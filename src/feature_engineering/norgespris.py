import pandas as pd


def clean_norgespris_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rydder og standardiserer norgespris-data
    """

    # Gi riktige kolonnenavn
    df = df.rename(columns={
        "TransformerStation": "transformer_station",
        "FromDate": "timestamp",
        "CountTotalMeteringPoint": "count_total"
    })

    # Parse timestamp 
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Fjern timezone
    df["timestamp"] = df["timestamp"].dt.tz_convert(None)

    # Sørg for riktig dtype eksplisitt
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")

    # Fjern komma i tall
    df["count_total"] = (
        df["count_total"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(int)
    )

    return df


def split_by_station(df: pd.DataFrame) -> dict:
    """
    Splitter dataframe i en per trafostasjon
    """
    station_dfs = {}

    for station, group in df.groupby("transformer_station"):
        station_name = station.replace(" ", "_").lower()
        station_dfs[station_name] = group.sort_values("timestamp")

    return station_dfs