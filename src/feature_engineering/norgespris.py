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

def fill_missing_days_per_station(
    station_dfs: dict,
    periods: list[tuple[str, str]]
) -> dict:
    filled_station_dfs = {}

    for station_name, df in station_dfs.items():
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        # 🔥 1. Lag komplett dato-range basert på hele datasettet
        full_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq="D"
        )

        df = df.reindex(full_range)

        # 🔥 2. Fyll hull globalt (viktig!)
        df["count_total"] = df["count_total"].ffill()
        df["transformer_station"] = df["transformer_station"].ffill()

        # 🔥 3. Filtrer til periodene etterpå
        period_dfs = []
        for start, end in periods:
            df_period = df.loc[start:end]
            period_dfs.append(df_period)

        df_filled = pd.concat(period_dfs)

        df_filled = df_filled.reset_index().rename(columns={"index": "timestamp"})

        filled_station_dfs[station_name] = df_filled

    return filled_station_dfs