import pandas as pd
import holidays

def rename_columns(df):
    """Endrer kolonnenavn"""
    df = df.rename(columns={
        "MeteringPointAnonymous": "metering_point_anonymous",
        "TimestampUtc": "timestamp",
        "Value": "value_kwh",
        "TransformerStation": "transformer_station",
        "ConsumptionCode": "consumption_code"
    })
    return df

def convert_timestamp(df):
    """Konverter timestamp til datetime"""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def add_time_features(df):
    """Legger til time features"""

    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month

    return df


def add_weekend_feature(df):
    """Helg eller ikke"""

    df["is_weekend"] = df["weekday"] >= 5

    return df


def add_holiday_feature(df):
    """Norske helligdager + julaften og nyttårsaften"""

    norwegian_holidays = holidays.Norway()
    dates = df["timestamp"].dt.date

    # vanlige helligdager
    is_holiday = dates.isin(norwegian_holidays)

    # julaften og nyttårsaften
    is_christmas_eve = (
        (df["timestamp"].dt.month == 12) &
        (df["timestamp"].dt.day == 24)
    )
    is_new_years_eve = (
        (df["timestamp"].dt.month == 12) &
        (df["timestamp"].dt.day == 31)
    )

    df["is_holiday"] = is_holiday | is_christmas_eve | is_new_years_eve

    return df


def add_day_night_feature(df):
    """Dag eller natt"""

    df["day_night"] = df["hour"].apply(
        lambda x: "day" if 6 <= x < 22 else "night"
    )

    return df

def add_norgespris_feature(df):
    """
    0 = før norgespris (nov 2024 - jan 2025)
    1 = etter norgespris (nov 2025 - jan 2026)
    """
    
    df["norgespris"] = pd.NA

    df.loc[
    (df["timestamp"] >= "2024-10-31") &
    (df["timestamp"] <= "2025-02-01"),
    "norgespris"
] = 0

    # sett perioder
    df.loc[
        (df["timestamp"] >= "2025-10-31") &
        (df["timestamp"] <= "2026-02-01"),
        "norgespris"
    ] = 1

    return df


def create_consumption_features(df):
    """
    Full feature engineering pipeline
    """

    df = rename_columns(df)
    df = convert_timestamp(df)
    df = add_time_features(df)
    df = add_weekend_feature(df)
    df = add_holiday_feature(df)
    df = add_day_night_feature(df)
    df = add_norgespris_feature(df)

    return df