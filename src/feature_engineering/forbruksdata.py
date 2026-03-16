import pandas as pd
import holidays


def convert_timestamp(df):
    """Konverter timestamp til datetime"""
    df["TimestampUtc"] = pd.to_datetime(df["TimestampUtc"])
    return df


def add_time_features(df):
    """Legger til time features"""

    df["hour"] = df["TimestampUtc"].dt.hour
    df["weekday"] = df["TimestampUtc"].dt.weekday
    df["month"] = df["TimestampUtc"].dt.month

    return df


def add_weekend_feature(df):
    """Helg eller ikke"""

    df["is_weekend"] = df["weekday"] >= 5

    return df


def add_holiday_feature(df):
    """Norske helligdager + julaften og nyttårsaften"""

    norwegian_holidays = holidays.Norway()
    dates = df["TimestampUtc"].dt.date

    # vanlige helligdager
    is_holiday = dates.isin(norwegian_holidays)

    # julaften og nyttårsaften
    is_christmas_eve = (
        (df["TimestampUtc"].dt.month == 12) &
        (df["TimestampUtc"].dt.day == 24)
    )
    is_new_years_eve = (
        (df["TimestampUtc"].dt.month == 12) &
        (df["TimestampUtc"].dt.day == 31)
    )

    df["is_holiday"] = is_holiday | is_christmas_eve | is_new_years_eve

    return df


def add_day_night_feature(df):
    """Dag eller natt"""

    df["day_night"] = df["hour"].apply(
        lambda x: "day" if 6 <= x < 22 else "night"
    )

    return df


def create_consumption_features(df):
    """
    Full feature engineering pipeline
    """

    df = convert_timestamp(df)
    df = add_time_features(df)
    df = add_weekend_feature(df)
    df = add_holiday_feature(df)
    df = add_day_night_feature(df)

    return df