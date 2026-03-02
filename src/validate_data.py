import pandas as pd

def is_data_complete(df: pd.DataFrame, date_column=None, freq="H"):
    """
    Returnerer:
    True  -> hvis ingen missing values og ingen manglende tidssteg
    False -> hvis noe mangler
    """

    # 1. Sjekk missing values
    if df.isnull().values.any():
        return False

    # 2. Sjekk tidsserie-hull (hvis dato-kolonne er gitt)
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)

        full_range = pd.date_range(
            start=df[date_column].min(),
            end=df[date_column].max(),
            freq=freq
        )

        if len(full_range) != df[date_column].nunique():
            return False

    return True