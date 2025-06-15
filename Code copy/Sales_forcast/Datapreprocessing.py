
import pandas as pd

def cleanData(df):
    """Remove duplicates and null values."""
    initial_rows = len(df)
    df = df.drop_duplicates()
    after_dedup = len(df)
    df = df.dropna()
    final_rows = len(df)
    log = [
        f"Initial rows: {initial_rows}",
        f"Removed duplicates: {initial_rows - after_dedup}",
        f"Removed nulls: {after_dedup - final_rows}"
    ]
    return df, log

def removeOutliersIqr(df, column):
    """Remove outliers from a specified column using the IQR method."""
    if column not in df.columns:
        return df, [f"Column '{column}' not found, skipping outlier removal."]
    initial_rows = len(df)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    final_rows = len(df)
    log = [f"Removed outliers from '{column}': {initial_rows - final_rows} rows"]
    return df, log

def preprocessData(df, column=None):
    """Apply all preprocessing steps."""
    df, log1 = cleanData(df)
    if column:
        df, log2 = removeOutliersIqr(df, column)
    else:
        log2 = ["No column specified for outlier removal."]
    return df, log1 + log2