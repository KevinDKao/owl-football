import pandas as pd
from utils import convert_height
import os


def load_data():
    """
    Load and preprocess the draft predictions data
    """
    df = pd.read_csv(
        "cache.csv"
    )

    # Clean up column names
    df.columns = [col.strip(":") for col in df.columns]

    # Clean data
    df["readable_height"] = df["HEIGHT"].apply(convert_height)
    df["position_group"] = df["position_group"].fillna("Unknown")
    df["school"] = df["SCHOOL"].fillna("Unknown")
    df["name"] = df["NAME"].fillna("Unknown")

    return df


def get_positions_and_schools(df):
    """
    Get unique positions and schools from the dataframe
    """
    positions = sorted(df["position_group"].unique())
    schools = sorted(df["school"].unique())

    return positions, schools
