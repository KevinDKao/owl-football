import pandas as pd
from app.utils import convert_height


def load_data():
    """
    Load and preprocess the draft predictions data
    """
    # Load the draft predictions data
    df = pd.read_csv(
        "notebooks/drafting_models/current_season/output/2025_draft_predictions_full.csv"
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
