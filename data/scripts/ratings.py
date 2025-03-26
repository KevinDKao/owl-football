from typing import Optional, Dict, Any
import requests
import pandas as pd
from credentials import API_KEY

BASE_URL = "https://apinext.collegefootballdata.com"


def get_sp_ratings(year: int, team: str) -> pd.DataFrame:
    """Retrieves SP+ ratings for a given year or school"""
    endpoint = f"{BASE_URL}/ratings/sp"

    params = {"year": year, "team": team}
    params = {k: v for k, v in params.items() if v is not None}

    if year is None and team is None:
        raise ValueError("Either year or team must be specified")

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_sp_conference_ratings(
    year: Optional[int] = None, conference: Optional[str] = None
) -> pd.DataFrame:
    """Retrieves aggregated historical conference SP+ data"""
    endpoint = f"{BASE_URL}/ratings/sp/conferences"

    params = {"year": year, "conference": conference}
    params = {k: v for k, v in params.items() if v is not None}

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_fpi_ratings(
    year: int, team: str, conference: Optional[str] = None
) -> pd.DataFrame:
    """Retrieves historical FPI ratings"""
    endpoint = f"{BASE_URL}/ratings/fpi"

    params = {"year": year, "team": team, "conference": conference}
    params = {k: v for k, v in params.items() if v is not None}

    if year is None and team is None:
        raise ValueError("Either year or team must be specified")

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_elo_ratings(
    year: Optional[int] = None,
    week: Optional[int] = None,
    season_type: Optional[str] = None,
    team: Optional[str] = None,
    conference: Optional[str] = None,
) -> pd.DataFrame:
    """Retrieves historical Elo ratings"""
    endpoint = f"{BASE_URL}/ratings/elo"

    params = {
        "year": year,
        "week": week,
        "seasonType": season_type,
        "team": team,
        "conference": conference,
    }
    params = {k: v for k, v in params.items() if v is not None}

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())
