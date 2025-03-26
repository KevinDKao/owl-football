from typing import Optional, Dict, Any
import requests
import pandas as pd
from credentials import API_KEY

BASE_URL = "https://apinext.collegefootballdata.com"


def get_recruiting_players(
    year: Optional[int] = None,
    team: Optional[str] = None,
    position: Optional[str] = None,
    state: Optional[str] = None,
    classification: Optional[str] = "HighSchool",
) -> pd.DataFrame:
    """Retrieves player recruiting rankings"""
    endpoint = f"{BASE_URL}/recruiting/players"

    params = {
        "year": year,
        "team": team,
        "position": position,
        "state": state,
        "classification": classification,
    }
    params = {k: v for k, v in params.items() if v is not None}

    if year is None and team is None:
        raise ValueError("Either year or team must be specified")

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_recruiting_teams(
    year: Optional[int] = None, team: Optional[str] = None
) -> pd.DataFrame:
    """Retrieves team recruiting rankings"""
    endpoint = f"{BASE_URL}/recruiting/teams"

    params = {"year": year, "team": team}
    params = {k: v for k, v in params.items() if v is not None}

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_recruiting_groups(
    team: Optional[str] = None,
    conference: Optional[str] = None,
    recruit_type: Optional[str] = "HighSchool",
    start_year: Optional[int] = 2000,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """Retrieves aggregated recruiting statistics by team and position grouping"""
    endpoint = f"{BASE_URL}/recruiting/groups"

    params = {
        "team": team,
        "conference": conference,
        "recruitType": recruit_type,
        "startYear": start_year,
        "endYear": end_year,
    }
    params = {k: v for k, v in params.items() if v is not None}

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())
