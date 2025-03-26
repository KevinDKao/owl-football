from typing import Optional, Dict, Any
import requests
import pandas as pd
from credentials import API_KEY

BASE_URL = "https://apinext.collegefootballdata.com"


def search_players(
    search_term: str,
    year: Optional[int] = None,
    team: Optional[str] = None,
    position: Optional[str] = None,
) -> pd.DataFrame:
    """Search for players (limit 100 results)"""
    endpoint = f"{BASE_URL}/player/search"

    params = {
        "searchTerm": search_term,
        "year": year,
        "team": team,
        "position": position,
    }
    params = {k: v for k, v in params.items() if v is not None}

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_player_usage(
    year: int,
    conference: Optional[str] = None,
    position: Optional[str] = None,
    team: Optional[str] = None,
    player_id: Optional[int] = None,
    exclude_garbage_time: Optional[bool] = False,
) -> pd.DataFrame:
    """Retrieves player usage data for a given season"""
    endpoint = f"{BASE_URL}/player/usage"

    params = {
        "year": year,
        "conference": conference,
        "position": position,
        "team": team,
        "playerId": player_id,
        "excludeGarbageTime": exclude_garbage_time,
    }
    params = {k: v for k, v in params.items() if v is not None}

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_player_returning(
    year: int, team: str, conference: Optional[str] = None
) -> pd.DataFrame:
    """Retrieves returning production data. Either a year or team filter must be specified"""
    endpoint = f"{BASE_URL}/player/returning"

    params = {"year": year, "team": team, "conference": conference}
    params = {k: v for k, v in params.items() if v is not None}

    if year is None and team is None:
        raise ValueError("Either year or team must be specified")

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())
