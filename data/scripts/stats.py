from typing import Optional, List, Dict, Any
import requests
import pandas as pd
from credentials import API_KEY

BASE_URL = "https://apinext.collegefootballdata.com"


def get_player_season_stats(
    year: int,
    conference: Optional[str] = None,
    team: Optional[str] = None,
    start_week: Optional[int] = None,
    end_week: Optional[int] = None,
    season_type: Optional[str] = None,
    category: Optional[str] = None,
) -> pd.DataFrame:
    """Retrieves aggregated player statistics for a given season"""
    endpoint = f"{BASE_URL}/stats/player/season"

    params = {
        "year": year,
        "conference": conference,
        "team": team,
        "startWeek": start_week,
        "endWeek": end_week,
        "seasonType": season_type,
        "category": category,
    }
    params = {k: v for k, v in params.items() if v is not None}

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_team_season_stats(
    year: Optional[int] = None,
    team: Optional[str] = None,
    conference: Optional[str] = None,
    start_week: Optional[int] = None,
    end_week: Optional[int] = None,
) -> pd.DataFrame:
    """Retrieves aggregated team season statistics"""
    endpoint = f"{BASE_URL}/stats/season"

    params = {
        "year": year,
        "team": team,
        "conference": conference,
        "startWeek": start_week,
        "endWeek": end_week,
    }
    params = {k: v for k, v in params.items() if v is not None}

    if year is None and team is None:
        raise ValueError("Either year or team must be specified")

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_stat_categories() -> List[str]:
    """Gets team statistical categories"""
    endpoint = f"{BASE_URL}/stats/categories"

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_advanced_season_stats(
    year: Optional[int] = None,
    team: Optional[str] = None,
    exclude_garbage_time: Optional[bool] = None,
    start_week: Optional[int] = None,
    end_week: Optional[int] = None,
) -> Dict[str, Any]:
    """Retrieves advanced season statistics for teams"""
    endpoint = f"{BASE_URL}/stats/season/advanced"

    params = {
        "year": year,
        "team": team,
        "excludeGarbageTime": exclude_garbage_time,
        "startWeek": start_week,
        "endWeek": end_week,
    }
    params = {k: v for k, v in params.items() if v is not None}

    if year is None and team is None:
        raise ValueError("Either year or team must be specified")

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_game_advanced_stats(
    year: Optional[int] = None,
    team: Optional[str] = None,
    week: Optional[int] = None,
    opponent: Optional[str] = None,
    exclude_garbage_time: Optional[bool] = None,
    season_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieves advanced statistics aggregated by game"""
    endpoint = f"{BASE_URL}/stats/game/advanced"

    params = {
        "year": year,
        "team": team,
        "week": week,
        "opponent": opponent,
        "excludeGarbageTime": exclude_garbage_time,
        "seasonType": season_type,
    }
    params = {k: v for k, v in params.items() if v is not None}

    if year is None and team is None:
        raise ValueError("Either year or team must be specified")

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())
