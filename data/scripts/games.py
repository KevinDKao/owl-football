from typing import Optional, List, Dict, Any
import requests
from IPython.display import display
import pandas as pd
from credentials import API_KEY

BASE_URL = "https://apinext.collegefootballdata.com"


def get_games(
    year: int,
    week: Optional[int] = None,
    session_type: Optional[str] = "regular",
    classification: Optional[str] = None,
    team: Optional[str] = None,
    home: Optional[str] = None,
    away: Optional[str] = None,
    conference: Optional[str] = None,
    game_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Retrieves historical game data"""
    endpoint = f"{BASE_URL}/games"

    params = {
        "year": year,
        "week": week,
        "seasonType": session_type,
        "classification": classification,
        "team": team,
        "home": home,
        "away": away,
        "conference": conference,
        "id": game_id,
    }
    params = {k: v for k, v in params.items() if v is not None}

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_games_team_box(
    year: Optional[int] = None,
    week: Optional[int] = None,
    team: Optional[str] = None,
    conference: Optional[str] = None,
    classification: Optional[str] = None,
    season_type: Optional[str] = None,
    game_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Retrieves team box score statistics"""
    endpoint = f"{BASE_URL}/games/teams"

    params = {
        "year": year,
        "week": week,
        "team": team,
        "conference": conference,
        "classification": classification,
        "seasonType": season_type,
        "id": game_id,
    }
    params = {k: v for k, v in params.items() if v is not None}

    if game_id is None and not any([week, team, conference]):
        raise ValueError("Either week, team, or conference is required")

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return response.json()


def get_player_stats(
    year: Optional[int] = None,
    week: Optional[int] = None,
    team: Optional[str] = None,
    conference: Optional[str] = None,
    classification: Optional[str] = None,
    season_type: Optional[str] = None,
    category: Optional[str] = None,
    game_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Retrieves player box score statistics"""
    endpoint = f"{BASE_URL}/games/players"

    params = {
        "year": year,
        "week": week,
        "team": team,
        "conference": conference,
        "classification": classification,
        "seasonType": season_type,
        "category": category,
        "id": game_id,
    }
    params = {k: v for k, v in params.items() if v is not None}

    if game_id is None and not any([week, team, conference]):
        raise ValueError("Either week, team, or conference is required")

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_game_weather(
    year: int,
    season_type: Optional[str] = None,
    week: Optional[int] = None,
    team: Optional[str] = None,
    conference: Optional[str] = None,
    classification: Optional[str] = None,
    game_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Retrieves historical and future weather data"""
    endpoint = f"{BASE_URL}/games/weather"

    params = {
        "year": year,
        "seasonType": season_type,
        "week": week,
        "team": team,
        "conference": conference,
        "classification": classification,
        "gameId": game_id,
    }
    params = {k: v for k, v in params.items() if v is not None}

    if game_id is None and year is None:
        raise ValueError("Either year or game_id must be specified")

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_team_records(
    year: Optional[int] = None,
    team: Optional[str] = None,
    conference: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieves historical team records"""
    endpoint = f"{BASE_URL}/records"

    params = {"year": year, "team": team, "conference": conference}
    params = {k: v for k, v in params.items() if v is not None}

    if year is None and team is None:
        raise ValueError("Either year or team must be specified")

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())
