from typing import Optional, Dict, Any
import requests
from credentials import API_KEY
import pandas as pd

BASE_URL = "https://apinext.collegefootballdata.com"


def get_draft_picks(
    year: Optional[int] = None,
    team: Optional[str] = None,
    school: Optional[str] = None,
    conference: Optional[str] = None,
    position: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieves NFL draft pick data"""
    endpoint = f"{BASE_URL}/draft/picks"

    params = {
        "year": year,
        "team": team,
        "school": school,
        "conference": conference,
        "position": position,
    }
    params = {k: v for k, v in params.items() if v is not None}

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, params=params, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_draft_positions() -> Dict[str, Any]:
    """Retrieves valid NFL draft positions"""
    endpoint = f"{BASE_URL}/draft/positions"

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_draft_teams() -> Dict[str, Any]:
    """Retrieves NFL teams that have participated in the draft"""
    endpoint = f"{BASE_URL}/draft/teams"

    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    response = requests.get(endpoint, headers=headers)
    response.raise_for_status()
    return pd.DataFrame(response.json())
