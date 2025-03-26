# 📊 Data Scripts

This directory contains scripts and notebooks for collecting, processing, and analyzing college football and NFL draft data using the College Football Data API.

## 🔑 Setup

Before using these scripts, you need to:

1. Obtain an API key from [College Football Data API](https://apinext.collegefootballdata.com)
2. Add your API key to the `credentials.py` file:
   ```python
   API_KEY = 'your_api_key_here'
   ```

## 📓 Key Notebooks

### [examples.ipynb](examples.ipynb)
A comprehensive guide demonstrating how to use the various data collection functions. This notebook shows examples of:
- Retrieving game data
- Getting team box scores
- Fetching player statistics
- Accessing game weather information
- Obtaining team records

### [draft_data.ipynb](draft_data.ipynb)
Collects and processes NFL draft data, including:
- Player season statistics
- Draft pick information
- Merging player performance with draft outcomes

### [boxScoreLogReg.ipynb](boxScoreLogReg.ipynb)
Analyzes box score data using logistic regression models.

## 🐍 Python Modules

The directory includes several Python modules that provide functions for accessing different aspects of college football data:

### Game Data
- **[games.py](games.py)**: Functions for retrieving game data, team box scores, player game statistics, game weather, and team records.

### Player Data
- **[players.py](players.py)**: Functions for searching players, getting player usage data, and retrieving returning player information.
- **[stats.py](stats.py)**: Functions for retrieving player and team statistics.

### Team Data
- **[teams.py](teams.py)**: Functions for accessing team information.
- **[ratings.py](ratings.py)**: Functions for retrieving team ratings from various systems (SP+, FPI, etc.).
- **[rankings.py](rankings.py)**: Functions for accessing team rankings.

### Recruiting Data
- **[recruiting.py](recruiting.py)**: Functions for retrieving recruiting data and rankings.

### Draft Data
- **[draft.py](draft.py)**: Functions for accessing NFL draft data.

### Other Data
- **[conferences.py](conferences.py)**: Functions for accessing conference information.
- **[venues.py](venues.py)**: Functions for retrieving venue/stadium data.
- **[plays.py](plays.py)**: Functions for accessing play-by-play data.
- **[drives.py](drives.py)**: Functions for retrieving drive data.
- **[coaches.py](coaches.py)**: Functions for accessing coach information.
- **[betting.py](betting.py)**: Functions for retrieving betting lines and odds.
- **[metrics.py](metrics.py)** and **[adjusted_metrics.py](adjusted_metrics.py)**: Functions for accessing various football metrics.

## 🚀 Getting Started

1. Make sure you have set up your API key in `credentials.py`
2. Start with the [examples.ipynb](examples.ipynb) notebook to understand how to use the various functions
3. Use the individual Python modules for more specific data collection needs
4. For NFL draft analysis, refer to [draft_data.ipynb](draft_data.ipynb)

## 📤 Data Output

By default, most functions return pandas DataFrames for easy data manipulation and analysis. You can save these DataFrames to various formats:

```python
# Example: Save to CSV
df = get_games(year=2023)
df.to_csv('../data_files/games_2023.csv', index=False)

# Example: Save to Feather (faster I/O)
df.to_feather('../data_files/games_2023.feather')
```

## 📝 Notes

- Some API endpoints have rate limits, so consider adding delays between requests if fetching large amounts of data
- Empty Python files (0 bytes) are placeholders for future implementation
