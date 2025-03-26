# 📊 Data Directory

This directory contains all components related to data collection, processing, and storage for the College Football Analytics project. It serves as the foundation for the analysis and modeling work in the project.

## 📁 Directory Structure

```
data/
├── scripts/              # Python scripts and notebooks for data collection
│   ├── examples.ipynb    # Examples of how to use the data collection functions
│   ├── draft_data.ipynb  # NFL draft data collection and processing
│   ├── games.py          # Functions for retrieving game data
│   ├── players.py        # Functions for retrieving player data
│   ├── ratings.py        # Functions for retrieving team ratings
│   └── ...               # Other data collection modules
│
└── data_files/           # Storage for collected and processed data
    ├── (CSV files)       # Data stored in CSV format
    └── (Feather files)   # Data stored in Feather format for faster I/O
```

## 🔑 Key Components

### [scripts/](scripts/)

The `scripts` directory contains all the code needed to collect and process data from various sources:

- **API Wrappers**: Python modules that provide functions for accessing the College Football Data API
- **Data Processing**: Scripts for cleaning, transforming, and preparing data for analysis
- **Example Notebooks**: Jupyter notebooks demonstrating how to use the data collection functions

For detailed information about the scripts, see the [scripts README](scripts/readme.md).

### [data_files/](data_files/)

The `data_files` directory is where collected and processed data is stored. Data is typically saved in one of two formats:

- **CSV**: Human-readable text files that can be opened in Excel or other spreadsheet applications
- **Feather**: A binary file format optimized for fast reading and writing of pandas DataFrames

This directory is initially empty and will be populated as you run the data collection scripts.

## 🚀 Getting Started with Data Collection

### Prerequisites

Before collecting data, you need to:

1. Obtain an API key from [College Football Data API](https://apinext.collegefootballdata.com)
2. Add your API key to the `scripts/credentials.py` file:
   ```python
   API_KEY = 'your_api_key_here'
   ```

### Basic Data Collection Workflow

1. **Start with Examples**: Begin by exploring the [examples.ipynb](scripts/examples.ipynb) notebook to understand the available data and how to access it.

2. **Collect Game Data**:
   ```python
   from scripts.games import get_games
   
   # Get all FBS games from the 2023 regular season
   games_data = get_games(year=2023, classification='fbs', session_type='regular')
   
   # Save to data_files directory
   games_data.to_csv('data_files/games_2023.csv', index=False)
   ```

3. **Collect Player Data**:
   ```python
   from scripts.players import search_players
   
   # Search for players from a specific team and year
   players = search_players(search_term='', team='Alabama', year=2023)
   
   # Save to data_files directory
   players.to_csv('data_files/alabama_players_2023.csv', index=False)
   ```

4. **Collect Draft Data**: Run the [draft_data.ipynb](scripts/draft_data.ipynb) notebook to collect and process NFL draft data.

## 📈 Data Types Available

The scripts in this directory provide access to a wide range of college football data:

- **Games**: Game results, schedules, and team performance
- **Players**: Player information, statistics, and game-by-game performance
- **Teams**: Team information, historical performance, and rankings
- **Ratings**: Team ratings from various systems (SP+, FPI, etc.)
- **Recruiting**: Recruiting rankings and player recruitment information
- **Draft**: NFL draft data including pick information and player outcomes

## 🔄 Data Flow

The typical data flow in this project is:

1. **Collection**: Raw data is collected from the College Football Data API using the scripts in the `scripts` directory
2. **Processing**: Data is cleaned, transformed, and prepared for analysis
3. **Storage**: Processed data is saved to the `data_files` directory
4. **Analysis**: Data is loaded from the `data_files` directory into notebooks for analysis and modeling

## 📝 Best Practices

- **Cache API Responses**: To avoid hitting API rate limits, save API responses to files and load from those files when possible
- **Use Feather Format**: For large datasets, use the Feather format for faster I/O operations
- **Document Data Sources**: When creating new datasets, document the source and any transformations applied
- **Version Control**: Consider versioning your datasets, especially if they change over time
- **Respect API Limits**: The College Football Data API has rate limits, so add delays between requests when fetching large amounts of data

## 🔗 Related Resources

- [College Football Data API Documentation](https://apinext.collegefootballdata.com)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Analysis Notebooks](../notebooks/)
