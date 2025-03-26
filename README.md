# 🏈 College Football Analytics

![cfb](https://github.com/user-attachments/assets/6059793e-fb3c-41d1-a668-c7b593085e2f)

Implementing deep learning techniques to predict college football game outcomes and NFL draft picks using historical data and advanced analytics.

## 📑 Key Files and Resources

- **[data/scripts/examples.ipynb](data/scripts/examples.ipynb)**: Demonstrates how to fetch and use college football data from the API, including games, team box scores, player stats, and more.
- **[data/scripts/draft_data.ipynb](data/scripts/draft_data.ipynb)**: Shows how to collect and process NFL draft data.
- **[notebooks/draft_prediction.ipynb](notebooks/draft_prediction.ipynb)**: Contains models for predicting NFL draft outcomes.
- **[app/dashapp.py](app/dashapp.py)**: Interactive dashboard application for visualizing football analytics.

## 📊 Data Sources

- [College Football Data API](https://apinext.collegefootballdata.com)
- [NFL Contract and Draft Dataset](https://www.kaggle.com/datasets/nicholasliusontag/nfl-contract-and-draft-data)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Conda

### Installation

1. Clone the repository
```bash
git clone https://github.com/KevinDKao/college-football-analytics.git
cd college-football-analytics
```

2. Create and activate Conda environment
```bash
conda create -n cfb-analytics python=3.11
conda activate cfb-analytics
```

3. Install requirements
```bash
pip install -r requirements.txt
```

## 📁 Repository Structure

```
OwlAboutFootball_S25/
├── 📊 data/                  # Data collection and processing
│   ├── scripts/              # Python scripts and notebooks for data collection
│   │   ├── examples.ipynb    # Examples of how to fetch and use the CFB API data
│   │   ├── draft_data.ipynb  # NFL draft data collection
│   │   ├── games.py          # Functions for retrieving game data
│   │   ├── players.py        # Functions for retrieving player data
│   │   ├── ratings.py        # Functions for retrieving team ratings
│   │   └── ...               # Other data collection modules
│   └── data_files/           # Storage for collected data
│
├── 🧪 notebooks/             # Jupyter notebooks for analysis and modeling
│   ├── boxscore_data.ipynb   # Analysis of game box scores
│   ├── draft_prediction.ipynb # NFL draft prediction models
│   ├── ratings_data.ipynb    # Analysis of team ratings
│   └── kkao_models/          # Model implementations
│
├── 📈 models/                # Model documentation and artifacts
│
├── 🖥️ app/                   # Web application
│   └── dashapp.py            # Dashboard application for visualizing analytics
│
├── 📦 archive/               # Archived code and models
├── 📋 requirements.txt       # Project dependencies
└── 📖 README.md              # This file
```

## 🔑 API Keys

1. Create a `.env` file in the root directory
2. Add your API keys:
```
CFB_API_KEY=your_college_football_api_key
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details
