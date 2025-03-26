# 📓 Analysis Notebooks

This directory contains Jupyter notebooks for data collection, analysis, and model development for the College Football Analytics project. These notebooks provide a step-by-step walkthrough of the data processing pipeline and model development process.

## 📊 Data Collection & Processing

### [boxscore_data.ipynb](boxscore_data.ipynb)
This notebook focuses on collecting and processing game-level statistics:
- Retrieves weekly team box score data for college football games
- Expands nested data structures into clean tabular format
- Organizes statistics for both teams in a single row per game
- Prepares data for further analysis and modeling

### [ratings_data.ipynb](ratings_data.ipynb)
Collects and analyzes team ratings and rankings data:
- Gathers SP+ conference ratings and team rankings from 2014 to 2024
- Processes and cleans ratings data for time-series analysis
- Visualizes conference performance trends over time
- Provides insights into team strength metrics used in predictive models

## 🏈 Draft Analysis & Prediction

### [draft_prediction.ipynb](draft_prediction.ipynb)
Explores factors that influence NFL draft outcomes:
- Analyzes college player statistics and their correlation with draft status
- Identifies key features that predict whether a player will be drafted
- Implements preliminary predictive models
- Visualizes feature importance and model performance

### [drafting_models/](drafting_models/)
A comprehensive directory containing advanced machine learning models for draft prediction:
- **[XGBoost](drafting_models/xgboost/)**: Gradient boosting implementation
- **[Random Forest](drafting_models/random_forest/)**: Ensemble learning approach
- **[AutoGluon](drafting_models/autogluon/)**: Automated machine learning framework
- **[comparison.py](drafting_models/comparison.py)**: Script for comparing model performance

For detailed information about the draft prediction models, see the [drafting_models README](drafting_models/readme.md).

## 🔄 Data Flow

The notebooks in this directory follow a logical progression:

1. **Data Collection**: 
   - `boxscore_data.ipynb` and `ratings_data.ipynb` gather raw data from the College Football Data API
   - Data is stored in structured formats (CSV, Feather) for further processing

2. **Exploratory Analysis**:
   - Initial data exploration and visualization
   - Feature engineering and data transformation
   - Statistical analysis of relationships between variables

3. **Model Development**:
   - `draft_prediction.ipynb` provides initial modeling approaches
   - `drafting_models/` contains advanced model implementations
   - Models are evaluated and compared for performance

4. **Visualization & Insights**:
   - Results are visualized to extract actionable insights
   - Key findings are documented for implementation in the dashboard application

## 🚀 Getting Started

To work with these notebooks:

1. Ensure you have set up your environment according to the main [README](../README.md)
2. Make sure you have configured your API key in `data/scripts/credentials.py`
3. Start with the data collection notebooks before moving to analysis and modeling
4. Run the notebooks in sequential order to ensure data dependencies are met

## 📦 Dependencies

These notebooks require the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- autogluon (for AutoGluon notebooks)
- plotly (for interactive visualizations)

All dependencies can be installed via the main `requirements.txt` file in the project root.

## 📝 Notes

- Some notebooks may take significant time to run, especially when fetching large amounts of data or training complex models
- Consider using checkpoints to save intermediate results when working with time-intensive processes
- The `drafting_models/` directory contains the most advanced and optimized models for draft prediction
