# 🤖 AutoGluon NFL Draft Predictor

This directory contains an implementation of AutoML for NFL draft prediction using [AutoGluon](https://auto.gluon.ai/stable/index.html), a powerful automated machine learning framework developed by Amazon.

## 📋 Overview

AutoGluon automates the end-to-end machine learning pipeline, including:
- Feature preprocessing
- Model selection
- Hyperparameter tuning
- Model ensembling

This implementation uses AutoGluon to solve two related problems:
1. **Classification**: Will a player be drafted? (binary prediction)
2. **Regression**: If drafted, what will be the player's draft position? (numerical prediction)

## 📊 Data

The model uses `draft_data.feather` as input, which contains:
- College player statistics
- Team performance metrics
- Player biographical information
- Historical draft outcomes

## 🧠 Model Architecture

The implementation creates two separate AutoGluon predictors:

### Classification Model
- **Target**: `is_drafted` (binary: 1 = drafted, 0 = undrafted)
- **Evaluation Metric**: ROC AUC
- **Problem Type**: Binary classification
- **Time Limit**: 10 minutes per training run
- **Preset**: "best_quality" (prioritizes accuracy over training speed)

### Regression Model
- **Target**: `overall` (draft position number)
- **Evaluation Metric**: RMSE (Root Mean Squared Error)
- **Problem Type**: Regression
- **Time Limit**: 10 minutes per training run
- **Preset**: "best_quality"
- **Note**: Only trained on players who were actually drafted

## 🚀 How to Use

### Running the Notebook

1. Ensure you have AutoGluon installed:
   ```bash
   pip install autogluon
   ```

2. Open and run the [autogluon.ipynb](autogluon.ipynb) notebook

3. The notebook will:
   - Load the data from `draft_data.feather`
   - Preprocess features
   - Train classification and regression models
   - Evaluate model performance
   - Provide a prediction function for new players

### Making Predictions

The notebook includes a `predict_player_draft` function that:
- Takes player data as input
- Returns a dictionary with:
  - `will_be_drafted`: Boolean prediction
  - `draft_probability`: Confidence score (0-1)
  - `draft_position`: Predicted overall pick number (if drafted)
  - `feature_importance`: Key factors influencing the prediction

Example usage:
```python
player_data = {
    'position': 'QB',
    'passing_yards': 4500,
    'passing_tds': 40,
    # ... other features
}

prediction = predict_player_draft(
    player_data, 
    predictor_clf,  # Classification model
    predictor_reg,  # Regression model
    categorical_columns
)

print(f"Will be drafted: {prediction['will_be_drafted']}")
print(f"Draft probability: {prediction['draft_probability']:.2f}")
if prediction['draft_position'] is not None:
    print(f"Predicted draft position: {prediction['draft_position']:.1f}")
```

## 🔍 Advantages of AutoGluon

1. **Model Diversity**: Automatically tests multiple model types (Random Forest, XGBoost, Neural Networks, etc.)

2. **Ensemble Learning**: Combines multiple models to improve prediction accuracy

3. **Automated Hyperparameter Tuning**: Optimizes model parameters without manual intervention

4. **Robust Feature Handling**: Automatically handles missing values, categorical features, and feature scaling

5. **Time Efficiency**: Produces high-quality models within specified time constraints

## 📝 Notes

- The models are saved in a directory structure created by AutoGluon during training
- Performance metrics are available through the predictor's `leaderboard()` method
- For production use, consider increasing the `time_limit` parameter for potentially better results
- The notebook is primarily exploratory and serves as a baseline for comparison with other model implementations

## 🔄 Comparison with Other Models

AutoGluon's automated approach provides a strong baseline that can be compared against the manually tuned models in the parent directory:
- [XGBoost](../xgboost/)
- [Random Forest](../random_forest/)

The comprehensive model comparison is available in the [comparison.py](../comparison.py) script.
