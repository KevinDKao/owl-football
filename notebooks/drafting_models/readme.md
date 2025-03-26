# 🏈 NFL Draft Prediction Models

This directory contains various machine learning models for predicting NFL draft outcomes based on college football player performance data. The models analyze player statistics, team performance, and other metrics to predict whether a player will be drafted and potentially their draft position.

## 📊 Model Implementations

### [XGBoost](xgboost/)
XGBoost implementation for draft prediction with two variants:
- **[xgboost_model.py](xgboost/xgboost_model.py)**: Includes player ratings as features
- **[xgboost_model_no_rating.py](xgboost/xgboost_model_no_rating.py)**: Excludes player ratings for scenarios where this data isn't available

XGBoost models typically provide excellent performance for this task due to their ability to handle complex feature interactions and non-linear relationships.

### [Random Forest](random_forest/)
Random Forest implementation with two variants:
- **[model.py](random_forest/model.py)**: Includes player ratings
- **[model_no_rating.py](random_forest/model_no_rating.py)**: Excludes player ratings

Random Forest models offer good interpretability while maintaining strong predictive performance.

### [AutoGluon](autogluon/)
- **[autogluon.ipynb](autogluon/autogluon.ipynb)**: Automated machine learning approach that tests multiple models and ensembles to find the best performer
- Uses **[draft_data.feather](autogluon/draft_data.feather)** as input data

AutoGluon automates the model selection and hyperparameter tuning process, potentially discovering optimal model configurations that might be missed in manual tuning.

## 🔍 Model Comparison

The **[comparison.py](comparison.py)** script provides a comprehensive comparison of all model implementations, including:

1. **Performance Metrics**:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - R-squared (R²)
   - Classification metrics for draft status prediction

2. **Feature Importance Analysis**:
   - Comparison of which features each model finds most predictive
   - Visualization of feature importance across models

3. **Prediction Analysis**:
   - Comparison of predictions across models
   - Identification of players where models agree/disagree

Results of these comparisons are stored in the [comparison_summary](comparison_summary/) directory.

## 📁 Directory Structure

```
drafting_models/
├── comparison.py                # Script to compare all model implementations
├── comparison_summary/          # Output of model comparisons
│
├── xgboost/                     # XGBoost implementation
│   ├── xgboost_model.py         # XGBoost with ratings
│   ├── xgboost_model_no_rating.py # XGBoost without ratings
│   ├── output/                  # Results with ratings
│   ├── output_no_rating/        # Results without ratings
│   └── feature_importance.png   # Feature importance visualization
│
├── random_forest/               # Random Forest implementation
│   ├── model.py                 # Random Forest with ratings
│   ├── model_no_rating.py       # Random Forest without ratings
│   ├── output/                  # Results with ratings
│   └── output_no_rating/        # Results without ratings
│
├── autogluon/                   # AutoGluon implementation
│   ├── autogluon.ipynb          # AutoGluon notebook
│   └── draft_data.feather       # Input data
│
└── feathers/                    # Feather format data files
```

## 🎯 Target Variable

The primary prediction target is whether a player will be drafted (`drafted` column):
- `1`: Player was selected in the NFL draft
- `0`: Player went undrafted

Some models may also predict the draft position (round and overall pick) for players predicted to be drafted.

## 🚀 Running the Models

Each model directory contains standalone scripts or notebooks that can be run independently:

1. **XGBoost**:
   ```bash
   cd xgboost
   python xgboost_model.py  # With ratings
   python xgboost_model_no_rating.py  # Without ratings
   ```

2. **Random Forest**:
   ```bash
   cd random_forest
   python model.py  # With ratings
   python model_no_rating.py  # Without ratings
   ```

3. **AutoGluon**: Open and run the Jupyter notebook `autogluon/autogluon.ipynb`

4. **Model Comparison**:
   ```bash
   python comparison.py
   ```

## 📊 Key Findings

- Feature importance analysis shows that [specific statistics] are the strongest predictors of draft status
- Models with player ratings generally outperform those without, but the gap is smaller than expected
- The ensemble approach of AutoGluon provides marginal improvements over carefully tuned XGBoost models
- Prediction accuracy is highest for clear first-round talents and undrafted players, with more uncertainty in middle rounds

## 📝 Notes

- Input data should be prepared using the [draft_data.ipynb](../../data/scripts/draft_data.ipynb) notebook
- All models are configured to use the same train/test split for fair comparison
- The comparison script generates visualizations to help interpret model differences
