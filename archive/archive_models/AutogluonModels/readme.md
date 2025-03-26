# College Football Draft Predictions

## Project Summary
This project uses machine learning to predict whether college football players will be drafted into the NFL and estimate their overall rating as professional players.

## Key Results

### Best Drafting Model: ag_20250225_030646
### Best Player Rating Model: ag_20250225_031653

### Draft Prediction Model
- **Accuracy**: Our model achieved perfect validation scores (100% accuracy) using several methods
- **Data Size**: Trained on 11,429 college player profiles
- **Features**: Analyzed 71 different player statistics and attributes
- **Best Performer**: The "WeightedEnsemble" algorithm provided the most reliable predictions

### Player Rating Prediction
- **Prediction Error**: Our model achieved a root mean squared error of 1.88 (lower is better)
- **Data Size**: Trained on 205 player profiles who were successfully drafted
- **Best Performer**: A combination of RandomForest, XGBoost, and ExtraTrees algorithms

## How It Works
The system analyzes player statistics, college performance, physical attributes, and other factors to make two key predictions:
1. Whether a player will be drafted (yes/no)
2. For players predicted to be drafted, their expected overall rating as an NFL player

## Example Output
For a sample player analysis:
- Draft prediction: Not likely to be drafted
- Draft probability: Less than 1%

---

*This project uses AutoGluon, an automated machine learning framework, to build high-performance prediction models with minimal manual tuning.*


# Auto-Gluon Results

Based on the AutoGluon output in your logs, here are the best models for each prediction task:

### For Draft Prediction (Binary Classification)
The best model was **WeightedEnsemble_L2** with:
- Perfect validation score (1.0 ROC AUC)
- Very fast prediction time
- This ensemble primarily relied on ExtraTreesGini_BAG_L1 (a bagged Extra Trees model using the Gini criterion)

### For Player Rating Prediction (Regression)
The best model was **WeightedEnsemble_L3** with:
- Best root mean squared error score (-1.8799, where closer to zero is better)
- This ensemble combined three models with specific weights:
  - RandomForestMSE_BAG_L2 (54.5%)
  - XGBoost_BAG_L1 (31.8%)
  - ExtraTreesMSE_BAG_L2 (13.6%)

In both cases, the weighted ensemble models performed best by combining the strengths of multiple underlying models. Tree-based models (Random Forest, Extra Trees, XGBoost) dominated the predictions, suggesting that these methods work particularly well for football player evaluation.