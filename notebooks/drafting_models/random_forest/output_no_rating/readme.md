# NFL Draft Position Prediction: Random Forest Model (No Rating Features)

This directory contains the results of a Random Forest model trained to predict NFL draft positions using only raw player statistics, without relying on subjective draft ratings.

## Model Overview

- **Training Data:** NFL player statistics from years 2011-2020
- **Testing Data:** Years 2021, 2023, and 2024
- **Model Type:** Random Forest with 100 estimators (random_state=42)
- **Key Feature:** Uses only objective player statistics, excluding subjective draft ratings
- **Excluded Rating Features:** 
  - preDraftGrade
  - preDraftRanking
  - preDraftPositionRanking

## Performance Summary

| Year | MSE    | RMSE   | MAE    | R² Score | Accuracy | Precision | Recall | F1 Score |
|------|--------|--------|--------|----------|----------|-----------|--------|----------|
| 2021 | 191.37 | 13.83  | 2.09   | 0.73     | 0.9996   | 0.989     | 1.000  | 0.994    |
| 2023 | 170.60 | 13.06  | 1.41   | 0.56     | 0.9971   | 0.915     | 0.915  | 0.915    |
| 2024 | 116.91 | 10.81  | 1.23   | 0.70     | 0.9983   | 0.926     | 0.981  | 0.953    |

### Classification Metrics (Drafted vs. Undrafted)

**2021 Test Results:**
- Actually Drafted: 259 players
- Predicted Drafted: 262 players
- True Positives: 259 (correctly predicted as drafted)
- False Positives: 3 (predicted drafted but actually undrafted)
- False Negatives: 0 (predicted undrafted but actually drafted)
- True Negatives: 7,763 (correctly predicted as undrafted)

**2023 Test Results:**
- Actually Drafted: 259 players
- Predicted Drafted: 259 players
- True Positives: 237
- False Positives: 22
- False Negatives: 22
- True Negatives: 14,676

**2024 Test Results:**
- Actually Drafted: 257 players
- Predicted Drafted: 272 players
- True Positives: 252
- False Positives: 20
- False Negatives: 5
- True Negatives: 14,010

## Top 10 Feature Importance

The model relies heavily on physical attributes, particularly weight, followed by various performance metrics:

| Feature          | Importance |
|------------------|------------|
| weight           | 0.835      |
| height           | 0.020      |
| defensive_TOT    | 0.010      |
| receiving_YPR    | 0.009      |
| interceptions_AVG| 0.009      |
| defensive_SOLO   | 0.008      |
| receiving_YDS    | 0.008      |
| receiving_LONG   | 0.008      |
| interceptions_YDS| 0.007      |
| interceptions_INT| 0.007      |

## Key Insights

1. **Accuracy vs. Precision/Recall:** The model achieves high overall accuracy across all test years (>99%), demonstrating strong general predictive power.

2. **Physical Attributes Matter:** Player weight (83.5%) and height (2.0%) are the dominant predictive features, suggesting physical attributes remain critical in draft evaluation even without subjective ratings.

3. **Performance Across Years:** The model performed best on the 2024 data with the lowest RMSE (10.81) and highest precision-recall balance, though 2021 had perfect recall.

4. **False Positives vs. False Negatives:** The model generally predicts more drafted players than actual (especially in 2024), showing a slight bias toward positive predictions.

5. **Position-Specific Analysis:** The position accuracy visualizations (see PNG files) provide detailed breakdowns of how the model performs across different football positions.

## Files in This Directory

- **CSV Files:**
  - `predictions_*.csv`: Model predictions for each test year
  - `feature_importance.csv`: Detailed importance of all features
  - `model_performance.csv`: Summary metrics for model evaluation

- **Visualizations:**
  - `position_accuracy_*.png`: Accuracy by player position for each test year
  - `actual_vs_predicted_drafted_only_*.png`: Scatter plots comparing actual vs. predicted draft positions
  - `drafted_confusion_matrix_*.png`: Confusion matrices for drafted/undrafted classification
  - `feature_importance.png`: Bar chart of feature importance

- **Reports:**
  - `summary_report.txt`: Detailed model metrics and performance analysis

## Conclusion

This model demonstrates that raw player statistics alone can effectively predict NFL draft positions without relying on subjective pre-draft ratings. The strong performance (R² scores between 0.56-0.73) suggests that objective measures capture much of what determines draft outcomes, with physical attributes like weight and height being particularly influential.

The model's high precision and recall in identifying drafted players (especially in 2024 with a 0.953 F1 score) makes it a valuable tool for talent evaluation and draft strategy planning.
