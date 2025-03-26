# NFL Draft Position Prediction: Random Forest Model With Scout Ratings

This directory contains the results of a Random Forest model trained to predict NFL draft positions using both objective player statistics and subjective scout rating data.

## Model Overview

- **Training Data:** NFL player statistics from years 2011-2020
- **Testing Data:** Years 2021, 2023, and 2024
- **Model Type:** Random Forest with 100 estimators (random_state=42)
- **Key Feature:** Incorporates both raw player statistics AND subjective scout ratings
- **Scout Rating Features Included:** 
  - preDraftGrade
  - preDraftRanking
  - preDraftPositionRanking

## Performance Summary

| Year | MSE    | RMSE   | MAE    | R² Score | Accuracy | Precision | Recall | F1 Score |
|------|--------|--------|--------|----------|----------|-----------|--------|----------|
| 2021 | 70.93  | 8.42   | 1.15   | 0.90     | 0.9996   | 0.989     | 1.000  | 0.994    |
| 2023 | 146.99 | 12.12  | 1.28   | 0.62     | 0.9972   | 0.916     | 0.923  | 0.919    |
| 2024 | 74.53  | 8.63   | 0.93   | 0.81     | 0.9983   | 0.926     | 0.981  | 0.953    |

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
- Predicted Drafted: 261 players
- True Positives: 239
- False Positives: 22
- False Negatives: 20
- True Negatives: 14,676

**2024 Test Results:**
- Actually Drafted: 257 players
- Predicted Drafted: 272 players
- True Positives: 252
- False Positives: 20
- False Negatives: 5
- True Negatives: 14,010

## Top 10 Feature Importance

The model combines physical attributes with scout evaluations, with weight and preDraftGrade being the dominant features:

| Feature                 | Importance |
|-------------------------|------------|
| weight                  | 0.746      |
| preDraftGrade           | 0.168      |
| preDraftRanking         | 0.026      |
| preDraftPositionRanking | 0.015      |
| height                  | 0.006      |
| receiving_YPR           | 0.003      |
| interceptions_AVG       | 0.002      |
| receiving_YDS           | 0.002      |
| receiving_LONG          | 0.002      |
| interceptions_INT       | 0.002      |

## Key Insights

1. **Improved Performance with Scout Ratings:** Compared to models using only raw statistics, this model achieves notably better performance metrics, particularly with lower RMSE values and higher R² scores (2021: 0.90, 2024: 0.81).

2. **Scout Ratings Add Value:** The three scout rating features collectively account for 21% of the model's predictive power, confirming their significant value in draft position prediction.

3. **Physical Attributes Still Dominant:** Despite including scout ratings, player weight (74.6%) remains by far the most influential feature, reinforcing the critical role of physical attributes in draft evaluation.

4. **Year-to-Year Consistency:** The model shows excellent consistency across test years in identifying drafted players, with F1 scores ranging from 0.919 to 0.994.

5. **2023 Performance Drop:** The model showed somewhat reduced performance on 2023 data (R² of 0.62) compared to 2021 and 2024, suggesting potential anomalies in that draft class or scouting data.

6. **Position-Specific Analysis:** The position accuracy visualizations (see PNG files) provide detailed breakdowns of how the model performs across different football positions.

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

## Comparison with No-Rating Model

Compared to our companion model without scout ratings (see `/output_no_rating`), this model demonstrates:

1. **Higher Accuracy:** Lower RMSE values across all test years (2021: 8.42 vs 13.83, 2024: 8.63 vs 10.81)
2. **Better Explanatory Power:** Improved R² scores (2021: 0.90 vs 0.73, 2024: 0.81 vs 0.70)
3. **Similar Classification Performance:** Both models excel at identifying drafted vs. undrafted players
4. **Shared Feature Importance:** Weight remains the dominant feature in both models

## Conclusion

This Random Forest model effectively combines objective player statistics with subjective scout ratings to predict NFL draft positions with high accuracy. The strong performance (R² scores between 0.62-0.90) demonstrates that incorporating scout evaluations significantly improves predictive power over raw statistics alone.

The model's exceptional performance in 2021 and 2024 (R² ≥ 0.81) makes it particularly valuable for draft strategy and player valuation. The significant contribution of scout ratings suggests that while physical attributes remain paramount, the expert evaluation captured in ratings adds substantial predictive value.
