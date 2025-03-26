import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shutil
from datetime import datetime

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "comparison_summary")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paths to model outputs
RF_OUTPUT = os.path.join(BASE_DIR, "random_forest", "output")
RF_NO_RATING_OUTPUT = os.path.join(BASE_DIR, "random_forest", "output_no_rating")
XGB_OUTPUT = os.path.join(BASE_DIR, "xgboost", "output")
XGB_NO_RATING_OUTPUT = os.path.join(BASE_DIR, "xgboost", "output_no_rating")

# Function to load model results
def load_model_results(model_dir):
    results = {}

    # Load metrics
    metrics_path = os.path.join(model_dir, "metrics.pkl")
    if os.path.exists(metrics_path):
        with open(metrics_path, "rb") as f:
            results["metrics"] = pickle.load(f)

    # Load predictions
    predictions_path = os.path.join(model_dir, "predictions.csv")
    if os.path.exists(predictions_path):
        results["predictions"] = pd.read_csv(predictions_path)

    # Load feature importance
    feature_importance_path = os.path.join(model_dir, "feature_importance.csv")
    if os.path.exists(feature_importance_path):
        results["feature_importance"] = pd.read_csv(feature_importance_path)

    return results


# Load results for all models
rf_results = load_model_results(RF_OUTPUT)
rf_no_rating_results = load_model_results(RF_NO_RATING_OUTPUT)
xgb_results = load_model_results(XGB_OUTPUT)
xgb_no_rating_results = load_model_results(XGB_NO_RATING_OUTPUT)

# Create a metrics comparison dataframe
def create_metrics_comparison():
    models = {
        "Random Forest": rf_results.get("metrics", {}),
        "Random Forest (No Rating)": rf_no_rating_results.get("metrics", {}),
        "XGBoost": xgb_results.get("metrics", {}),
        "XGBoost (No Rating)": xgb_no_rating_results.get("metrics", {}),
    }

    metrics_data = []
    for model_name, metrics in models.items():
        if metrics:
            for year, year_metrics in metrics.items():
                if isinstance(year_metrics, dict):
                    metrics_data.append(
                        {
                            "Model": model_name,
                            "Year": year,
                            "RMSE": year_metrics.get("rmse", np.nan),
                            "MAE": year_metrics.get("mae", np.nan),
                            "R²": year_metrics.get("r2", np.nan),
                        }
                    )

    metrics_df = pd.DataFrame(metrics_data)

    # Save to CSV
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_comparison.csv")
    metrics_df.to_csv(metrics_path, index=False)

    return metrics_df


# Create feature importance comparison
def create_feature_importance_comparison():
    feature_data = []

    # Process each model's feature importance
    models = {
        "Random Forest": rf_results.get("feature_importance"),
        "Random Forest (No Rating)": rf_no_rating_results.get("feature_importance"),
        "XGBoost": xgb_results.get("feature_importance"),
        "XGBoost (No Rating)": xgb_no_rating_results.get("feature_importance"),
    }

    for model_name, importance_df in models.items():
        if isinstance(importance_df, pd.DataFrame):
            # Get top 20 features
            top_features = importance_df.sort_values(
                "Importance", ascending=False
            ).head(20)

            for _, row in top_features.iterrows():
                feature_data.append(
                    {
                        "Model": model_name,
                        "Feature": row["Feature"],
                        "Importance": row["Importance"],
                    }
                )

    if feature_data:
        feature_df = pd.DataFrame(feature_data)

        # Save to CSV
        feature_path = os.path.join(OUTPUT_DIR, "feature_importance_comparison.csv")
        feature_df.to_csv(feature_path, index=False)

        return feature_df

    return None


# Create prediction comparison
def create_prediction_comparison():
    predictions_data = []

    # Process each model's predictions
    models = {
        "Random Forest": rf_results.get("predictions"),
        "Random Forest (No Rating)": rf_no_rating_results.get("predictions"),
        "XGBoost": xgb_results.get("predictions"),
        "XGBoost (No Rating)": xgb_no_rating_results.get("predictions"),
    }

    # Combine predictions from all models
    combined_df = None

    for model_name, pred_df in models.items():
        if isinstance(pred_df, pd.DataFrame):
            # Rename prediction column to model name
            pred_df = pred_df.copy()
            if "predicted_overall" in pred_df.columns:
                pred_df.rename(
                    columns={"predicted_overall": f"{model_name}_predicted"},
                    inplace=True,
                )

            # Merge with combined dataframe
            if combined_df is None:
                combined_df = pred_df
            else:
                # Keep only necessary columns for merging
                merge_cols = [
                    col
                    for col in pred_df.columns
                    if col.endswith("_predicted")
                    or col == "name"
                    or col == "year"
                    or col == "overall"
                ]
                combined_df = pd.merge(
                    combined_df,
                    pred_df[merge_cols],
                    on=["name", "year", "overall"],
                    how="outer",
                )

    if combined_df is not None:
        # Save to CSV
        combined_path = os.path.join(OUTPUT_DIR, "predictions_comparison.csv")
        combined_df.to_csv(combined_path, index=False)

        return combined_df

    return None


# Generate visualizations
def create_visualizations(metrics_df, feature_df, predictions_df):
    # 1. Metrics comparison bar chart
    if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
        plt.figure(figsize=(15, 10))

        # RMSE comparison
        plt.subplot(3, 1, 1)
        sns.barplot(x="Year", y="RMSE", hue="Model", data=metrics_df)
        plt.title("RMSE Comparison Across Models")
        plt.ylabel("RMSE (lower is better)")
        plt.legend(loc="upper right")

        # MAE comparison
        plt.subplot(3, 1, 2)
        sns.barplot(x="Year", y="MAE", hue="Model", data=metrics_df)
        plt.title("MAE Comparison Across Models")
        plt.ylabel("MAE (lower is better)")
        plt.legend(loc="upper right")

        # R² comparison
        plt.subplot(3, 1, 3)
        sns.barplot(x="Year", y="R²", hue="Model", data=metrics_df)
        plt.title("R² Comparison Across Models")
        plt.ylabel("R² (higher is better)")
        plt.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "metrics_comparison.png"))
        plt.close()

    # 2. Feature importance comparison
    if isinstance(feature_df, pd.DataFrame) and not feature_df.empty:
        # Get top 10 features across all models
        top_features = (
            feature_df.groupby("Feature")["Importance"]
            .mean()
            .nlargest(10)
            .index.tolist()
        )
        filtered_df = feature_df[feature_df["Feature"].isin(top_features)]

        plt.figure(figsize=(15, 10))
        sns.barplot(x="Importance", y="Feature", hue="Model", data=filtered_df)
        plt.title("Top 10 Feature Importance Comparison")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_comparison.png"))
        plt.close()

    # 3. Prediction error distribution
    if isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty:
        plt.figure(figsize=(15, 10))

        # Calculate error for each model
        models = [col for col in predictions_df.columns if col.endswith("_predicted")]

        for i, model in enumerate(models, 1):
            model_name = model.replace("_predicted", "")
            predictions_df[f"{model_name}_error"] = (
                predictions_df[model] - predictions_df["overall"]
            )

            plt.subplot(2, 2, i)
            sns.histplot(predictions_df[f"{model_name}_error"].dropna(), kde=True)
            plt.title(f"{model_name} Prediction Error Distribution")
            plt.xlabel("Error (Predicted - Actual)")
            plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "prediction_error_distribution.png"))
        plt.close()

        # 4. Scatter plot of actual vs predicted
        plt.figure(figsize=(15, 10))

        for i, model in enumerate(models, 1):
            model_name = model.replace("_predicted", "")

            plt.subplot(2, 2, i)
            plt.scatter(predictions_df["overall"], predictions_df[model], alpha=0.5)

            # Add perfect prediction line
            min_val = min(predictions_df["overall"].min(), predictions_df[model].min())
            max_val = max(predictions_df["overall"].max(), predictions_df[model].max())
            plt.plot([min_val, max_val], [min_val, max_val], "r--")

            plt.title(f"{model_name}: Actual vs Predicted")
            plt.xlabel("Actual Draft Position")
            plt.ylabel("Predicted Draft Position")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"))
        plt.close()


# Create a summary report
def create_summary_report(metrics_df):
    if not isinstance(metrics_df, pd.DataFrame) or metrics_df.empty:
        return

    # Calculate average metrics across years for each model
    avg_metrics = (
        metrics_df.groupby("Model")[["RMSE", "MAE", "R²"]].mean().reset_index()
    )

    # Find best model for each metric
    best_rmse = avg_metrics.loc[avg_metrics["RMSE"].idxmin()]
    best_mae = avg_metrics.loc[avg_metrics["MAE"].idxmin()]
    best_r2 = avg_metrics.loc[avg_metrics["R²"].idxmax()]

    # Create summary text
    summary = f"""# Model Comparison Summary
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Performance

| Model | Avg RMSE | Avg MAE | Avg R² |
|-------|----------|---------|--------|
"""

    for _, row in avg_metrics.iterrows():
        summary += f"| {row['Model']} | {row['RMSE']:.4f} | {row['MAE']:.4f} | {row['R²']:.4f} |\n"

    summary += f"""
## Best Performing Models

- **Best RMSE**: {best_rmse['Model']} (RMSE = {best_rmse['RMSE']:.4f})
- **Best MAE**: {best_mae['Model']} (MAE = {best_mae['MAE']:.4f})
- **Best R²**: {best_r2['Model']} (R² = {best_r2['R²']:.4f})

## Impact of Draft Ratings

The comparison between models with and without draft ratings shows:
- Random Forest: {'Improved' if avg_metrics[avg_metrics['Model'] == 'Random Forest']['RMSE'].values[0] < avg_metrics[avg_metrics['Model'] == 'Random Forest (No Rating)']['RMSE'].values[0] else 'Worsened'} performance with draft ratings
- XGBoost: {'Improved' if avg_metrics[avg_metrics['Model'] == 'XGBoost']['RMSE'].values[0] < avg_metrics[avg_metrics['Model'] == 'XGBoost (No Rating)']['RMSE'].values[0] else 'Worsened'} performance with draft ratings

## Conclusion

Based on the metrics, the {best_r2['Model']} model provides the best overall performance for predicting draft positions.
"""

    # Write to file
    with open(os.path.join(OUTPUT_DIR, "model_comparison_summary.md"), "w") as f:
        f.write(summary)


# Copy important visualizations from model directories
def copy_model_visualizations():
    # Define source and destination paths
    visualization_files = [
        (
            os.path.join(RF_OUTPUT, "feature_importance.png"),
            os.path.join(OUTPUT_DIR, "rf_feature_importance.png"),
        ),
        (
            os.path.join(RF_NO_RATING_OUTPUT, "feature_importance.png"),
            os.path.join(OUTPUT_DIR, "rf_no_rating_feature_importance.png"),
        ),
        (
            os.path.join(XGB_OUTPUT, "feature_importance.png"),
            os.path.join(OUTPUT_DIR, "xgb_feature_importance.png"),
        ),
        (
            os.path.join(XGB_NO_RATING_OUTPUT, "feature_importance.png"),
            os.path.join(OUTPUT_DIR, "xgb_no_rating_feature_importance.png"),
        ),
    ]

    # Copy files if they exist
    for src, dst in visualization_files:
        if os.path.exists(src):
            shutil.copy2(src, dst)


# Main execution
def main():
    print("Starting model comparison...")

    # Create comparison dataframes
    metrics_df = create_metrics_comparison()
    feature_df = create_feature_importance_comparison()
    predictions_df = create_prediction_comparison()

    # Generate visualizations
    create_visualizations(metrics_df, feature_df, predictions_df)

    # Copy existing visualizations
    copy_model_visualizations()

    # Create summary report
    create_summary_report(metrics_df)

    print(f"Comparison complete. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
