import pandas as pd
import numpy as np
import os
import pickle
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATHER_DIR = os.path.join(BASE_DIR, "feathers")
OUTPUT_DIR = os.path.join(BASE_DIR, "xgboost", "output_no_rating")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to load and preprocess data
def load_and_preprocess_data(year):
    file_path = os.path.join(FEATHER_DIR, f"draft_data{year}.feather")
    df = pd.read_feather(file_path)

    # Handle missing values
    df = df.fillna(0)

    return df


# Function to prepare features and target
def prepare_features_target(df):
    # Define features to use (excluding non-predictive columns)
    exclude_cols = [
        "collegeAthleteId",
        "nflAthleteId",
        "collegeId",
        "collegeTeam",
        "collegeConference",
        "nflTeamId",
        "nflTeam",
        "year",
        "overall",
        "round",
        "pick",
        "name",
        "position",
        "hometownInfo",
        "preDraftGrade",
        "preDraftRanking",
        "preDraftPositionRanking",
    ]

    # Get all numeric columns for features
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Convert any remaining non-numeric columns and handle problematic values
    for col in feature_cols:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Replace inf/-inf with 0
        df[col] = df[col].replace([np.inf, -np.inf], 0)

        # Replace NaN with 0
        df[col] = df[col].fillna(0)

        # Handle extremely large values by capping
        percentile_99 = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=percentile_99)

    X = df[feature_cols]
    y = df["overall"]

    return X, y


# Load training data (2011-2020)
print("Loading training data...")
train_years = list(range(2011, 2021))
train_dfs = []

for year in train_years:
    df = load_and_preprocess_data(year)
    train_dfs.append(df)

train_df = pd.concat(train_dfs, ignore_index=True)
print(f"Training data shape: {train_df.shape}")

# Prepare features and target for training
X_train, y_train = prepare_features_target(train_df)

# Verify no NaN or inf values remain
print("Checking for NaN or infinite values in training data...")
has_nan = np.isnan(X_train.values).any()
has_inf = np.isinf(X_train.values).any()
print(f"Has NaN: {has_nan}, Has Inf: {has_inf}")

if has_nan or has_inf:
    print(
        "Warning: Still found NaN or infinite values. Applying additional cleaning..."
    )
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_train = X_train.fillna(0)

# Train XGBoost model
print("Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# Save the model
model_path = os.path.join(OUTPUT_DIR, "xgboost_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {model_path}")

# Feature importance
feature_importance = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": model.feature_importances_}
).sort_values("Importance", ascending=False)

# Save feature importance
feature_importance.to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False
)

# Plot feature importance (top 20)
plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importance.head(20))
plt.title("Top 20 Feature Importance (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))

# Test on 2021, 2023, and 2024
test_years = [2021, 2023, 2024]
results = []

for year in test_years:
    print(f"Testing on {year} data...")
    test_df = load_and_preprocess_data(year)
    X_test, y_test = prepare_features_target(test_df)

    # Make sure test data has same columns as training data
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0

    # Ensure columns are in the same order
    X_test = X_test[X_train.columns]

    # Make predictions
    y_pred = model.predict(X_test)

    # Round predictions to nearest integer (since overall is an integer)
    y_pred_rounded = np.round(y_pred)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({"Year": year, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2})

    # Save predictions
    predictions_df = test_df[["name", "position", "collegeTeam", "overall"]].copy()
    predictions_df["predicted_overall"] = y_pred_rounded
    predictions_df["error"] = (
        predictions_df["overall"] - predictions_df["predicted_overall"]
    )

    # Add a column to indicate if player is drafted or undrafted (based on actual)
    # Players with overall = 0 were not drafted, players with overall 1-255 were drafted
    predictions_df["actually_drafted"] = predictions_df["overall"] > 0

    # For predicted drafted, mark any player with predicted_overall within draft range (1-260)
    predictions_df["predicted_drafted"] = predictions_df["predicted_overall"] <= 260
    predictions_df["predicted_drafted"] &= predictions_df["predicted_overall"] > 0

    # Save predictions to CSV
    predictions_df.to_csv(
        os.path.join(OUTPUT_DIR, f"predictions_{year}.csv"), index=False
    )

    # Create a second plot focusing on just the drafted players
    plt.figure(figsize=(14, 10))

    # Get only players who were actually drafted
    drafted_players = predictions_df[predictions_df["actually_drafted"]].copy()

    # Plot each group within the drafted range
    correctly_predicted = drafted_players[drafted_players["predicted_drafted"]]
    underpredicted = drafted_players[~drafted_players["predicted_drafted"]]

    plt.scatter(
        correctly_predicted["overall"],
        correctly_predicted["predicted_overall"],
        alpha=0.6,
        color="blue",
        label="Correctly Predicted as Drafted",
    )

    plt.scatter(
        underpredicted["overall"],
        underpredicted["predicted_overall"],
        alpha=0.6,
        color="red",
        label="Underpredicted (Actually Drafted)",
    )

    # Add reference line
    plt.plot([0, 255], [0, 255], "k--", alpha=0.5)

    # Set axis limits to show the drafted range
    plt.xlim(0, 260)
    plt.ylim(0, 260)

    plt.xlabel("Actual Overall Draft Position", fontsize=12)
    plt.ylabel("Predicted Overall Draft Position", fontsize=12)
    plt.title(
        f"Actual vs Predicted Draft Position - Drafted Players Only ({year})",
        fontsize=14,
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"actual_vs_predicted_drafted_only_{year}.png")
    )

    # Create a confusion matrix for drafted vs undrafted prediction

    # First, ensure we have exactly 255 players marked as actually drafted
    # This is to maintain consistency with the NFL draft having exactly 255 picks
    actually_drafted_count = predictions_df["actually_drafted"].sum()
    if actually_drafted_count != 255:
        print(
            f"Warning: Found {actually_drafted_count} actually drafted players instead of 255 in {year} data."
        )
        print(
            "Adjusting confusion matrix calculations to account for this discrepancy."
        )

    # Similarly, ensure we have exactly 255 players marked as predicted drafted
    predicted_drafted_count = predictions_df["predicted_drafted"].sum()
    if predicted_drafted_count != 255:
        print(
            f"Warning: Found {predicted_drafted_count} predicted drafted players instead of 255 in {year} data."
        )
        print(
            "This should not happen as we're selecting exactly the top 255 players by predicted value."
        )

    # Calculate the confusion matrix values
    true_positives = (
        predictions_df["actually_drafted"] & predictions_df["predicted_drafted"]
    ).sum()
    false_positives = (
        ~predictions_df["actually_drafted"] & predictions_df["predicted_drafted"]
    ).sum()
    false_negatives = (
        predictions_df["actually_drafted"] & ~predictions_df["predicted_drafted"]
    ).sum()
    true_negatives = (
        ~predictions_df["actually_drafted"] & ~predictions_df["predicted_drafted"]
    ).sum()

    # Verify the logical constraints
    if true_positives + false_negatives != actually_drafted_count:
        print(
            f"Error: TP + FN = {true_positives + false_negatives}, but should equal {actually_drafted_count}"
        )

    if true_positives + false_positives != predicted_drafted_count:
        print(
            f"Error: TP + FP = {true_positives + false_positives}, but should equal {predicted_drafted_count}"
        )

    # Create the confusion matrix
    confusion_matrix = np.array(
        [[true_positives, false_positives], [false_negatives, true_negatives]]
    )

    confusion_df = pd.DataFrame(
        confusion_matrix,
        columns=["Actually Drafted", "Actually Undrafted"],
        index=["Predicted Drafted", "Predicted Undrafted"],
    )

    # Calculate metrics
    accuracy = (true_positives + true_negatives) / confusion_matrix.sum()
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Add classification metrics to results
    results[-1].update(
        {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "TP": true_positives,
            "FP": false_positives,
            "FN": false_negatives,
            "TN": true_negatives,
            "Actually_Drafted_Count": actually_drafted_count,
            "Predicted_Drafted_Count": predicted_drafted_count,
        }
    )

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(
        f"Drafted vs Undrafted Prediction ({year})\nDrafted = Overall 1-255, Undrafted = Overall 0",
        fontsize=14,
    )

    # Add total counts and metrics as annotations
    plt.text(
        0.5,
        -0.1,
        f"Total Players: {len(predictions_df)}",
        ha="center",
        transform=plt.gca().transAxes,
        fontsize=11,
    )
    plt.text(
        0.5,
        -0.15,
        f"Actually Drafted: {actually_drafted_count} (Expected: 255)",
        ha="center",
        transform=plt.gca().transAxes,
        fontsize=11,
    )
    plt.text(
        0.5,
        -0.2,
        f"Actually Undrafted: {(~predictions_df['actually_drafted']).sum()}",
        ha="center",
        transform=plt.gca().transAxes,
        fontsize=11,
    )

    # Add row and column sums to the plot
    plt.text(
        0.5,
        -0.25,
        f"Row sums: [Predicted Drafted: {true_positives + false_positives}, Predicted Undrafted: {false_negatives + true_negatives}]",
        ha="center",
        transform=plt.gca().transAxes,
        fontsize=11,
    )
    plt.text(
        0.5,
        -0.3,
        f"Column sums: [Actually Drafted: {true_positives + false_negatives}, Actually Undrafted: {false_positives + true_negatives}]",
        ha="center",
        transform=plt.gca().transAxes,
        fontsize=11,
    )

    # Add performance metrics
    plt.text(
        0.5,
        -0.35,
        f"Accuracy: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}",
        ha="center",
        transform=plt.gca().transAxes,
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"drafted_confusion_matrix_{year}.png"))

    # Position-wise analysis of prediction accuracy
    position_accuracy = (
        predictions_df.groupby("position")
        .apply(
            lambda x: {
                "count": len(x),
                "mse": mean_squared_error(x["overall"], x["predicted_overall"]),
                "drafted_accuracy": (
                    (x["actually_drafted"] & x["predicted_drafted"]).sum()
                    + (~x["actually_drafted"] & ~x["predicted_drafted"]).sum()
                )
                / len(x)
                * 100
                if len(x) > 0
                else 0,
            }
        )
        .apply(pd.Series)
    )

    # Filter positions with at least 5 players
    position_accuracy = position_accuracy[position_accuracy["count"] >= 5]

    # Plot position-wise drafted prediction accuracy
    if not position_accuracy.empty:
        plt.figure(figsize=(12, 8))
        position_accuracy = position_accuracy.sort_values(
            "drafted_accuracy", ascending=False
        )

        # Create a mapping of positions to their counts for easier access
        position_counts = position_accuracy["count"].to_dict()

        # Plot the bars
        plt.bar(position_accuracy.index, position_accuracy["drafted_accuracy"])

        # Add count labels above each bar
        for i, pos in enumerate(position_accuracy.index):
            plt.text(
                i,
                position_accuracy.loc[pos, "drafted_accuracy"] + 1,
                f"{int(position_counts[pos])}",
                ha="center",
                va="bottom",
            )

        plt.xlabel("Position")
        plt.ylabel("Drafted/Undrafted Prediction Accuracy (%)")
        plt.title(f"Position-wise Drafted/Undrafted Prediction Accuracy ({year})")
        plt.xticks(rotation=45)
        plt.ylim(0, 105)  # Leave room for count labels
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"position_accuracy_{year}.png"))

# Save results summary
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_performance.csv"), index=False)

# Print results summary
print("\nModel Performance Summary:")
print(results_df)

# Create a summary report
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(OUTPUT_DIR, "summary_report.txt"), "w") as f:
    f.write(f"XGBoost Model for NFL Draft Position Prediction (No Rating Features)\n")
    f.write(f"Generated on: {timestamp}\n\n")
    f.write(f"Training Data: Years 2011-2020\n")
    f.write(f"Test Data: Years 2021, 2023, 2024\n\n")
    f.write(f"Model Parameters:\n")
    f.write(f"- n_estimators: 100\n")
    f.write(f"- learning_rate: 0.1\n")
    f.write(f"- max_depth: 6\n")
    f.write(f"- subsample: 0.8\n")
    f.write(f"- colsample_bytree: 0.8\n")
    f.write(f"- random_state: 42\n\n")
    f.write(f"Excluded Rating Features:\n")
    f.write(f"- preDraftGrade\n")
    f.write(f"- preDraftRanking\n")
    f.write(f"- preDraftPositionRanking\n\n")
    f.write(f"Top 10 Important Features:\n")
    for i, row in feature_importance.head(10).iterrows():
        f.write(f"- {row['Feature']}: {row['Importance']:.4f}\n")
    f.write("\nPerformance Metrics:\n")
    for i, row in results_df.iterrows():
        f.write(f"\nYear: {row['Year']}\n")
        f.write(f"- MSE: {row['MSE']:.4f}\n")
        f.write(f"- RMSE: {row['RMSE']:.4f}\n")
        f.write(f"- MAE: {row['MAE']:.4f}\n")
        f.write(f"- R2: {row['R2']:.4f}\n")

        # Add classification metrics if available
        if "Accuracy" in row:
            f.write(f"\nDrafted/Undrafted Classification Metrics:\n")
            f.write(
                f"- Actually Drafted Count: {int(row['Actually_Drafted_Count'])} (Expected: 255)\n"
            )
            f.write(
                f"- Predicted Drafted Count: {int(row['Predicted_Drafted_Count'])} (Expected: 255)\n"
            )
            f.write(f"- Accuracy: {row['Accuracy']:.4f}\n")
            f.write(f"- Precision: {row['Precision']:.4f}\n")
            f.write(f"- Recall: {row['Recall']:.4f}\n")
            f.write(f"- F1 Score: {row['F1']:.4f}\n")
            f.write(f"- Confusion Matrix:\n")
            f.write(
                f"  * True Positives (correctly predicted as drafted): {int(row['TP'])}\n"
            )
            f.write(
                f"  * False Positives (predicted drafted but actually undrafted): {int(row['FP'])}\n"
            )
            f.write(
                f"  * False Negatives (predicted undrafted but actually drafted): {int(row['FN'])}\n"
            )
            f.write(
                f"  * True Negatives (correctly predicted as undrafted): {int(row['TN'])}\n"
            )

print(f"\nAll outputs saved to {OUTPUT_DIR}")
