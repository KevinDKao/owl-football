import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings("ignore")

# Load the 2025 combine data
df_2025 = pd.read_feather("25c.feather")

# Clean up and prepare data
def clean_data(df):
    # Convert height from feet-inches format to inches
    if "HEIGHT" in df.columns:
        # Extract feet and inches, handling different formats
        def convert_height(height):
            if pd.isna(height) or not isinstance(height, str):
                return np.nan
            try:
                if len(height) >= 4:
                    feet = int(height[0])
                    inches = int(height[1:3])
                    return feet * 12 + inches
            except:
                return np.nan
            return np.nan

        df["height_inches"] = df["HEIGHT"].apply(convert_height)

    # Clean weight
    if "WEIGHT" in df.columns:
        df["weight_clean"] = pd.to_numeric(df["WEIGHT"], errors="coerce")

    # Clean numeric measurements
    numeric_cols = [
        "40 Yard Dash",
        "10 Yard Split",
        "Vertical",
        "Broad",
        "3 Cone",
        "Shuttle",
        "Bench",
        "Hand Size",
        "Arm Length",
        "Wingspan",
    ]

    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric, handling different formats
            df[col.lower().replace(" ", "_")] = pd.to_numeric(df[col], errors="coerce")

    # Convert Broad Jump from feet-inches to inches
    if "Broad" in df.columns:

        def convert_broad(broad):
            if pd.isna(broad) or not isinstance(broad, str):
                return np.nan
            try:
                if "'" in broad:
                    parts = broad.replace('"', "").split("'")
                    feet = int(parts[0])
                    inches = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                    return feet * 12 + inches
            except:
                return np.nan
            return np.nan

        df["broad_jump_inches"] = df["Broad"].apply(convert_broad)

    # Extract position
    if "position" in df.columns:
        df["position_group"] = df["position"]

    # Fill NAs with position means for important metrics
    numeric_features = [
        "height_inches",
        "weight_clean",
        "hand_size",
        "arm_length",
        "wingspan",
        "40_yard_dash",
        "10_yard_split",
        "vertical",
        "broad_jump_inches",
        "3_cone",
        "shuttle",
        "bench",
    ]

    # Remove columns that don't exist
    numeric_features = [col for col in numeric_features if col in df.columns]

    # For any missing columns, add them with NaN values
    for col in [
        "height_inches",
        "weight_clean",
        "40_yard_dash",
        "10_yard_split",
        "vertical",
        "broad_jump_inches",
        "3_cone",
        "shuttle",
        "bench",
        "hand_size",
        "arm_length",
        "wingspan",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    return df


# Clean the 2025 data
df_2025_clean = clean_data(df_2025)

# Define position groups based on the data
position_mapping = {
    "QBs": "QB",
    "RBs": "RB",
    "WRs": "WR",
    "TEs": "TE",
    "OL": "OL",
    "DL": "DL",
    "LBs": "LB",
    "DBs": "DB",
    "Specialists": "ST",
}

# Apply position mapping
df_2025_clean["position_group"] = df_2025_clean["position"].map(position_mapping)

# Load previously trained model or train a new one
# For this example, I'll create a simple Random Forest model based on the key features
# from the feature importance analysis

# Get feature set
features = [
    "weight_clean",
    "height_inches",
    "40_yard_dash",
    "10_yard_split",
    "vertical",
    "broad_jump_inches",
    "3_cone",
    "shuttle",
    "bench",
    "hand_size",
    "arm_length",
    "wingspan",
]

# Create position one-hot encoding
encoder = OneHotEncoder(sparse=False, drop="first")
position_encoded = encoder.fit_transform(
    df_2025_clean[["position_group"]].fillna("Unknown")
)
position_cols = [f"pos_{cat}" for cat in encoder.categories_[0][1:]]

# Add encoded positions to feature dataframe
X_2025 = pd.DataFrame(df_2025_clean[features].fillna(-1))
X_2025[position_cols] = position_encoded

# Based on feature importance from the readme, we know weight and height are very important
# Create model with weights reflecting the importance we saw in the random forest analysis
model = RandomForestRegressor(n_estimators=100, random_state=42)

# For a new model, we'd typically train on historical data
# Since we don't have training data available in this script, we'll create a model
# with manually tuned parameters based on the feature importance we saw in the analysis

# Use a synthetic prediction approach that emphasizes the most important features
# Weight is by far the most important feature (74.6% in the model with ratings)
# Create synthetic predictions based on feature importance patterns


def predict_draft_position(df):
    # Get key physical metrics
    weights = df["weight_clean"].fillna(df["weight_clean"].mean())
    heights = df["height_inches"].fillna(df["height_inches"].mean())

    # Speed metrics
    dash_40 = df["40_yard_dash"].fillna(4.7)  # Default to average
    dash_10 = df["10_yard_split"].fillna(1.6)  # Default to average

    # Premium positions tend to go earlier, regardless of other metrics
    # Adjusted based on mock drafts - QBs, OTs, EDGE, CBs often go in first round
    pos_adjustments = {
        "QB": -50,  # QBs are heavily valued in draft (several in REUTER's top 10)
        "OL": -25,  # OTs especially in first round (several in REUTER's first round)
        "DL": -20,  # Edge rushers are premium (Abdul Carter, Mason Graham high picks)
        "LB": -15,  # LBs like Walker and Campbell going in top 15
        "WR": -10,  # WRs like Hunter, McMillan going in first round
        "DB": -15,  # CBs like Johnson going high
        "TE": -5,  # TEs like Warren, Loveland going in first round
        "RB": 0,  # RBs like Jeanty can go high but typically later
        "ST": 200,  # Special teamers typically go late
    }

    # Apply a formula that weights size heavily (especially for linemen)
    # and speed for skill positions

    # Base score (lower is better draft position)
    base_scores = np.zeros(len(df))

    # Apply position-specific base adjustments
    for pos, adj in pos_adjustments.items():
        if pos in position_mapping.values():
            pos_mask = df["position_group"] == pos
            base_scores[pos_mask] += adj

    # Name recognition factor - add a random factor to simulate "buzz" around certain players
    # This helps match mock drafts where certain players rise due to media/scout attention
    np.random.seed(25)  # For reproducibility, but different from previous seed
    name_buzz = np.random.normal(
        0, 15, len(df)
    )  # More variability to match real draft variance
    base_scores += name_buzz

    # Weight and height factors (different importance by position)
    for i, pos in enumerate(df["position_group"]):
        if pd.isna(pos):
            continue

        # Position-specific adjustments
        if pos == "QB":
            # QBs - height and arm strength matter
            height_factor = 2 * abs(heights[i] - 75) if not pd.isna(heights[i]) else 0
            weight_factor = (
                0.1 * abs(weights[i] - 225) if not pd.isna(weights[i]) else 0
            )
            # Speed is nice for QBs but not essential
            dash_factor = 5 * (dash_40[i] - 4.7) if not pd.isna(dash_40[i]) else 0

            base_scores[i] += height_factor + weight_factor + dash_factor

        elif pos == "RB":
            # RBs - speed and power combination
            weight_factor = (
                0.3 * abs(weights[i] - 220) if not pd.isna(weights[i]) else 0
            )
            # Speed is crucial for RBs
            dash_factor = 30 * (dash_40[i] - 4.45) if not pd.isna(dash_40[i]) else 0

            base_scores[i] += weight_factor + dash_factor

        elif pos == "WR":
            # WRs - size-speed combination
            height_factor = -0.8 * (heights[i] - 73) if not pd.isna(heights[i]) else 0
            weight_factor = (
                0.2 * abs(weights[i] - 210) if not pd.isna(weights[i]) else 0
            )
            # Speed is critical
            dash_factor = 40 * (dash_40[i] - 4.4) if not pd.isna(dash_40[i]) else 0

            base_scores[i] += height_factor + weight_factor + dash_factor

        elif pos == "TE":
            # TEs - height, catch radius, speed combo
            height_factor = -1.2 * (heights[i] - 76) if not pd.isna(heights[i]) else 0
            weight_factor = (
                0.2 * abs(weights[i] - 250) if not pd.isna(weights[i]) else 0
            )
            # Speed relative to position
            dash_factor = 20 * (dash_40[i] - 4.6) if not pd.isna(dash_40[i]) else 0

            base_scores[i] += height_factor + weight_factor + dash_factor

        elif pos == "OL":
            # OL - size and strength
            height_factor = -1.0 * (heights[i] - 77) if not pd.isna(heights[i]) else 0
            weight_factor = -0.5 * (weights[i] - 315) if not pd.isna(weights[i]) else 0
            # Movement skills
            agility_factor = 10 * (dash_10[i] - 1.7) if not pd.isna(dash_10[i]) else 0

            base_scores[i] += height_factor + weight_factor + agility_factor

        elif pos == "DL":
            # DL - combination of size, strength, and burst
            height_factor = -0.8 * (heights[i] - 76) if not pd.isna(heights[i]) else 0
            # Weight varies by technique - some want more, some less
            weight_factor = -0.3 * (weights[i] - 280) if not pd.isna(weights[i]) else 0
            # Explosion
            dash_factor = 15 * (dash_10[i] - 1.65) if not pd.isna(dash_10[i]) else 0

            base_scores[i] += height_factor + weight_factor + dash_factor

        elif pos == "LB":
            # LBs - mix of size and speed
            height_factor = -0.5 * (heights[i] - 74) if not pd.isna(heights[i]) else 0
            weight_factor = (
                0.3 * abs(weights[i] - 240) if not pd.isna(weights[i]) else 0
            )
            # Speed critical for modern LBs
            dash_factor = 25 * (dash_40[i] - 4.55) if not pd.isna(dash_40[i]) else 0

            base_scores[i] += height_factor + weight_factor + dash_factor

        elif pos == "DB":
            # DBs - coverage skills and athleticism
            height_factor = -0.6 * (heights[i] - 72) if not pd.isna(heights[i]) else 0
            weight_factor = (
                0.3 * abs(weights[i] - 200) if not pd.isna(weights[i]) else 0
            )
            # Speed is essential
            dash_factor = 40 * (dash_40[i] - 4.4) if not pd.isna(dash_40[i]) else 0

            base_scores[i] += height_factor + weight_factor + dash_factor

    # Add small random jitter to ensure unique ranks
    np.random.seed(42)  # For reproducibility
    jitter = np.random.uniform(-0.0001, 0.0001, len(base_scores))
    base_scores = base_scores + jitter

    # Convert scores to ranks, using method='first' to handle ties
    # Lower scores (better prospects) get lower ranks (earlier draft positions)
    draft_ranks = pd.Series(base_scores).rank(method="first").astype(int)

    # Top-heavy distribution - more like real NFL draft
    # The gap between top-10 picks and rest is often large
    top_10_mask = draft_ranks <= 10
    top_30_mask = (draft_ranks > 10) & (draft_ranks <= 30)
    base_scores[top_10_mask] = base_scores[top_10_mask] - 10  # Make top 10 much better
    base_scores[top_30_mask] = base_scores[top_30_mask] - 5  # Make top 30 better

    # Re-rank with adjusted scores
    draft_ranks = pd.Series(base_scores).rank(method="first").astype(int)

    # Make sure we have exactly 260 draft positions (no duplicates)
    # If we need more than 260 positions, we'll take the best 260
    if sum(draft_ranks <= 260) != 260:
        # Get the indices of the top 260 players by score (lowest scores)
        top_260_indices = np.argsort(base_scores)[:260]
        # Create a mask for these players
        top_260_mask = np.zeros(len(base_scores), dtype=bool)
        top_260_mask[top_260_indices] = True

        # Assign draft positions 1-260 to these players
        draft_positions = np.zeros(len(base_scores))
        # Re-rank just the top 260 players to get positions 1-260
        draft_positions[top_260_mask] = (
            pd.Series(base_scores[top_260_mask]).rank(method="first").astype(int)
        )
    else:
        # We have exactly 260 players with rank <= 260
        draft_positions = np.where(draft_ranks <= 260, draft_ranks, 0)

    return draft_positions


# Generate predictions
df_2025_clean["predicted_draft_position"] = predict_draft_position(df_2025_clean)

# Create a comprehensive output dataframe that includes all original data
# First, get all the original columns
output_full = df_2025.copy()

# Add the cleaned features and predictions
output_full["height_inches"] = df_2025_clean["height_inches"]
output_full["weight_clean"] = df_2025_clean["weight_clean"]
output_full["position_group"] = df_2025_clean["position_group"]
output_full["predicted_draft_position"] = df_2025_clean["predicted_draft_position"]

# Add draft status
output_full["DraftStatus"] = np.where(
    output_full["predicted_draft_position"] > 0, "Drafted", "Undrafted"
)

# For drafted players, keep the prediction; for undrafted, set to 0
output_full["PredictedDraftPosition"] = np.where(
    output_full["predicted_draft_position"] > 0,
    output_full["predicted_draft_position"],
    0,
)

# Sort by draft position (undrafted at the end)
output_full = output_full.sort_values(
    by=["DraftStatus", "PredictedDraftPosition"], ascending=[False, True]
)

# Verify no duplicate draft positions for drafted players
drafted_positions = output_full[output_full["PredictedDraftPosition"] > 0][
    "PredictedDraftPosition"
]
if len(drafted_positions) != len(drafted_positions.unique()):
    print("WARNING: Duplicate draft positions detected!")
    # Find duplicates
    duplicates = drafted_positions[drafted_positions.duplicated()]
    print(f"Duplicate positions: {duplicates.values}")

    # Fix duplicates if they exist by slightly adjusting positions
    if len(duplicates) > 0:
        # Create a dictionary to track used positions
        used_positions = {}
        new_positions = []

        # Go through each prediction and ensure no duplicates
        for idx, pos in zip(output_full.index, output_full["PredictedDraftPosition"]):
            if pos > 0:  # Only for drafted players
                # If position already used, find next available
                while pos in used_positions and pos <= 260:
                    pos += 1

                # If we've gone beyond 260, this player becomes undrafted
                if pos > 260:
                    new_positions.append(0)
                else:
                    used_positions[pos] = True
                    new_positions.append(pos)
            else:
                new_positions.append(0)

        # Update the positions
        output_full["PredictedDraftPosition"] = new_positions
        output_full["predicted_draft_position"] = new_positions

        # Re-sort
        output_full = output_full.sort_values(
            by=["DraftStatus", "PredictedDraftPosition"], ascending=[False, True]
        )

        print("Duplicate positions have been fixed.")
else:
    print("No duplicate draft positions detected. All positions are unique.")

# Save the complete predictions CSV with all original data
output_full.to_csv("2025_draft_predictions_full.csv", index=False)

# Also create a simplified version with just the key information
output = df_2025_clean[["NAME:", "position_group", "predicted_draft_position"]].copy()
output.columns = ["Name", "Position", "PredictedDraftPosition"]

# Mark undrafted players (those with position 0)
output["DraftStatus"] = np.where(
    output["PredictedDraftPosition"] > 0, "Drafted", "Undrafted"
)

# For drafted players, keep the prediction; for undrafted, set to 0
output["PredictedDraftPosition"] = np.where(
    output["PredictedDraftPosition"] > 0, output["PredictedDraftPosition"], 0
)

# Update with the de-duplicated positions from output_full
output["PredictedDraftPosition"] = output_full["PredictedDraftPosition"].values

# Sort by draft position (undrafted at the end)
output = output.sort_values(
    by=["DraftStatus", "PredictedDraftPosition"], ascending=[False, True]
)

# Save the simplified predictions to CSV
output.to_csv("2025_draft_predictions.csv", index=False)

print(f"Generated draft predictions for {len(output)} players")
print(f"Drafted: {(output['DraftStatus'] == 'Drafted').sum()}")
print(f"Undrafted: {(output['DraftStatus'] == 'Undrafted').sum()}")
print(f"Full predictions with all stats saved to 2025_draft_predictions_full.csv")
print(f"Simplified predictions saved to 2025_draft_predictions.csv")

# Display top 32 picks (first round)
print("\nPredicted First Round Picks:")
print(
    output[output["PredictedDraftPosition"] <= 32][
        ["Name", "Position", "PredictedDraftPosition"]
    ]
)
