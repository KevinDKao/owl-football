# Models Directory

This directory stores machine learning models generated from the notebooks in the `notebooks/kkao_models` directory. The models are used to predict NFL draft outcomes for college football players.

## Directory Purpose

The `models` directory serves as a centralized storage location for trained machine learning models that can be loaded and used for predictions in the application. By keeping models separate from the notebooks that generate them, we maintain a clean separation between model development and model deployment.

## Available Models

The following models can be generated from the notebooks:

### XGBoost Models
- Standard XGBoost model (with player ratings)
- XGBoost model without player ratings

### Random Forest Models
- Standard Random Forest model (with player ratings)
- Random Forest model without player ratings

### AutoGluon Models
- Ensemble models created using AutoGluon's AutoML framework

## How to Generate Models

To generate the models and store them in this directory, run the following notebooks:

1. **XGBoost Models**:
   - Run `notebooks/kkao_models/xgboost/xgboost_model.py` for the standard model
   - Run `notebooks/kkao_models/xgboost/xgboost_model_no_rating.py` for the model without ratings

2. **Random Forest Models**:
   - Run `notebooks/kkao_models/random_forest/model.py` for the standard model
   - Run `notebooks/kkao_models/random_forest/model_no_rating.py` for the model without ratings

3. **AutoGluon Models**:
   - Run `notebooks/kkao_models/autogluon/autogluon.ipynb` to generate AutoGluon models

## Model Data

The models are trained using historical college football player data stored as feather files in the `notebooks/kkao_models/feathers` directory. These files contain player statistics and draft outcomes from 2011 to 2024.

## Model Comparison

After generating the models, you can compare their performance by running:
```
python notebooks/kkao_models/comparison.py
```

This will generate comparison metrics and visualizations to help determine which model performs best for predicting draft outcomes.

## Model Usage

Once generated, these models can be loaded in the application to:
1. Predict whether a player will be drafted
2. Estimate a player's draft position
3. Identify key factors influencing draft status
4. Compare players based on their predicted draft outcomes

## Note on Model Storage

This directory is intentionally kept empty in the repository. Models should be generated locally by running the appropriate notebooks, as model files can be large and are specific to the local environment in which they were trained.
