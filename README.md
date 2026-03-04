#AQI Predictions.
This repository is developed with the dedication of the team which is supposed to predict AQI for multiple cities of India(major) using regression models. Advanced technologies like mlops, mlflow, and GenAI has also been implemented to keep it different from traditional system and helps users assisting with their requirements in relevant field.

AQI Forecasting System - Modular Architecture
A comprehensive air quality index (AQI) forecasting system built with a modular architecture for better maintainability and scalability.

🚀 Quick Start
Prerequisites
pip install -r requirements.txt
Run the Complete System
# Run with user input for city
python main.py

# Or specify city directly
python main.py "Delhi"

📁 Module Details
config.py
Contains all configuration constants:

•MLflow settings
•Model hyperparameters
•Data paths and directories
•AQI risk categories
•Pollution thresholds

data_loader
Handles data operations:

•Loading preprocessed datasets
•Creating time-series train/val/test splits
•Basic EDA and statistical analysis
•Feature metadata management

feature_engineering.py
Manages feature processing:

•Cyclical encoding for time features
•Feature scaling with StandardScaler
•City-specific data preparation
•Forecast dataframe creation

model_training
Contains the ModelTrainer class:

•Training multiple ML models (LR, DT, RF, XGBoost)
•Model evaluation and comparison
•Best model selection
•City-specific model training

model_evaluation
Provides comprehensive evaluation:

•Regression metrics (RMSE, MAE, R2, MAPE)
•Forecast horizon evaluation
•AQI category accuracy
•Residual analysis and error distribution

visualization
Creates all plots and charts:

•AQI distribution histograms
•Correlation heatmaps
•Feature importance plots
•Actual vs predicted comparisons
•Risk zone overlays

mlflow_utils
Manages experiment tracking:

•MLflow run management
•Model logging and registration
•Artifact tracking
•Experiment comparison

anomaly_detection
Detects pollution spikes:

•Rolling statistics anomaly detection
•Pollutant threshold monitoring
•Static AQI threshold alerts
•Pattern analysis and reporting

model_persistence
Handles model artifacts:

•Joblib-based model serialization
•Scaler and feature metadata saving
•Model validation and loading
•Deployment-ready artifact creation

As Whole
Orchestrates the entire pipeline:

•End-to-end execution
•Error handling and logging
•User interaction
•Pipeline coordination

🔧 Usage Examples
Training Models
from model_training import ModelTrainer

trainer = ModelTrainer()
trainer.train_all_models(X_train, y_train)
metrics = trainer.evaluate_all_models(X_test, y_test)
best_model, best_score = trainer.get_best_model()

City-Specific Analysis
from feature_engineering import prepare_city_specific_data
from anomaly_detection import detect_pollution_spikes_city

# Prepare city data
city_data = prepare_city_specific_data(train_df, val_df, test_df, "Delhi", features)

# Detect anomalies
anomaly_df, summary = detect_pollution_spikes_city(city_test_df, "Delhi")

Visualization
from visualization import AQIPlotter

plotter = AQIPlotter()
plotter.plot_forecast_with_risk_zones(forecast_df, "Delhi", "XGBoost")
plotter.plot_actual_vs_predicted(y_true, y_pred, "XGBoost", "Delhi")

Model Persistence
from model_persistence import ModelPersistence

persistence = ModelPersistence()
persistence.save_best_model(model, "xgboost", scaler, features, metrics)
artifacts = persistence.load_model_artifacts("best_model")

📊 Key Features
Modular Design: Each module has a single responsibility
Time-Series Aware: Proper temporal splitting and feature engineering
Multiple Models: Linear Regression, Decision Tree, Random Forest, XGBoost
Comprehensive Evaluation: Multiple metrics and forecast horizons
Anomaly Detection: Rolling statistics and threshold-based alerts
Experiment Tracking: Full MLflow integration
Production Ready: Model serialization and deployment artifacts
Rich Visualization: Professional plots with AQI risk categories
🔄 Migration from Monolithic
The original model.py (1407 lines) has been split into:

| Module | Purpose | |--------|-------|---------| | data_loader| Data operations | | feature_engineering |Feature processing | | model_training |Model training | | model_evaluation |Evaluation metrics | | visualization |Plotting functions | | mlflow_utils |Experiment tracking | | anomaly_detection | Anomaly detection | | model_persistence |Model I/O |

Total: ~2400 lines across 10 modules

🎯 Benefits of Modular Architecture
Maintainability: Easier to modify individual components
Testability: Each module can be tested independently
Reusability: Modules can be imported and used separately
Scalability: New features can be added without affecting others
Collaboration: Multiple developers can work on different modules
Debugging: Issues can be isolated to specific modules

📈 Performance
The modular system maintains the same performance as the original while providing:

Better code organization
Improved error handling
Enhanced logging and monitoring
Easier deployment and maintenance

🤝 Contributing
When adding new features:

Identify which module should contain the functionality
Follow the existing code patterns and naming conventions
Add appropriate error handling and logging
Update docstrings and comments
Test the changes thoroughly

📝 License
This project is part of the AQI forecasting system for air quality prediction and analysis.
