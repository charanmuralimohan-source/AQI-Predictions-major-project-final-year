import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import json
import mlflow
import mlflow.sklearn
import os
from mlflow.models.signature import infer_signature
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor



GLOBAL_MODEL_METRICS = {}#for registering best model to mlflow automatically
GLOBAL_MODEL_OBJECTS = {}  #for registering best model to mlflow automatically


matplotlib.use("Agg")   # non-GUI backend



def safe_log_artifact(path):
    if os.path.exists(path):
        mlflow.log_artifact(path)
    else:
        print(f"Artifact not found, skipping MLflow log: {path}")



#mlflow setup------------------------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("AQI_Forecasting_System")

#mlflow.end_run()  #REQUIRED safety reset

# HARD RESET (safe)
if mlflow.active_run():
    mlflow.end_run()

# PARENT RUN (ONLY ONE)
parent_run = mlflow.start_run(run_name="AQI_Global_Training_Run")


#to select best model by mlflow
def evaluate_global_model(name, model, X_test, y_test):
    preds = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae  = float(mean_absolute_error(y_test, preds))
    r2   = float(r2_score(y_test, preds))

    print(f"\n{name} (GLOBAL)")
    print(f"RMSE: {rmse}")
    print(f"MAE : {mae}")
    print(f"R2  : {r2}")

    GLOBAL_MODEL_METRICS[name] = rmse
    GLOBAL_MODEL_OBJECTS[name] = model

    return rmse, mae, r2



# LOAD PREPROCESSED DATASET
df = pd.read_csv("notebook/aqi_cleaned_processed.csv")

#EDA on dataset
print("\nData Types & Null Info:")
print(df.info())

# EDA: MISSING VALUES

missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_%": missing_pct
})

print("\nMissing Value Summary:")
print(missing_df[missing_df["Missing_Count"] > 0])


EDA_DIR = os.path.join("src", "components", "EDA on dataset")
os.makedirs(EDA_DIR, exist_ok=True)


#imputers pipeline
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
#  AQI ANALYSIS(histogram)

print("\nAQI Statistics:")
print(df["aqi"].describe())

plt.figure(figsize=(8, 5))
plt.hist(df["aqi"], bins=50)
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.title("AQI Distribution")
plt.tight_layout()

# Save to parent directory
plot_path = os.path.join(EDA_DIR, "aqi_distribution.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"AQI distribution plot saved as: {plot_path}")


# SCATTER PLOT: PM2.5 vs AQI

plt.figure(figsize=(8, 5))
plt.scatter(
    df["pm2_5"],
    df["aqi"],
    alpha=0.3,
    s=10
)

plt.xlabel("PM2.5 (µg/m³)")
plt.ylabel("AQI")
plt.title("Scatter Plot: PM2.5 vs AQI")
plt.grid(True)
plt.tight_layout()

scatter_path = os.path.join(EDA_DIR, "aqi_vs_pm2_5_scatter.png")
plt.savefig(scatter_path, dpi=300)
plt.close()

print(f"Scatter plot saved as: {scatter_path}")



# ================= STATISTICAL INSIGHTS =================

STATS_DIR = os.path.join(EDA_DIR, "statistical_insights")
os.makedirs(STATS_DIR, exist_ok=True)


# Descriptive statistics for numeric columns
numeric_cols = [
    "aqi", "pm2_5", "pm10", "no2", "so2", "co", "o3",
    "temperature", "humidity", "wind_speed", "rainfall"
]

stats_df = df[numeric_cols].describe().T
stats_df["IQR"] = stats_df["75%"] - stats_df["25%"]

stats_path = os.path.join(STATS_DIR, "descriptive_statistics.csv")
stats_df.to_csv(stats_path)

print("Descriptive statistics saved:", stats_path)




# Correlation with AQI
corr_df = df[numeric_cols].corr()

aqi_corr = corr_df["aqi"].sort_values(ascending=False)

corr_path = os.path.join(STATS_DIR, "aqi_correlation.csv")
aqi_corr.to_csv(corr_path)

print("\nCorrelation of pollutants with AQI:")
print(aqi_corr)




#heatmap for correlation
import seaborn as sns
plt.figure(figsize=(12, 8))

sns.heatmap(
    corr_df,
    annot=False,
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("Correlation Heatmap of Air Quality Features")
plt.tight_layout()

heatmap_path = os.path.join(STATS_DIR, "correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()

print(f"Correlation heatmap saved at: {heatmap_path}")

# Convert date column (safety)
df["date"] = pd.to_datetime(df["date"], errors="coerce")





# ---------- DATE FEATURE ENGINEERING ----------
df["dayofweek"] = df["date"].dt.dayofweek      # 0–6
df["month"] = df["date"].dt.month              # 1–12

# Cyclical encoding
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

#TIME FEATURE ENGINEERED


df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
df = df.sort_values(["city", "datetime"])



pollutants = ["pm2_5","pm10","no2","so2","co","o3"]

HORIZON = 6

for col in pollutants:
    df[f"{col}_lag_{HORIZON}"] = df.groupby("city")[col].shift(HORIZON)

df["aqi_future"] = df.groupby("city")["aqi"].shift(-HORIZON)

df = df.dropna(subset=[
    *(f"{col}_lag_{HORIZON}" for col in pollutants),
    "aqi_future"
])

X = df[
    [f"{col}_lag_{HORIZON}" for col in pollutants] +
    ["temperature","humidity","wind_speed","rainfall",
     "dow_sin","dow_cos","month_sin","month_cos"]
]


y = df["aqi_future"]





# TIME-SERIES SPLIT PER CITY
train_list = []
val_list = []
test_list = []

for city in df["city"].unique():
    city_df = df[df["city"] == city].sort_values(by=["date", "hour"])
    
    n = len(city_df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    
    train_list.append(city_df.iloc[:train_end])
    val_list.append(city_df.iloc[train_end:val_end])
    test_list.append(city_df.iloc[val_end:])

# CONCAT FINAL SPLITS
train_df = pd.concat(train_list)
val_df = pd.concat(val_list)
test_df = pd.concat(test_list)

# SAVE SPLITS
train_df.to_csv("aqi_train.csv", index=False)
val_df.to_csv("aqi_validation.csv", index=False)
test_df.to_csv("aqi_test.csv", index=False)


# VERIFY SPLIT
print("Training records:", len(train_df))
print("Validation records:", len(val_df))
print("Testing records:", len(test_df))

#----------------------------------------------------------------------------




# LOAD DATASETS
train_df = pd.read_csv("aqi_train.csv")
val_df   = pd.read_csv("aqi_validation.csv")
test_df  = pd.read_csv("aqi_test.csv")

# Ensure datetime format
for df in [train_df, val_df, test_df]:
    df["date"] = pd.to_datetime(df["date"])

# STEP 1: FEATURE–TARGET SEPARATION

TARGET = "aqi"

FEATURES = [
    "pm2_5", "pm10", "no2", "so2", "co", "o3",
    "temperature", "humidity", "wind_speed", "rainfall",
    "hour",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos"
]




# STEP 1: FEATURE–TARGET SEPARATION
X_train = train_df[FEATURES].copy()
y_train = train_df[TARGET]

X_val   = val_df[FEATURES].copy()
y_val   = val_df[TARGET]

X_test  = test_df[FEATURES].copy()
y_test  = test_df[TARGET]

# Fit pipeline
pipeline.fit(X_train, y_train)

# Predict AFTER X_test exists
y_test_pred = pipeline.predict(X_test)



# STEP 2: TIME FEATURE ENCODING (CYCLICAL)

def encode_hour(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df.drop(columns=["hour"], inplace=True)
    return df

X_train = encode_hour(X_train)
X_val   = encode_hour(X_val)
X_test  = encode_hour(X_test)

# STEP 3: FEATURE SCALING
scaler = StandardScaler()

# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform validation and test sets
X_val_scaled  = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# OPTIONAL: CONVERT BACK TO DATAFRAMES
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled   = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled, columns=X_test.columns)

#create signature for mlflow
signature = infer_signature(X_train_scaled, y_train)

# VERIFICATION
print("Training shape:", X_train_scaled.shape)
print("Validation shape:", X_val_scaled.shape)
print("Testing shape:", X_test_scaled.shape)


X_train_scaled.to_csv("X_train_scaled.csv", index=False)
X_val_scaled.to_csv("X_val_scaled.csv", index=False)
X_test_scaled.to_csv("X_test_scaled.csv", index=False)



#------------------------add metadata--------------------------------------

metadata = {
    "features": list(X_train_scaled.columns),
    "scaler": "StandardScaler",
    "time_features": ["hour_sin", "hour_cos"],
    "target": "aqi"
}

FEATURE_STORE_DIR = "feature_store"
os.makedirs(FEATURE_STORE_DIR, exist_ok=True)

with open("feature_store/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
print("Feature store metadata saved successfully")

#----------------------------------------------------------------------------

# TRAIN LINEAR REGRESSION
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)


# Ask user for city
user_city = input("Enter city name for AQI prediction: ").strip()


# : TIME SERIES

# Case-insensitive filtering
city_ts = (
    df[df["city"].str.lower() == user_city.lower()]
      .sort_values(["date", "hour"])
)

# Safety check
if city_ts.empty:
    raise ValueError(f"No data found for city: {user_city}")

# Plot AQI time series
plt.figure(figsize=(14, 5))
plt.plot(city_ts["aqi"].values)
plt.title(f"AQI Time Series – {user_city.title()}")
plt.xlabel("Time Index")
plt.ylabel("AQI")
plt.grid(True)
plt.tight_layout()

# Safe filename (spaces → underscores)
safe_city_name = user_city.lower().replace(" ", "_")
plot_path = f"aqi_time_series_{safe_city_name}.png"

plt.savefig(plot_path, dpi=300)
plt.close()

print(f"AQI time-series plot saved as: {plot_path}")


# Filter training data for selected city
city_train_df = train_df[train_df["city"].str.lower() == user_city.lower()]

if city_train_df.empty:
    raise ValueError(f"No training data found for city: {user_city}")


# Feature–target split for city
X_city_train = city_train_df[FEATURES].copy()
y_city_train = city_train_df["aqi"]

# Encode hour
X_city_train = encode_hour(X_city_train)

# Scale using SAME scaler
X_city_train_scaled = scaler.transform(X_city_train)
X_city_train_scaled = pd.DataFrame(
    X_city_train_scaled,
    columns=X_train_scaled.columns
)


# Train city-specific LR model
lr_city_model = LinearRegression()
lr_city_model.fit(X_city_train_scaled, y_city_train)


lr_city_coefficients = pd.DataFrame({
    "Feature": X_city_train_scaled.columns,
    "Coefficient": lr_city_model.coef_
})

lr_city_coefficients["Abs_Coefficient"] = lr_city_coefficients["Coefficient"].abs()

lr_city_coefficients["Importance_%"] = (
    lr_city_coefficients["Abs_Coefficient"] /
    lr_city_coefficients["Abs_Coefficient"].sum()
) * 100

# Sort for plotting
lr_city_coefficients = lr_city_coefficients.sort_values(
    "Importance_%", ascending=True
)



#mlflow for linear regression(global model)
with mlflow.start_run(
    run_name="LinearRegression_Global",
    nested=True
):
    mlflow.log_param("model_type", "LinearRegression")

    y_test_pred = lr_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae  = mean_absolute_error(y_test, y_test_pred)
    r2   = lr_model.score(X_test_scaled, y_test)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(
        sk_model=lr_model,
        name="linear_regression_model",
        signature=signature
    )



    

#feature importance for selected city
print(lr_city_coefficients)


plt.figure(figsize=(10, 6))
plt.barh(
    lr_city_coefficients["Feature"],
    lr_city_coefficients["Importance_%"]
)

plt.xlabel("Importance (%) [Log Scale]")
plt.title(f"Feature Influence on AQI – {user_city} (Linear Regression)")
plt.xscale("log")
plt.tight_layout()

plt.savefig(
    f"feature_importance_{user_city}_linear_regression.png",
    dpi=300
)
plt.close()

print(f"Feature importance plot saved for city: {user_city}")


# Log artifacts
artifact_path = f"feature_importance_{user_city}_linear_regression.png"

safe_log_artifact(f"pollution_spikes_{user_city}.png")
safe_log_artifact(f"feature_importance_{user_city}_linear_regression.png")


# ---------------- CITY-SPECIFIC METEOROLOGICAL INFLUENCE ----------------

met_features = ["temperature", "humidity", "wind_speed", "rainfall"]

city_met_influence = lr_city_coefficients[
    lr_city_coefficients["Feature"].isin(met_features)
].sort_values("Importance_%", ascending=False)

city_met_total = city_met_influence["Importance_%"].sum()

print(f"\nMeteorological Feature Influence for city: {user_city}")
print(city_met_influence)

print(
    f"\nTotal Meteorological Contribution for {user_city}: "
    f"{city_met_total:.2f}%"
)



plt.figure(figsize=(8, 5))

plt.barh(
    city_met_influence["Feature"],
    city_met_influence["Importance_%"]
)

plt.xlabel("Importance (%)")
plt.title(f"Meteorological Influence on AQI - {user_city}")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    f"meteorological_influence_{user_city}.png",
    dpi=300
)
plt.close()

print(f"Meteorological influence plot saved for city: {user_city}")



# PREDICTIONS data
y_train_pred = lr_model.predict(X_train_scaled)
y_test_pred  = lr_model.predict(X_test_scaled)

# EVALUATION of lr model
lr_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))


lr_mae  = mean_absolute_error(y_test, y_test_pred)
lr_r2 = r2_score(y_test, y_test_pred)



print("Linear Regression Performance")
print("RMSE:", lr_rmse)
print("MAE :", lr_mae)
print("lr_R2:", lr_r2)


#lr_model's(global) metrics and objects to be tracked in mlflow
evaluate_global_model("LinearRegression", lr_model, X_test_scaled, y_test)



# Predict last 72 hours using LR
y_pred_lr = lr_model.predict(X_test_scaled)



# Last 24, 48, 72 hours
lr_24 = y_pred_lr[-24:]
lr_48 = y_pred_lr[-48:]
lr_72 = y_pred_lr[-72:]

y_true_24 = y_test.values[-24:]
y_true_48 = y_test.values[-48:]
y_true_72 = y_test.values[-72:]

print("\nLinear Regression Forecast Performance")

print("24 Hour RMSE:", np.sqrt(mean_squared_error(y_true_24, lr_24)))
print("24 Hour MAE :", mean_absolute_error(y_true_24, lr_24))

print("48 Hour RMSE:", np.sqrt(mean_squared_error(y_true_48, lr_48)))
print("48 Hour MAE :", mean_absolute_error(y_true_48, lr_48))





print("72 Hour RMSE:", np.sqrt(mean_squared_error(y_true_72, lr_72)))
print("72 Hour MAE :", mean_absolute_error(y_true_72, lr_72))



#City-Specific Runs(lr model)

with mlflow.start_run(run_name=f"City_{user_city}_LinearRegression", nested=True):


    mlflow.log_param("city", user_city)
    mlflow.log_param("model_type", "LinearRegression")

    mlflow.log_metric("rmse", lr_rmse)
    mlflow.log_metric("mae", lr_mae)

    mlflow.sklearn.log_model(
    sk_model=lr_city_model,
    name="city_linear_regression_model",
    signature=signature
    )


    
    
# Create forecast dataframe using Linear Regression predictions
forecast_df = pd.DataFrame({
    "Hour": np.arange(1, 73),
    "Predicted_AQI": np.concatenate([
        lr_24,
        lr_48[24:],   # hours 25–48
        lr_72[48:]    # hours 49–72
    ])
})

print(forecast_df.head(10))
print(forecast_df.tail(10))

# Filter test data for selected city
city_test_df = test_df[test_df["city"].str.lower() == user_city.lower()]

if city_test_df.empty:
    raise ValueError(f"No data found for city: {user_city}")



# Filter training data for selected city
city_train_df = train_df[
    train_df["city"].str.lower() == user_city.lower()
]

if city_train_df.empty:
    raise ValueError(f"No training data found for city: {user_city}")



# Extract features and target for selected city
X_city = city_test_df[FEATURES].copy()
y_city = city_test_df["aqi"].values

# Encode hour cyclically (same logic as training)
X_city = encode_hour(X_city)

# Scale using the trained scaler
X_city_scaled = scaler.transform(X_city)

# Convert to DataFrame to preserve feature order
X_city_scaled = pd.DataFrame(
    X_city_scaled,
    columns=X_train_scaled.columns
)

# Predict AQI for the selected city
y_city_pred = lr_model.predict(X_city_scaled)

# Take last 72 hours
city_72_pred = y_city_pred[-72:]
city_72_true = y_city[-72:]

# Create forecast dataframe
forecast_df = pd.DataFrame({
    "Hour": np.arange(1, 73),
    "Predicted_AQI": city_72_pred
})


plt.figure(figsize=(12, 5))
plt.plot(
    forecast_df["Hour"],
    forecast_df["Predicted_AQI"],
    marker="o",
    linewidth=2
)

plt.xlabel("Forecast Hour")
plt.ylabel("Predicted AQI")
plt.title(f"72-Hour AQI Forecast - {user_city}")

plt.grid(True)
plt.tight_layout()

plt.savefig(f"aqi_72_hour_forecast_{user_city}.png", dpi=300)
plt.close()

print(f"72-hour AQI forecast graph saved for city: {user_city}")

safe_log_artifact(f"meteorological_influence_{user_city}.png")


#actual v/s predicted aqi


hours = np.arange(1, 73)

plt.figure(figsize=(14, 6))

# ---------- AQI CATEGORY BACKGROUND ----------
plt.axhspan(0, 50, color="green", alpha=0.12, label="Good (0-50)")
plt.axhspan(50, 100, color="lime", alpha=0.12, label="Satisfactory (51-100)")
plt.axhspan(100, 200, color="orange", alpha=0.12, label="Moderate (101-200)")
plt.axhspan(200, 300, color="red", alpha=0.12, label="Poor (201-300)")
plt.axhspan(300, 400, color="purple", alpha=0.12, label="Very Poor (301-400)")
plt.axhspan(400, 500, color="brown", alpha=0.12, label="Severe (401-500)")

# ---------- ACTUAL AQI ----------
plt.plot(
    hours,
    city_72_true,
    linestyle="--",
    linewidth=2,
    marker="o",
    markersize=5,
    color="black",
    label="Actual AQI"
)

# ---------- PREDICTED AQI ----------
plt.plot(
    hours,
    city_72_pred,
    linestyle="-",
    linewidth=2,
    marker="x",
    markersize=6,
    color="blue",
    label="Predicted AQI"
)

plt.xlabel("Hour")
plt.ylabel("AQI")
plt.title(f"Actual vs Predicted AQI (72 Hours) with Risk Levels - {user_city}")

plt.legend(loc="upper right", ncol=2)
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig(
    f"actual_vs_predicted_aqi_72hr_risk_{user_city}.png",
    dpi=300
)
plt.close()
safe_log_artifact(f"actual_vs_predicted_aqi_72hr_risk_{user_city}.png")


# ================= DECISION TREE REGRESSION =================

dt_model = DecisionTreeRegressor(
    max_depth=12,
    min_samples_leaf=20,
    random_state=42
)

# Train
dt_model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred_dt = dt_model.predict(X_test_scaled)

# Evaluation
dt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_dt))
dt_mae  = mean_absolute_error(y_test, y_pred_dt)
dt_r2   = r2_score(y_test, y_pred_dt)

print("\nDecision Tree Performance (Global)")
print("RMSE:", dt_rmse)
print("MAE :", dt_mae)
print("R2  :", dt_r2)

# Track for best-model selection
evaluate_global_model("DecisionTree", dt_model, X_test_scaled, y_test)


# ---------------- MLflow: Decision Tree ----------------

with mlflow.start_run(
    run_name="DecisionTree_Global",
    nested=True
):
    mlflow.log_param("model_type", "DecisionTreeRegressor")
    mlflow.log_param("max_depth", 12)
    mlflow.log_param("min_samples_leaf", 20)

    mlflow.log_metric("rmse", dt_rmse)
    mlflow.log_metric("mae", dt_mae)
    mlflow.log_metric("r2", dt_r2)

    mlflow.sklearn.log_model(
        sk_model=dt_model,
        name="decision_tree_model",
        signature=infer_signature(X_train_scaled, dt_model.predict(X_train_scaled))
    )



# ---------------- CITY-SPECIFIC RANDOM FOREST PREDICTION ----------------

# Filter test data for selected city
city_test_df = test_df[
    test_df["city"].str.lower() == user_city.lower()
]



if city_test_df.empty:
    raise ValueError(f"No test data found for city: {user_city}")

# Prepare features for selected city
X_city_test = city_test_df[FEATURES].copy()
y_city_test = city_test_df["aqi"].values

# Encode hour (same as training)
X_city_test = encode_hour(X_city_test)

# Scale using trained scaler
X_city_test_scaled = scaler.transform(X_city_test)
X_city_test_scaled = pd.DataFrame(
    X_city_test_scaled,
    columns=X_train_scaled.columns
)
# ---------------- TRAIN RANDOM FOREST (GLOBAL TRAINING) ----------------

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)


# ---------------- GLOBAL EVALUATION (FOR MODEL SELECTION) ----------------

rf_global_rmse, rf_global_mae, rf_global_r2 = evaluate_global_model(
    "RandomForest",
    rf_model,
    X_test_scaled,
    y_test
)


# ---------------- CITY-SPECIFIC PREDICTION ----------------

y_city_pred_rf = rf_model.predict(X_city_test_scaled)

print(
    "\nRaw array of predicted AQI values "
    f"for city {user_city} (Random Forest):"
)
print(y_city_pred_rf[:10])


# ---------------- CITY-SPECIFIC EVALUATION ----------------

rf_city_rmse = np.sqrt(mean_squared_error(y_city_test, y_city_pred_rf))
rf_city_mae  = mean_absolute_error(y_city_test, y_city_pred_rf)

print("\nRandom Forest Performance (City-Specific)")
print("RMSE:", rf_city_rmse)
print("MAE :", rf_city_mae)


# ---------------- MLflow: CITY RUN (NO GLOBAL METRIC NAME) ----------------

with mlflow.start_run(
    run_name=f"City_{user_city}_RandomForest",
    nested=True
):
    mlflow.set_tag("scope", "city")
    mlflow.log_param("city", user_city)
    mlflow.log_param("model_type", "RandomForest")

    mlflow.log_metric("city_rmse", rf_city_rmse)
    mlflow.log_metric("city_mae", rf_city_mae)


print("\nForecast is for city:", user_city)


# ---------------- MLflow: Random Forest (City-Specific) ----------------
'''with mlflow.start_run(
    run_name=f"City_{user_city}_RandomForest",
    nested=True
):
    mlflow.log_param("city", user_city)
    mlflow.log_param("model_type", "RandomForest")

    mlflow.log_metric("rmse", rf_rmse)
    mlflow.log_metric("mae", rf_mae)'''


#MLflow for Random Forest
with mlflow.start_run(
    run_name="RandomForest_Global",
    nested=True
):
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 15)

    preds = rf_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = rf_model.score(X_test_scaled, y_test)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(
        sk_model=rf_model,
        name="random_forest_model",
        signature=signature
    )






# ================= XGBOOST REGRESSION =================

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

# Train XGBoost on SAME data
xgb_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test_scaled)



# Evaluation
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_mae  = mean_absolute_error(y_test, y_pred_xgb)

print("\nXGBoost Performance")
print("RMSE:", xgb_rmse)
print("MAE :", xgb_mae)

#XGB model's metrics to be tracked in mlflow
evaluate_global_model("XGBoost", xgb_model, X_test_scaled, y_test)



# ---------------- XGBOOST 72-HOUR FORECAST ----------------

xgb_24 = y_pred_xgb[-24:]
xgb_48 = y_pred_xgb[-48:]
xgb_72 = y_pred_xgb[-72:]

xgb_forecast_df = pd.DataFrame({
    "Hour": np.arange(1, 73),
    "Predicted_AQI": np.concatenate([
        xgb_24,
        xgb_48[24:],
        xgb_72[48:]
    ])
})

print("\nXGBoost 72-Hour Forecast")
print(xgb_forecast_df.head())


#logging xgboost to mlflow
with mlflow.start_run(
    run_name="XGBoost_Global",
    nested=True
):
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 6)

    mlflow.log_metric("rmse", xgb_rmse)
    mlflow.log_metric("mae", xgb_mae)

    mlflow.sklearn.log_model(
        sk_model=xgb_model,
        name="xgboost_model",
        signature=signature
    )



# ---------- ACTUAL VS PREDICTED FOR BEST MODEL ----------

# Prepare city test data (already scaled)
X_city_best = X_city_scaled   # same features & scaler
y_city_true = city_72_true

# ================= BEST MODEL SELECTION BY MLFLOW=================

best_model_name = min(GLOBAL_MODEL_METRICS, key=GLOBAL_MODEL_METRICS.get)
best_model_rmse = GLOBAL_MODEL_METRICS[best_model_name]
best_model = GLOBAL_MODEL_OBJECTS[best_model_name]

print("\n BEST MODEL SELECTED")
print(f"Model: {best_model_name}")
print(f"RMSE : {best_model_rmse:.4f}")

# Predict using BEST model

y_city_best_pred = best_model.predict(X_city_best)[-72:]



best_plot_name = f"actual_vs_predicted_aqi_{best_model_name}_{user_city}.png"

plt.figure(figsize=(14, 6))

# AQI risk zones
plt.axhspan(0, 50, color="green", alpha=0.12, label="Good")
plt.axhspan(50, 100, color="lime", alpha=0.12, label="Satisfactory")
plt.axhspan(100, 200, color="orange", alpha=0.12, label="Moderate")
plt.axhspan(200, 300, color="red", alpha=0.12, label="Poor")
plt.axhspan(300, 400, color="purple", alpha=0.12, label="Very Poor")
plt.axhspan(400, 500, color="brown", alpha=0.12, label="Severe")

hours = np.arange(1, 73)

plt.plot(hours, y_city_true, "--o", label="Actual AQI", color="black")
plt.plot(hours, y_city_best_pred, "-x", label="Predicted AQI", color="blue")

plt.xlabel("Hour")
plt.ylabel("AQI")
plt.title(f"Actual vs Predicted AQI (Best Model: {best_model_name}) – {user_city}")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(best_plot_name, dpi=300)
plt.close()

safe_log_artifact(best_plot_name)





#Register best model in MLflow Registry
registered_model_name = "AQI_Best_Model"

with mlflow.start_run(
    run_name="Best_Model_Registration",
    nested=True
):
    safe_log_artifact(best_plot_name)
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("best_rmse", best_model_rmse)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="best_model",
        registered_model_name=registered_model_name,
        signature=signature
    )

print(f"\n Model registered as: {registered_model_name}")


# ================= XGBOOST FEATURE IMPORTANCE =================

xgb_importance = pd.DataFrame({
    "Feature": X_train_scaled.columns,
    "Importance": xgb_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nXGBoost Feature Importance")
print(xgb_importance)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(
    xgb_importance["Feature"],
    xgb_importance["Importance"]
)
plt.xlabel("Importance Score")
plt.title("Feature Importance - XGBoost")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("xgboost_feature_importance.png", dpi=300)
plt.close()





#day 3------------------------------------------------------------

#Extract Coefficients from Linear Regression
lr_coefficients = pd.DataFrame({
    "Feature": X_train_scaled.columns,
    "Coefficient": lr_model.coef_
})


#Compute Absolute Influence (Magnitude)
lr_coefficients["Abs_Coefficient"] = lr_coefficients["Coefficient"].abs()
lr_coefficients = lr_coefficients.sort_values(
    by="Abs_Coefficient", ascending=False
)

print(lr_coefficients)


#Normalize Influence (Percentage Contribution)
lr_coefficients["Importance_%"] = (
    lr_coefficients["Abs_Coefficient"] /
    lr_coefficients["Abs_Coefficient"].sum()
) * 100

print(lr_coefficients)



# Sort features by importance (descending)
lr_plot = lr_coefficients.sort_values("Importance_%", ascending=True)

#feature importance for entire datase
'''
plt.figure(figsize=(10, 6))
plt.barh(
    lr_plot["Feature"],
    lr_plot["Importance_%"]
)

plt.xlabel("Importance (%) [Log Scale]")
plt.title("Feature Influence on AQI (Linear Regression)")
plt.xscale("log")              
plt.tight_layout()

plt.savefig("feature_importance_linear_regression_logscale.png", dpi=300)
plt.close()



plt.figure(figsize=(12, 5))

plt.plot(
    forecast_df["Hour"],
    forecast_df["Predicted_AQI"],
    marker="o",
    linewidth=2
)

plt.xlabel("Forecast Hour")
plt.ylabel("Predicted AQI")
plt.title(f"72-Hour AQI Forecast (Linear Regression) - {last_city}")

plt.grid(True)
plt.tight_layout()

# Save plot to current directory
plt.savefig("aqi_72_hour_forecast.png", dpi=300)
plt.close()
'''




#Linear Regression does not directly encode city

city_avg_aqi = train_df.groupby("city")["aqi"].agg(
    mean="mean",
    max="max",
    std="std"
).sort_values(by="max", ascending=False)


print("avg(approx.) aqi for cities:\n ", city_avg_aqi)

#Rolling Spike Detection 

# Copy city data
city_anomaly_df = city_test_df.copy()

# Rolling window (24 hours)
WINDOW = 24
THRESHOLD_K = 2

city_anomaly_df["rolling_mean"] = (
    city_anomaly_df["aqi"].rolling(WINDOW).mean()
)
city_anomaly_df["rolling_std"] = (
    city_anomaly_df["aqi"].rolling(WINDOW).std()
)

# Detect spike anomalies
city_anomaly_df["anomaly_rolling"] = (
    city_anomaly_df["aqi"] >
    city_anomaly_df["rolling_mean"] + THRESHOLD_K * city_anomaly_df["rolling_std"]
)


#POLLUTANT-WISE SPIKE DETECTION
pollutant_thresholds = {
    "pm2_5": 60,     # µg/m³
    "pm10": 100,    # µg/m³
    "no2": 80,      # µg/m³
    "co": 2,        # mg/m³
    "o3": 100       # µg/m³
}

for pollutant, threshold in pollutant_thresholds.items():
    city_anomaly_df[f"{pollutant}_spike"] = (
        city_anomaly_df[pollutant] > threshold
    )

# Static AQI threshold (CPCB: Poor and above)
AQI_STATIC_THRESHOLD = 200

city_anomaly_df["anomaly_aqi_static"] = (
    city_anomaly_df["aqi"] > AQI_STATIC_THRESHOLD
)


#Combine all spike conditions
spike_columns = (
    ["anomaly_rolling", "anomaly_aqi_static"] +
    [f"{p}_spike" for p in pollutant_thresholds]
)

city_anomaly_df["pollution_episode"] = (
    city_anomaly_df[spike_columns].any(axis=1)
)



#view detected pollution episodes 
pollution_events = city_anomaly_df[
    city_anomaly_df["pollution_episode"]
]

print(f"\nDetected pollution episodes for {user_city}:")
print(
    pollution_events[
        ["date", "hour", "aqi"] + list(pollutant_thresholds.keys())
    ].head()
)


#alert message
if not pollution_events.empty:
    print("🚨 ALERT: Sudden pollution spike detected!")
else:
    print("✅ No significant pollution spikes detected.")


# visualization of spikes (ALWAYS create artifact)
x_axis = np.arange(len(city_anomaly_df))
plt.figure(figsize=(14, 6))
# AQI line
plt.plot(
    x_axis,
    city_anomaly_df["aqi"].values,
    label="AQI",
    color="blue"
)

# Pollution episodes
plt.scatter(
    x_axis[city_anomaly_df["pollution_episode"]],
    city_anomaly_df.loc[city_anomaly_df["pollution_episode"], "aqi"],
    color="red",
    label="Pollution Episode",
    s=80
)

plt.xlabel("Time Index")
plt.ylabel("AQI")
plt.title(f"Pollution Spike Detection – {user_city}")
plt.legend()
plt.grid(True)
plt.tight_layout()



plt.xlabel("Time Index")
plt.ylabel("AQI")
plt.title(f"Pollution Spike Detection – {user_city}")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(f"pollution_spikes_{user_city}.png", dpi=300)
plt.close()

#----------------get grphs in mlflow ui--------------------------------------------
def log_city_artifacts(city):
    artifact_files = [
        f"actual_vs_predicted_aqi_72hr_risk_{city}.png",
        f"meteorological_influence_{city}.png",
        f"pollution_spikes_{city}.png",
    ]

    for file in artifact_files:
        if os.path.exists(file):
            mlflow.log_artifact(file)
        else:
            print(f"Missing artifact: {file}")

with mlflow.start_run(
    run_name=f"City_{user_city}",
    nested=True
):
    # ---------------- PARAMS ----------------
    mlflow.log_param("city", user_city)
    mlflow.log_param("model_type", "Best_Model")

    # ---------------- METRICS ----------------
    mlflow.log_metric("rmse", best_model_rmse)
    mlflow.log_param("best_model_name", best_model_name)


    # ---------------- ARTIFACTS ----------------
    log_city_artifacts(user_city)
    
    
    mlflow.log_artifact(
    f"aqi_72_hour_forecast_{user_city}.png",
    artifact_path="forecast"
)

#close parent run of mlflow
mlflow.end_run()



#connection to flask

import joblib



import os
from pathlib import Path

def save_inference_artifacts(best_model, scaler, feature_columns, base_dir):

    Path(base_dir).mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, os.path.join(base_dir, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(base_dir, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(base_dir, "feature_columns.pkl"))

    meta = {
        "model_type": type(best_model).__name__
    }

    with open(os.path.join(base_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print("Saved best model artifacts:")
    print(" best_model.pkl")
    print(" scaler.pkl")
    print(" feature_columns.pkl")
    print(" model_meta.json")




BASE_DIR = r"C:\Users\Ill-Us-Ion\Desktop\aqiproject\src\components"

ARTIFACT_DIR = os.path.join(os.getcwd(), "artifacts")

ARTIFACT_DIR = r"C:\Users\Ill-Us-Ion\Desktop\aqiproject\src\components\artifacts"

save_inference_artifacts(    
    best_model=best_model,
    scaler=scaler,
    feature_columns=list(X_train_scaled.columns),
    base_dir=ARTIFACT_DIR)

print("Inference artifacts saved successfully.")






