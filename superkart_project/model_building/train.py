
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import xgboost as xgb
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# -----------------------------
# Hugging Face Authentication
# -----------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))

# -----------------------------
# Load Data from Hugging Face Dataset Hub
# -----------------------------
REPO_ID = "JyalaHarsha-2025/MLOPS_Superkart_Sales_Forcast"

Xtrain_path = f"hf://datasets/{REPO_ID}/X_train.csv"
Xtest_path = f"hf://datasets/{REPO_ID}/X_test.csv"
ytrain_path = f"hf://datasets/{REPO_ID}/y_train.csv"
ytest_path = f"hf://datasets/{REPO_ID}/y_test.csv"

X_train = pd.read_csv(Xtrain_path)
X_test = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path).squeeze()  # convert to Series
y_test = pd.read_csv(ytest_path).squeeze()

# -----------------------------
# Feature Configuration
# -----------------------------
numeric_features = [
    'Product_Weight', 'Product_Allocated_Area', 'Product_MRP',
    'Store_Establishment_Year'
]

categorical_features = [
    'Product_Sugar_Content', 'Product_Type',
    'Store_Size', 'Store_Location_City_Type', 'Store_Type'
]

# -----------------------------
# Preprocessing Pipeline
# -----------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features),
    remainder='passthrough'
)

# -----------------------------
# Define XGBoost Regressor
# -----------------------------
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Hyperparameter Grid
param_grid = {
    'xgbregressor__n_estimators': [100, 150],
    'xgbregressor__max_depth': [4, 6],
    'xgbregressor__learning_rate': [0.05, 0.1],
    'xgbregressor__subsample': [0.7, 0.8],
    'xgbregressor__colsample_bytree': [0.7, 0.8],
}

# Pipeline
pipeline = make_pipeline(preprocessor, xgb_model)

# -----------------------------
# Grid Search
# -----------------------------
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("[INFO] Best Parameters:", grid_search.best_params_)

# -----------------------------
# Predictions
# -----------------------------
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# -----------------------------
# Evaluation Metrics
# -----------------------------
def evaluate(y_true, y_pred, dataset=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n[{dataset}] RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

evaluate(y_train, y_train_pred, "Train")
evaluate(y_test, y_test_pred, "Test")

# -----------------------------
# Save Model Locally
# -----------------------------
model_path = "superkart_sales_model_v1.joblib"
joblib.dump(best_model, model_path)
print(f"[INFO] Model saved to {model_path}")

# -----------------------------
# Upload Model to Hugging Face Hub
# -----------------------------
MODEL_REPO_ID = REPO_ID  # same as dataset repo for this case
repo_type = "model"

try:
    api.repo_info(repo_id=MODEL_REPO_ID, repo_type=repo_type)
    print(f"[INFO] Model repo '{MODEL_REPO_ID}' already exists.")
except RepositoryNotFoundError:
    print(f"[INFO] Model repo '{MODEL_REPO_ID}' not found. Creating new repo...")
    create_repo(repo_id=MODEL_REPO_ID, repo_type=repo_type, private=False)
    print(f"[INFO] Model repo '{MODEL_REPO_ID}' created.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=MODEL_REPO_ID,
    repo_type=repo_type
)
print(f"[INFO] Model uploaded to Hugging Face Hub.")
