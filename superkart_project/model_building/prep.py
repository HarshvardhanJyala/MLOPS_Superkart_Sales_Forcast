
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

# -----------------------------
# Configurations & Constants
# -----------------------------
REPO_ID = "JyalaHarsha-2025/MLOPS_Superkart_Sales_Forcast"
DATASET_PATH = f"hf://datasets/{REPO_ID}/superkart.csv"
TARGET_COL = "Product_Store_Sales_Total"
OUTPUT_FILES = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]

# -----------------------------
# Authenticate to Hugging Face Hub
# -----------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(DATASET_PATH)
print("[INFO] Dataset loaded successfully.")

# -----------------------------
# Drop Irrelevant Columns
# -----------------------------
# No identifier drop yet, keeping Product_ID and Store_ID for feature use
# Optionally convert them later if needed

# -----------------------------
# Handle Missing Values
# -----------------------------
# Fill numerical columns with median
numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns.difference([TARGET_COL])
df[numerical_cols] = df[numerical_cols].apply(lambda col: col.fillna(col.median()))

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include=["object"]).columns.difference(["Product_ID", "Store_ID"])
df[categorical_cols] = df[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]))

# -----------------------------
# Encode Categorical Columns
# -----------------------------
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Optional: Convert Product_ID and Store_ID to numeric hash (or leave as-is if using deep models)
df["Product_ID"] = df["Product_ID"].astype(str).apply(lambda x: hash(x) % 10000)
df["Store_ID"] = df["Store_ID"].astype(str).apply(lambda x: hash(x) % 10000)

# -----------------------------
# Split Dataset
# -----------------------------
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("[INFO] Data split into train and test sets.")

# -----------------------------
# Save Locally
# -----------------------------
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("[INFO] Files saved locally.")

# -----------------------------
# Upload to Hugging Face Dataset Repo
# -----------------------------
for file_path in OUTPUT_FILES:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"[INFO] Uploaded {file_path} to Hugging Face Hub.")
