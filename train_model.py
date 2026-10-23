import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset

data = pd.read_csv("dataset/customer_churn.csv")

print("Dataset loaded successfully!")

# Step 2: Clean the data

data.columns = data.columns.str.strip()

data["TotalCharges"] = pd.to_numeric(
data["TotalCharges"], errors="coerce"
)

# Remove customer ID because it does not help predict churn

data = data.drop(columns=["customerID"])

# Remove rows where the target is missing

data = data.dropna(subset=["Churn"])

# Step 3: Separate features and target

X = data.drop(columns=["Churn"])
y = data["Churn"].map({"No": 0, "Yes": 1})

# Step 4: Identify numerical and categorical columns

numerical_columns = X.select_dtypes(
include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
include=["object"]
).columns.tolist()

# Step 5: Prepare the data

numerical_pipeline = Pipeline([
("imputer", SimpleImputer(strategy="median")),
("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
("imputer", SimpleImputer(strategy="most_frequent")),
("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
("numerical", numerical_pipeline, numerical_columns),
("categorical", categorical_pipeline, categorical_columns)
])

# Step 6: Create the machine learning model

model = Pipeline([
("preprocessor", preprocessor),
("classifier", RandomForestClassifier(
n_estimators=100,
random_state=42,
class_weight="balanced"
))
])

# Step 7: Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

print("\nTraining the machine learning model...")

# Step 8: Train the model

model.fit(X_train, y_train)

# Step 9: Make predictions

y_pred = model.predict(X_test)

# Step 10: Evaluate the model

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
y_test,
y_pred,
target_names=["No Churn", "Churn"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 11: Save the trained model

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/churn_model.pkl")

print("\nModel saved successfully!")
print("Location: models/churn_model.pkl")
