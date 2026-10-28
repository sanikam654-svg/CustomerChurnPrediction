from pathlib import Path
import warnings
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "customer_churn.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Keep customer_churn.csv inside the dataset folder."
        )

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    if "Churn" not in df.columns:
        raise ValueError("Dataset must contain a 'Churn' column.")

    # Convert TotalCharges into numeric values
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    # Clean target column
    df = df.dropna(subset=["Churn"]).copy()
    df["Churn"] = df["Churn"].astype(str).str.strip()
    df = df[df["Churn"].isin(["Yes", "No"])].copy()

    y = df["Churn"].map({"No": 0, "Yes": 1})

    # Remove target and customer ID from training features
    drop_columns = ["Churn"]

    if "customerID" in df.columns:
        drop_columns.append("customerID")

    X = df.drop(columns=drop_columns)

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Preprocessing for numeric data
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # Models to compare
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    best_name = None
    best_score = -1
    best_pipeline = None

    print(f"Dataset loaded: {len(df)} customers")
    print(f"Churn rate: {y.mean() * 100:.2f}%")
    print("\nComparing models using 5-fold cross-validation...")

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=1
        )

        mean_score = scores.mean()

        print(
            f"{name}: CV F1 = {mean_score:.4f} "
            f"(+/- {scores.std():.4f})"
        )

        if mean_score > best_score:
            best_name = name
            best_score = mean_score
            best_pipeline = pipeline

    # Train the selected model
    best_pipeline.fit(X_train, y_train)

    predictions = best_pipeline.predict(X_test)
    probabilities = best_pipeline.predict_proba(X_test)[:, 1]

    # Evaluate model performance
    print(f"\nSelected model: {best_name}")
    print("\nTest-set evaluation")

    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(
        f"Precision: "
        f"{precision_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall   : "
        f"{recall_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        f"F1 score : "
        f"{f1_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(f"ROC-AUC  : {roc_auc_score(y_test, probabilities):.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save the trained pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    print(f"\nModel saved successfully to:\n{MODEL_PATH}")


if __name__ == "__main__":
    main()