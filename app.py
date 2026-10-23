from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "customer_churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

st.sidebar.title("📊 Churn Prediction")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Churn Analysis", "Predict Churn"],
)

st.title("Customer Churn Prediction")
st.write("Analyze customer data and predict the likelihood of churn.")

df = load_data()
model = load_model()

if df is None:
    st.error(f"Dataset not found: {DATA_PATH}")

elif page == "Dashboard":
    st.header("Dashboard Overview")

    total_customers = len(df)

    if "Churn" in df.columns:
        churn_count = (
            df["Churn"].astype(str).str.strip().str.lower() == "yes"
        ).sum()
        churn_rate = (
            churn_count / total_customers * 100
            if total_customers else 0
        )
    else:
        churn_count = 0
        churn_rate = 0

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Customers Churned", f"{churn_count:,}")
    col3.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    if "Churn" in df.columns:
        st.subheader("Churn Distribution")
        st.bar_chart(df["Churn"].value_counts())

elif page == "Churn Analysis":
    st.header("Customer Churn Analysis")

    if "Churn" in df.columns:
        st.subheader("Churn Distribution")
        st.bar_chart(df["Churn"].value_counts())

    if "Contract" in df.columns and "Churn" in df.columns:
        st.subheader("Churn by Contract Type")
        contract_data = pd.crosstab(
            df["Contract"],
            df["Churn"],
        )
        st.bar_chart(contract_data)

    if "MonthlyCharges" in df.columns:
        st.subheader("Monthly Charges Distribution")
        st.bar_chart(df["MonthlyCharges"].value_counts().sort_index())

elif page == "Predict Churn":
    st.header("Predict Customer Churn")

    if model is None:
        st.error(
            "The trained model was not found. "
            "Check that models/churn_model.pkl exists."
        )

    elif not hasattr(model, "feature_names_in_"):
        st.error(
            "The model does not contain feature names. "
            "Please check how train_model.py trained the model."
        )

    else:
        st.write("Enter customer details to estimate churn risk.")

        input_data = {}

        with st.form("prediction_form"):
            for feature in model.feature_names_in_:
                if feature in df.columns:
                    if pd.api.types.is_numeric_dtype(df[feature]):
                        default_value = float(
                            pd.to_numeric(
                                df[feature], errors="coerce"
                            ).median()
                        )
                        if pd.isna(default_value):
                            default_value = 0.0

                        input_data[feature] = st.number_input(
                            feature,
                            value=default_value,
                        )
                    else:
                        options = (
                            df[feature]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )

                        if options:
                            input_data[feature] = st.selectbox(
                                feature,
                                options,
                            )
                        else:
                            input_data[feature] = st.text_input(
                                feature,
                                value="",
                            )
                else:
                    st.warning(
                        f"Feature '{feature}' is missing from the dataset."
                    )

            submitted = st.form_submit_button("Predict Churn")

        if submitted:
            try:
                input_df = pd.DataFrame(
                    [input_data],
                    columns=list(model.feature_names_in_),
                )

                prediction = model.predict(input_df)[0]

                st.subheader("Prediction Result")

                if str(prediction).strip().lower() in ["yes", "1", "true"]:
                    st.error("⚠️ This customer may churn.")
                else:
                    st.success("✅ This customer is likely to stay.")

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_df)[0]
                    st.write("Prediction probabilities:")

                    probability_df = pd.DataFrame({
                        "Class": model.classes_,
                        "Probability (%)": probabilities * 100,
                    })

                    st.dataframe(
                        probability_df,
                        use_container_width=True,
                    )

            except Exception as error:
                st.error(f"Prediction failed: {error}")
                st.info(
                    "The model may require preprocessing or a different "
                    "input format. Check train_model.py."
                )