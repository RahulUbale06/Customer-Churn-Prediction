import streamlit as st
import joblib
from pathlib import Path

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)
# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("📊 Customer Churn Prediction")

st.sidebar.markdown("""
### About Project

This application predicts whether a telecom customer is likely to churn using a **Random Forest Classifier**.

### Model Information

- **Algorithm:** Random Forest
- **Accuracy:** **80.41%**
- **Dataset:** IBM Telecom Customer Churn

---

""")
# ----------------------------
# Load Model
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "churn_model.pkl")
# scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
features = joblib.load(BASE_DIR / "models" / "features.pkl")

# ----------------------------
# Title
# ----------------------------
st.title("📊 Customer Churn Prediction")

st.caption(
    "Predict whether a telecom customer is likely to churn using Machine Learning."
)

st.markdown("---")

st.markdown("---")

st.header("📝 Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=800.0
    )

    cltv = st.number_input(
        "CLTV",
        min_value=0,
        value=3000
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

with col2:

    phone = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    device = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

predict = st.button(
    "🔍 Predict Churn",
    use_container_width=True,
)
import pandas as pd

if predict:

    # Create input dictionary with all features initialized to 0
    input_data = {feature: 0 for feature in features}

    # Numerical features
    input_data["Tenure Months"] = tenure
    input_data["Monthly Charges"] = monthly_charges
    input_data["Total Charges"] = total_charges
    input_data["CLTV"] = cltv

    # Binary categorical features
    input_data["Gender_Male"] = 1 if gender == "Male" else 0
    input_data["Senior Citizen_Yes"] = 1 if senior == "Yes" else 0
    input_data["Partner_Yes"] = 1 if partner == "Yes" else 0
    input_data["Dependents_Yes"] = 1 if dependents == "Yes" else 0
    input_data["Phone Service_Yes"] = 1 if phone == "Yes" else 0
    input_data["Paperless Billing_Yes"] = 1 if paperless == "Yes" else 0

    # Multiple Lines
    if multiple == "Yes":
        input_data["Multiple Lines_Yes"] = 1
    elif multiple == "No phone service":
        input_data["Multiple Lines_No phone service"] = 1

    # Internet Service
    if internet == "Fiber optic":
        input_data["Internet Service_Fiber optic"] = 1
    elif internet == "No":
        input_data["Internet Service_No"] = 1

    # Online Security
    if online_security == "Yes":
        input_data["Online Security_Yes"] = 1
    elif online_security == "No internet service":
        input_data["Online Security_No internet service"] = 1

    # Online Backup
    if online_backup == "Yes":
        input_data["Online Backup_Yes"] = 1
    elif online_backup == "No internet service":
        input_data["Online Backup_No internet service"] = 1

    # Device Protection
    if device == "Yes":
        input_data["Device Protection_Yes"] = 1
    elif device == "No internet service":
        input_data["Device Protection_No internet service"] = 1

    # Tech Support
    if tech == "Yes":
        input_data["Tech Support_Yes"] = 1
    elif tech == "No internet service":
        input_data["Tech Support_No internet service"] = 1

    # Streaming TV
    if tv == "Yes":
        input_data["Streaming TV_Yes"] = 1
    elif tv == "No internet service":
        input_data["Streaming TV_No internet service"] = 1

    # Streaming Movies
    if movies == "Yes":
        input_data["Streaming Movies_Yes"] = 1
    elif movies == "No internet service":
        input_data["Streaming Movies_No internet service"] = 1

    # Contract
    if contract == "One year":
        input_data["Contract_One year"] = 1
    elif contract == "Two year":
        input_data["Contract_Two year"] = 1

    # Payment Method
    if payment == "Credit card (automatic)":
        input_data["Payment Method_Credit card (automatic)"] = 1
    elif payment == "Electronic check":
        input_data["Payment Method_Electronic check"] = 1
    elif payment == "Mailed check":
        input_data["Payment Method_Mailed check"] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])



    # Prediction
    if predict:

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Keep feature order
        input_df = input_df[features]

        # Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        st.subheader("📈 Prediction Result")

        if prediction == 1:
            st.error("🔴 Customer is likely to churn.")
        else:
            st.success("🟢 Customer is not likely to churn.")

        st.progress(float(probability))

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Churn Probability", f"{probability:.2%}")

        with col2:
            st.metric(
                "Prediction",
                "Churn" if prediction else "No Churn"
            )
        st.markdown("---")

        st.header("📌 About")

        st.write("""
        This application uses a **Random Forest Machine Learning model** trained on the IBM Telecom Customer Churn dataset to predict whether a customer is likely to leave the telecom service.

        ### Technologies Used

        - Python
        - Streamlit
        - Pandas
        - Scikit-Learn
        - Joblib
        - Random Forest Classifier
        """)

        st.markdown("---")

        st.markdown(
            "<h5 style='text-align:center;'>Developed & Designed by <b>Rahul Ubale</b></h5>",
            unsafe_allow_html=True
        )
    st.markdown("---")

