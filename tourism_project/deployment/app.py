import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Wellness Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to purchase a newly introduced
**Wellness Tourism Package** based on their demographic and interaction characteristics.
""")

# User input
st.subheader("Customer Demographics")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
with col2:
    monthly_income = st.number_input("Monthly Income (USD)", min_value=1000.0, max_value=100000.0, value=22000.0, step=500.0)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

st.subheader("Travel & Interactions")
col3, col4 = st.columns(2)
with col3:
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=300.0, value=15.0, step=1.0)
    num_followups = st.number_input("Number of Follow-ups", min_value=1.0, max_value=10.0, value=3.0, step=1.0)
with col4:
    num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3, step=1)
    num_children_visiting = st.number_input("Number of Children Visiting (< 5 yrs)", min_value=0, max_value=5, value=1, step=1)
    num_trips = st.number_input("Number of Trips Annually", min_value=1.0, max_value=30.0, value=3.0, step=1.0)
    property_star = st.selectbox("Preferred Property Star Rating", [3.0, 4.0, 5.0])

col5, col6 = st.columns(2)
with col5:
    passport = st.selectbox("Has Passport?", ["No", "Yes"])
    passport_val = 1 if passport == "Yes" else 0
with col6:
    own_car = st.selectbox("Owns a Car?", ["No", "Yes"])
    own_car_val = 1 if own_car == "Yes" else 0

pitch_satisfaction = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)

# Assemble input into DataFrame
# The preprocessor pipeline expects all numeric features (including the target feature ProdTaken as dummy) and categorical features
input_data = pd.DataFrame([{
    'ProdTaken': 0, # Dummy placeholder required by numerical preprocessor
    'Age': float(age),
    'CityTier': int(city_tier),
    'DurationOfPitch': float(duration_pitch),
    'NumberOfPersonVisiting': int(num_person_visiting),
    'NumberOfFollowups': float(num_followups),
    'PreferredPropertyStar': float(property_star),
    'NumberOfTrips': float(num_trips),
    'Passport': int(passport_val),
    'PitchSatisfactionScore': int(pitch_satisfaction),
    'OwnCar': int(own_car_val),
    'NumberOfChildrenVisiting': float(num_children_visiting),
    'MonthlyIncome': float(monthly_income),
    'TypeofContact': typeof_contact,
    'Occupation': occupation,
    'Gender': gender,
    'ProductPitched': product_pitched,
    'MaritalStatus': marital_status,
    'Designation': designation
}])

# Predict button
if st.button("Predict Purchase Likelihood"):
    # Using threshold 0.45 or continuous score regression prediction
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    if prediction >= 0.5: # Regressor score mapping to binary
        st.success("The customer is **Likely** to purchase the Wellness Tourism Package!")
    else:
        st.warning("The customer is **Unlikely** to purchase the Wellness Tourism Package.")
