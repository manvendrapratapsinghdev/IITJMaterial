import streamlit as st
import pandas as pd
import joblib
import os
import json
import threading

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
model = joblib.load(model_path)

# Load translations from the JSON file
with open("translations.json", "r") as file:
    translations = json.load(file)

# Ensure Hindi is at the top of the available languages list
available_languages = ["Hindi"] + [lang for lang in translations.keys() if lang != "Hindi"]

# UI Header
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #f9f9f9; padding: 20px;}
        .stButton>button {background-color: #007BFF; color: white; font-size: 16px; padding: 10px 20px; border-radius: 5px;}
        .stTextInput, .stNumberInput, .stSelectbox, .stRadio, .stSlider {margin-bottom: 15px;}
        h1, h2, h3, h4, h5, h6 {font-weight: bold;}
        h1 {font-size: 3rem;}
        h2 {font-size: 2.5rem;}
        h3 {font-size: 2rem;}
        .section-heading {font-size: 1.75rem; font-weight: bold; margin-top: 10px; margin-bottom: 10px;}
        .streamlit-expanderHeader {font-size: 1.5rem; font-weight: bold;} /* Custom font size for expander titles */
    </style>
""", unsafe_allow_html=True)

# Use Streamlit's layout options to position the dropdown in the top-right corner
col1, col2 = st.columns([9, 1])
with col2:
    selected_language = st.selectbox("", available_languages, index=available_languages.index("Hindi"))
current_translations = translations[selected_language]

# Replace hardcoded English text with dynamic translations
st.markdown(
    f"""
    <div style=\"text-align: center; margin-bottom: 20px; margin-top: -40px;\">
        <h1 style=\"font-size: 2.5rem; color: #007BFF;\">{current_translations['app_title']}</h1>
        <p style=\"font-size: 1.2rem; color: #555;\">{current_translations['app_description']}</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
# Input form
def calculate_bmi(data):
    height_in_meters = data["Height"] * 0.3048  # Convert height from feet to meters
    data["BMI"] = data["Weight"] / (height_in_meters ** 2)
    return data

def user_input():
    # Replace hardcoded fields with JSON equivalent text based on the selected language
    st.header(current_translations["user_input_header"])
    st.write(current_translations["user_input_description"])

    data = {}

    with st.expander(current_translations["general_info"]):
        data["Sex"] = st.selectbox(current_translations["gender"], [current_translations["male"], current_translations["female"]], key="sex")
        data["Age"] = st.number_input(current_translations["age"], min_value=10, max_value=100, value=30, step=1, key="age")
        data["Weight"] = st.number_input(current_translations["weight"], min_value=30.0, max_value=200.0, value=70.0, step=0.1, key="weight")

        # Replace height input with dropdown for height in feet with increments of 0.1
        height_options = [round(x * 0.1, 1) for x in range(30, 76)]  # Generate options from 3.0 to 7.5
        data["Height"] = st.selectbox(current_translations["height"], options=height_options, index=25, key="height_in_feet")

    # Calculate BMI
    data = calculate_bmi(data)

    with st.expander(current_translations["health_conditions"]):
        fields = {
            "HighBP": current_translations["high_bp"],
            "HighChol": current_translations["high_chol"],
            "Smoker": current_translations["smoker"],
            "Stroke": current_translations["stroke"],
            "HeartDiseaseorAttack": current_translations["heart_disease"],
            "PhysActivity": current_translations["phys_activity"],
            "Fruits": current_translations["fruits"],
            "Veggies": current_translations["veggies"],
            "HvyAlcoholConsump": current_translations["heavy_alcohol"],
            "DiffWalk": current_translations["difficulty_walking"],
        }
        cols = st.columns(2)  # Split into 2 columns
        for i, (field, label) in enumerate(fields.items()):
            col = cols[i % 2]
            data[field] = col.radio(label, [current_translations["yes"], current_translations["no"]], index=1, key=field)

    # Hardcode "AnyHealthcare" to 1
    data["CholCheck"] = 1
    data["AnyHealthcare"] = 1
    data["NoDocbcCost"] = 0

    with st.expander(current_translations["health_ratings"]):
        data["GenHlth"] = st.slider(current_translations["general_health"], 1, 5, value=3, key="genhlth")
        data["MentHlth"] = st.slider(current_translations["mental_health"], 0, 30, value=5, key="menthlth")
        data["PhysHlth"] = st.slider(current_translations["physical_health"], 0, 30, value=5, key="physhlth")

    # Ensure the order of features matches the trained model
    feature_order = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", 
                     "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", 
                     "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age"]

    # Convert categorical inputs to numeric
    data["Sex"] = 1 if data["Sex"] == "Male" else 0
    for field in fields.keys():
        data[field] = 1 if data[field] == "Yes" else 0

    return pd.DataFrame([[data[feature] for feature in feature_order]], columns=feature_order)

# Get user input
input_df = user_input()

# Predict button
if input_df is not None:
    st.markdown("---")
    st.subheader(f"**{current_translations['predict_button']}**")

    if st.button(current_translations["predict_button"]):
        with st.spinner(current_translations["analyzing_data"]):
            prediction = model.predict(input_df)[0]

            # Update BMI category text
            bmi_value = round(input_df["BMI"].iloc[0], 2)
            if bmi_value < 18.5:
                bmi_category = current_translations["underweight"]
                bmi_color = "#FFA500"  # Orange
            elif 18.5 <= bmi_value <= 24.9:
                bmi_category = current_translations["normal"]
                bmi_color = "#008000"  # Darker green for better contrast
            elif 25 <= bmi_value <= 29.9:
                bmi_category = current_translations["overweight"]
                bmi_color = "#FFD700"  # Yellow
            else:
                bmi_category = current_translations["obese"]
                bmi_color = "#FF0000"  # Red

            # Display BMI and category below the prediction result
            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 20px;">
                    <p style="font-size: 18px; color: {bmi_color}; font-weight: bold;">{current_translations['bmi_message'].format(bmi_value=bmi_value, bmi_category=bmi_category)}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if prediction == 1:  # Diabetic
                st.markdown(
                    f"""
                    <div style=\"text-align: center; padding: 20px; background-color: #ffe6e6; border-radius: 10px;\">
                        <h2 style=\"color: red;\"><b>{current_translations['diabetic']}</b></h2>
                        <p style=\"color: red;\">{current_translations['diabetic_message']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:  # Non-Diabetic
                st.markdown(
                    f"""
                    <div style=\"text-align: center; padding: 20px; background-color: #e6ffe6; border-radius: 10px;\">
                        <h2 style=\"color: green;\"><b>{current_translations['non_diabetic']}</b></h2>
                        <p style=\"color: green;\">{current_translations['non_diabetic_message']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Add disclaimer about model limitations
            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 10px; border: 1px solid #ddd;">
                    <p style="font-size: 14px; color: #6c757d;">
                        <i>{current_translations['disclaimer']}</i>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

# Add a professional footer
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #f9f9f9;
            padding: 10px 20px;
            text-align: center;
            font-size: 14px;
            color: #6c757d;
            border-top: 1px solid #eaeaea;
        }
        .footer a {
            color: #007BFF;
            text-decoration: none;
            font-weight: bold;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
    <div class="footer">
        Connect with me on 
        <a href="https://www.linkedin.com/in/manvendrapratapsinghdev/" target="_blank">LinkedIn</a>.
    </div>
    """,
    unsafe_allow_html=True
)
