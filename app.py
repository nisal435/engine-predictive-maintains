import streamlit as st
import pickle
import numpy as np

# Load the trained model (replace 'model.pkl' with your actual file name)
with open('hhmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the customized ranges for each feature based on dataset statistics
custom_ranges = {
    'Engine rpm': (61.0, 2239.0),
    'Lub oil pressure': (0.003384, 7.265566),
    'Fuel pressure': (0.003187, 21.138326),
    'Coolant pressure': (0.002483, 7.478505),
    'lub oil temp': (71.321974, 89.580796),
    'Coolant temp': (61.673325, 195.527912),
    'Temperature_difference': (-22.669427, 119.008526)
}

# Feature Descriptions
feature_descriptions = {
    'Engine rpm': 'Revolution per minute of the engine.',
    'Lub oil pressure': 'Pressure of the lubricating oil.',
    'Fuel pressure': 'Pressure of the fuel.',
    'Coolant pressure': 'Pressure of the coolant.',
    'lub oil temp': 'Temperature of the lubricating oil.',
    'Coolant temp': 'Temperature of the coolant.',
    'Temperature_difference': 'Temperature difference between components.'
}

# Engine Condition Prediction App
def main():
    st.set_page_config(page_title="Engine Condition Prediction", page_icon="⚙️", layout="wide")
    st.markdown("""
        <style>
        .main { 
            background-color: #f7f7f7; 
            padding: 20px;
            border-radius: 10px;
        }
        .title {
            color: #0073e6;
            font-size: 36px;
            font-weight: bold;
        }
        .sidebar-title {
            color: #0073e6;
            font-size: 18px;
            font-weight: bold;
        }
        .sidebar {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
        }
        .prediction {
            font-size: 18px;
            font-weight: bold;
        }
        .button {
            background-color: #0073e6;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Engine Condition Prediction ⚙️")

    # Display feature descriptions
    st.sidebar.title("Feature Descriptions")
    st.sidebar.markdown("Learn more about each engine feature and its significance.")
    for feature, description in feature_descriptions.items():
        st.sidebar.markdown(f"**{feature}:** {description}")

    # Add the image under the feature description
    image_url = "https://www.intuceo.com/blog/wp-content/uploads/2020/08/predictive-maintenance-using-machine-learning.jpg"
    st.sidebar.image(image_url, use_container_width=True)

    # Create a container for the input fields
    with st.container():
        st.markdown('<div class="main">', unsafe_allow_html=True)

        # Input widgets with customized ranges
        engine_rpm = st.slider("Engine RPM", min_value=float(custom_ranges['Engine rpm'][0]), 
                               max_value=float(custom_ranges['Engine rpm'][1]), 
                               value=float(custom_ranges['Engine rpm'][1] / 2))
        lub_oil_pressure = st.slider("Lub Oil Pressure", min_value=custom_ranges['Lub oil pressure'][0], 
                                     max_value=custom_ranges['Lub oil pressure'][1], 
                                     value=(custom_ranges['Lub oil pressure'][0] + custom_ranges['Lub oil pressure'][1]) / 2)
        fuel_pressure = st.slider("Fuel Pressure", min_value=custom_ranges['Fuel pressure'][0], 
                                  max_value=custom_ranges['Fuel pressure'][1], 
                                  value=(custom_ranges['Fuel pressure'][0] + custom_ranges['Fuel pressure'][1]) / 2)
        coolant_pressure = st.slider("Coolant Pressure", min_value=custom_ranges['Coolant pressure'][0], 
                                     max_value=custom_ranges['Coolant pressure'][1], 
                                     value=(custom_ranges['Coolant pressure'][0] + custom_ranges['Coolant pressure'][1]) / 2)
        lub_oil_temp = st.slider("Lub Oil Temperature", min_value=custom_ranges['lub oil temp'][0], 
                                 max_value=custom_ranges['lub oil temp'][1], 
                                 value=(custom_ranges['lub oil temp'][0] + custom_ranges['lub oil temp'][1]) / 2)
        coolant_temp = st.slider("Coolant Temperature", min_value=custom_ranges['Coolant temp'][0], 
                                 max_value=custom_ranges['Coolant temp'][1], 
                                 value=(custom_ranges['Coolant temp'][0] + custom_ranges['Coolant temp'][1]) / 2)
        temp_difference = st.slider("Temperature Difference", min_value=custom_ranges['Temperature_difference'][0], 
                                    max_value=custom_ranges['Temperature_difference'][1], 
                                    value=(custom_ranges['Temperature_difference'][0] + custom_ranges['Temperature_difference'][1]) / 2)

        # Predict button
        if st.button("Predict Engine Condition", key="predict", use_container_width=True):
            result, confidence = predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference)
            
            # Explanation
            if result == 0:
                st.info(f"The engine is predicted to be in a **normal condition**. Confidence: {1.0 - confidence:.2%}")
            else:
                st.warning(f"**Warning!** Please investigate further. Confidence: {1.0 - confidence:.2%}")

        # Reset button
        if st.button("Reset Values", key="reset", use_container_width=True):
            st.experimental_rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# Function to predict engine condition
def predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference):
    input_data = np.array([engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference]).reshape(1, -1)
    prediction = model.predict(input_data)
    confidence = model.predict_proba(input_data)[:, 1]  # For binary classification, adjust as needed
    return prediction[0], confidence[0]

if __name__ == "__main__":
    main()
