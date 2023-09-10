import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def predict_diabetes(features):
    input_data = np.array(features).reshape(1, -1)
    std_data = scaler.transform(input_data)  # Use the loaded scaler
    prediction = model.predict(std_data)
    return prediction[0]

# Streamlit interface
st.title('Diabetes Prediction App')

pregnancies = st.number_input('Pregnancies', 0)
glucose = st.number_input('Glucose', 0)
bp = st.number_input('BloodPressure', 0)
skin_thickness = st.number_input('SkinThickness', 0)
insulin = st.number_input('Insulin', 0)
bmi = st.number_input('BMI', 0.0)
dpf = st.number_input('DiabetesPedigreeFunction', 0.0)
age = st.number_input('Age', 0)

if st.button('Predict'):
    result = predict_diabetes([pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age])
    st.write('The person is Diabetic' if result == 1 else 'The person is not Diabetic')
