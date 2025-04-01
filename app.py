import streamlit as st
import joblib
# import pickle
import pandas as pd
# import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Load the saved model and encoder
xgb_model = joblib.load('attrition_model.pkl')
encoder = joblib.load('encoder.pkl')  # Load the saved encoder

# Function to preprocess the user input and make predictions
def preprocess_input(user_input):
    # Convert the user input into a DataFrame
    input_df = pd.DataFrame([user_input])

    # Apply the same OneHotEncoder used during training
    encoded_input = encoder.transform(input_df[['Job Title', 'Department', 'Gender', 'Marital Status']])

    # Create a DataFrame with encoded features
    encoded_df = pd.DataFrame(encoded_input.toarray(), columns=encoder.get_feature_names_out())
    
    # Combine with the other features (Age, Years of Service, Salary)
    input_df = input_df.drop(columns=['Job Title', 'Department', 'Gender', 'Marital Status'])
    input_df = pd.concat([input_df, encoded_df], axis=1)
    
    return input_df

# Streamlit app UI
st.title('Employee Exit Prediction')

# User input
st.subheader('Enter details about the new hire')

age = st.number_input('Age', min_value=18, max_value=100, value=30)
years_of_service = st.number_input('Years of Service', min_value=0, max_value=50, value=1)
salary = st.number_input('Salary', min_value=100, max_value=200000, value=5000)

job_title = st.selectbox('Job Title', ['Customer Support Agent', 'Data Scientist', 'HR Specialist', 'IT Support Engineer', 'Marketing Analyst', 'Network Engineer'])
department = st.selectbox('Department', ['Customer Service', 'Data Analytics', 'Field Operations', 'Human Resources', 'IT & Software', 'Network Operations', 'Project Management', 'Sales & Marketing'])
gender = st.selectbox('Gender', ['Male', 'Female'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married'])

# Store input data in a dictionary
user_input = {
    'Age': age,
    'Years of Service': years_of_service,
    'Salary': salary,
    'Job Title': job_title,
    'Department': department,
    'Gender': gender,
    'Marital Status': marital_status
}

# Preprocess the input
processed_input = preprocess_input(user_input)

# Make prediction
exit_probability = xgb_model.predict_proba(processed_input)[:, 1][0]
exit_prediction = xgb_model.predict(processed_input)[0]

# Display results
st.subheader('Prediction Results')

if exit_prediction == 1:
    st.write("The new hire is **likely to exit** the company.")
else:
    st.write("The new hire is **unlikely to exit** the company.")

st.write(f"The probability of exit is: **{exit_probability * 100:.2f}%**")

