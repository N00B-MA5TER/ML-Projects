# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:38:19 2025

@author: NOOBMASTER
"""

import numpy as np
import pickle
import streamlit as st
import os

# Get the working directory of the main.py file
working_dir = os.path.dirname(os.path.abspath(__file__))
print(working_dir)

# loading the saved model
loaded_model = pickle.load(open(f"{working_dir}/trained_model.sav", 'rb')) 

# creating a function
def diabetes_prediction(input_data):
    
    #changing the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshaping the numpy array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0] == 0):
        return "The patient is Non-diabetic"
    else:
        return "The patient is a Diabetic"
    
def main():
    
    #giving a title
    st.title('🩺Diabetes Prediction Web App')
    
    #getting the input data from the user
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    
    #columns for input fields
    col1, col2 = st.columns(2)
    
    with col1: 
        Pregnancies = st.text_input('Number of Pregnancies')
    with col1:
        Glucose = st.text_input('Glucose Level')
    with col1:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin level')
    with col2:
        BMI = st.text_input('BMI value')
    with col2:
        DiabetesPedigreeFunction = st.text_input('Diabetets Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()