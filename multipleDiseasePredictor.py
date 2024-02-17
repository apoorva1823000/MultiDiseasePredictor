# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 22:06:49 2024

@author: Apoorva .S. Mehta
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

# loading the models
diabetes_model = pickle.load(open('trained_model.sav','rb'))
heart_disease_model = pickle.load(open('heart_disease_trained_model.sav','rb'))
parkinsons_disease_model = pickle.load(open('Parkinsons_trained_model.sav','rb'))

#sidebar menu
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'], 
                           icons=['activity','heart','person'],
                           default_index=0)
    
    
#Setting theme


if (selected=='Diabetes Prediction'):
    st.title('Diabetes Prediction using ML')
    
    data = {
    'Attribute': [
        'Pregnancies', 'Glucose', 'BloodPressure', 
        'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ],
    'Description': [
        'Number of times pregnant',
        'Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
        'Diastolic blood pressure (mm Hg)',
        'Triceps skin fold thickness (mm)',
        '2-Hour serum insulin (mu U/ml)',
        'Body mass index (weight in kg/(height in m)^2)',
        'Diabetes pedigree function',
        'Age (years)',
        'Class variable (0 or 1) indicating whether the individual has diabetes or not'
    ]
}

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Display the DataFrame as a table using Streamlit
    st.write(
    df
    .style
    .set_properties(**{'max-width': '100%', 'font-size': '1vw'})
    .set_table_styles([{
        'selector': 'tr:hover',
        'props': 'background-color: #ffff99;'
    }])
)
    
    col1,col2,col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI Level')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Value')
    with col2:
        Age = st.text_input('Age Value')
    diabetes_diagnosis = ''
    #creating a button for prediction
    if st.button('Diabetes Test Results'):
        diabetes_prediction = diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        if(diabetes_prediction[0]==1):
            diabetes_diagnosis = 'The person is Diabetic'
        else:
            diabetes_diagnosis = 'The person is Non-Diabetic'
    st.success(diabetes_diagnosis)
    
    
if (selected=='Heart Disease Prediction'):
    st.title('Heart Disease Prediction using ML')
    
    data = {
    'Attribute': [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ],
    'Description': [
        'Age (in years)',
        'Sex (0 = female, 1 = male)',
        'Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)',
        'Resting blood pressure (in mm Hg)',
        'Serum cholesterol (in mg/dl)',
        'Fasting blood sugar > 120 mg/dl (0 = false, 1 = true)',
        'Resting electrocardiographic results (0: normal, 1: having ST-T wave abnormality, 2: showing probable or definite left ventricular hypertrophy)',
        'Maximum heart rate achieved',
        'Exercise induced angina (0 = no, 1 = yes)',
        'ST depression induced by exercise relative to rest',
        'Slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)',
        'Number of major vessels (0-3) colored by flourosopy',
        'Thalassemia (0: normal, 1: fixed defect, 2: reversible defect)',
        'Presence of heart disease (0 = no, 1 = yes)'
    ]
}

    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Display the DataFrame as a table using Streamlit
    st.write(
    df
    .style
    .set_properties(**{'max-width': '100%', 'font-size': '1vw'})
    .set_table_styles([{
        'selector': 'tr:hover',
        'props': 'background-color: #ffff99;'
    }])
)
    
    col1,col2,col3 = st.columns(3)
    with col1:
        age = st.text_input('Enter your Age')
    with col2:
        sex = st.text_input('Enter 0 if Female and 1 if Male')
    with col3:
        cp = st.text_input('CP Value')
    with col1:
        trest = st.text_input('Resting Blood Pressure Value')
    with col2:
        chol = st.text_input('Cholestrol Level')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar Value')
    with col1:
        restecg = st.text_input('Resting electrocardiographic results ')
    with col2:
        thalach = st.text_input('Maximum heart rate achieved Value')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('OldPeak Value')
    with col2:
        slope = st.text_input('Slope Value')
    with col3:
        ca = st.text_input('Number of major vessels (0-3) colored by flourosopy')
    with col1:
        thal = st.text_input('Thal Value')
    heart_disease_diagnosis = ''
    #creating a button for prediction
    if st.button('Heart Disease Test Results'):
        heart_disease_prediction = heart_disease_model.predict([[age,sex,cp,trest,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        if(heart_disease_prediction[0]==1):
            heart_disease_diagnosis = 'The person has chances of Heart Disease'
        else:
            heart_disease_diagnosis = 'The person is Healthy'
    st.success(heart_disease_diagnosis)
    
    
if (selected=='Parkinsons Prediction'):
    st.title('Parkinsons Prediction using ML')
    
    data = {
    'Feature': [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 
        'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
        'NHR', 'HNR',
        'RPDE', 'D2',
        'DFA',
        'spread1', 'spread2', 'PPE'
    ],
    'Description': [
        'Average vocal fundamental frequency',
        'Maximum vocal fundamental frequency',
        'Minimum vocal fundamental frequency',
        'Several measures of variation in fundamental frequency',
        'Several measures of variation in fundamental frequency (absolute)',
        'Variation in fundamental frequency - Relative amplitude perturbation',
        'Variation in fundamental frequency - Period perturbation quotient',
        'Variation in fundamental frequency - Jitter:DDP',
        'Several measures of variation in amplitude',
        'Several measures of variation in amplitude (in dB)',
        'Amplitude perturbation quotient - Three-point average',
        'Amplitude perturbation quotient - Five-point average',
        'Variation in amplitude',
        'Amplitude perturbation quotient - Three-point average (in dB)',
        'Noise to tonal components ratio',
        'Harmonic to noise ratio',
        'Nonlinear dynamical complexity measure',
        'Nonlinear dynamical complexity measure',
        'Signal fractal scaling exponent',
        'Nonlinear measure of fundamental frequency variation',
        'Nonlinear measure of fundamental frequency variation',
        'Nonlinear measure of fundamental frequency variation'
    ]
}

# Create a DataFrame
    df = pd.DataFrame(data)
    
    # Display the DataFrame as a table using Streamlit
    st.write(
    df
    .style
    .set_properties(**{'max-width': '100%', 'font-size': '1vw'})
    .set_table_styles([{
        'selector': 'tr:hover',
        'props': 'background-color: #ffff99;'
    }])
)
    
    col1,col2,col3 = st.columns(3)
    
    with col1:
        Fo = st.text_input('Enter MDVP:Fo value')
    with col2:
        Fhi = st.text_input('Enter MDVP:Fhi value')
    with col3:
        Flo = st.text_input('Enter MDVP:Flo value')
    with col1:
        Jitter = st.text_input('Enter MDVP:Jitter value')
    with col2:
        Jitter2 = st.text_input('Enter Jitter(Abs) value')
    with col3:
        RAP = st.text_input('Enter MDVP:RAP value')
    with col1:
        PPQ = st.text_input('Enter MDVP:PPQ value')
    with col2:
        Jitter3 = st.text_input('Enter Jitter(DDP) value')
    with col3:
        Shimmer = st.text_input('Enter MDVP(Shimmer) Value')
    with col1:
        Shimmer2 = st.text_input('Enter the Shimmer(db) Value')
    with col2:
        APQ3 = st.text_input('Enter the Shimmer(APQ3) Value')
    with col3:
        APQ5 = st.text_input('Enter the Shimmer(APQ5) Value')
    with col1:
        APQ = st.text_input('Enter the MDVP(APQ) Value')
    with col2:
        DDA = st.text_input('Enter the Shimmer (DDA) Value')
    with col3:
        NHR = st.text_input('Enter the NHR Value')
    with col1:
        HNR = st.text_input('Enter the HNR Value')
    with col2:
        RPDE = st.text_input('Enter the RPDE Value')
    with col3:
        DFA = st.text_input('Enter the DFA Value')
    with col1:
        spread1 = st.text_input('Enter the Spread1 Value')
    with col2:
        spread2 = st.text_input('Enter the Spread2 Value')
    with col3:
        D2 = st.text_input('Enter the D2 Value')
    with col1:
        PPE = st.text_input('Enter the PPE Value')
    parkinsons_disease_diagnosis = ''
    #creating a button for prediction
    if st.button('Parkinson Disease Test Results'):
        parkinsons_disease_prediction = parkinsons_disease_model.predict([[Fo,Fhi,Flo,Jitter,Jitter2,RAP,PPQ,Jitter3,Shimmer,Shimmer2,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])
        if(parkinsons_disease_prediction[0]==1):
            parkinsons_disease_diagnosis = 'The person has chances of Parkinsons Disease'
        else:
            parkinsons_disease_diagnosis = 'The person is Healthy'
    st.success(parkinsons_disease_diagnosis)
    
    