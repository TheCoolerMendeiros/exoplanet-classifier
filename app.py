import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('exoplanet_xgb_model.pkl')
feature_medians = pd.read_csv('feature_medians.csv', index_col=0)


# Title
st.title('🪐 Exoplanet Classifier')
st.write('Classify Kepler Objects of Interest as CONFIRMED exoplanets or FALSE POSITIVES')

# Sidebar for inputs
st.sidebar.header('Input Features')

 
koi_smet_err2 = st.sidebar.number_input('Stellar Metallicity Error (upper)', value=0.15, step=0.01)
koi_prad = st.sidebar.number_input('Planet Radius (Earth radii)', value=2.0, step=0.1, min_value=0.0)
koi_dor = st.sidebar.number_input('Distance over Star Radius', value=15.0, step=0.5, min_value=0.0)
koi_prad_err2 = st.sidebar.number_input('Planet Radius Error (upper)', value=0.2, step=0.01, min_value=0.0)
koi_teq = st.sidebar.number_input('Equilibrium Temperature (K)', value=500.0, step=10.0, min_value=0.0)


# Predict button
if st.sidebar.button('Classify'):    
    input_df = pd.DataFrame([feature_medians.squeeze()])
    input_df['koi_smet_err2'] = koi_smet_err2
    input_df['koi_prad'] = koi_prad
    input_df['koi_dor'] = koi_dor
    input_df['koi_prad_err2'] = koi_prad_err2
    input_df['koi_teq'] =  koi_teq

    input_df = input_df[model.get_booster().feature_names]
        
    


    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    st.subheader('Prediction')
    if prediction == 1:
        st.success(f'✅ CONFIRMED Exoplanet (Confidence: {probability[1]:.2%})')
    else:
        st.error(f'❌ FALSE POSITIVE (Confidence: {probability[0]:.2%})')
    
    st.write('---')
    st.write('Probability Distribution:')
    col1, col2 = st.columns(2)
    col1.metric('FALSE POSITIVE', f'{probability[0]:.2%}')
    col2.metric('CONFIRMED', f'{probability[1]:.2%}')

# Info section
st.write('---')
st.subheader('About')
st.write('''
This model was trained using data from Kepler's Objects of Interest (KOI), available through the NASA Exoplanet Archive — a curated catalog of potential and confirmed exoplanets observed by the Kepler space telescope.
         
Each KOI entry includes dozens of astrophysical and orbital parameters, such as stellar temperature, radius, metallicity, and planetary transit properties. Using this data, an XGBoost classification model was trained to distinguish between confirmed exoplanets and false positives (signals that mimic planetary transits but originate from other astrophysical sources).
         
The model achieved an accuracy of ≈96% on the test set and relies on 74 input features.
However, for simplicity and interactivity, this demo focuses on the most influential parameters in the prediction process:
- Stellar metallicity error asymmetry
- Planet radius
- Orbital distance over star radius
- Equilibrium temperature

All remaining features are filled with their median values from the training dataset, allowing the classifier to remain consistent with its original training setup while letting you explore how variations in the most important properties affect the model’s confidence.
This tool offers a glimpse into how machine learning can assist in exoplanet validation — helping astronomers prioritize promising candidates for further study.
''')