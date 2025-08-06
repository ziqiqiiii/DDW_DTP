import streamlit as st
import pandas as pd
import numpy as np
from utils.Utils import get_column_options
from library import predict_food_waste

st.set_page_config(page_title="Food Waste Predictor", page_icon="📈")

url = './datasets/food_wastage_data.csv'
independent_variables = ['Type of Food', 'Event Type', 'Preparation Method', 'Pricing', 'Geographical Location', 'Quantity of Food']
options = get_column_options(url)

# Initialize session state
for iv in independent_variables:
    if iv not in st.session_state:
        st.session_state[iv] = options[iv][0] if len(options[iv]) > 1 else 150  # default values

with st.form("predict_food_waster"):
    for iv in independent_variables:
        if len(options[iv]) == 1:
            st.session_state[iv] = st.slider(iv, 0, 1000, int(st.session_state[iv]))
        else:
            st.session_state[iv] = st.selectbox(iv, options[iv], index=options[iv].index(st.session_state[iv]))

    submit = st.form_submit_button("Predict !!!")

if submit:
    input_data = {iv: st.session_state[iv] for iv in independent_variables}
    prediction = predict_food_waste(input_data)
    st.success(f"🧠 Predicted Food Waste: {prediction:.2f} kg")
    