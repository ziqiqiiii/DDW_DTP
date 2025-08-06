import streamlit as st
import pandas as pd
import numpy as np
import time
from utils.Utils import get_column_options

st.set_page_config(page_title="Plotting Demo", page_icon="📈")

st.markdown("# Predictive Food Waste")
st.sidebar.header("Model Demo")
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

url = './datasets/food_wastage_data.csv'
independent_variables = ['Type of Food', 'Event Type', 'Preparation Method', 'Pricing', 'Geographical Location', 'Number of Guests']
options = get_column_options(url)

with st.form("predict_food_waster", clear_on_submit=True):
    result = {}
    for iv in independent_variables:
        op = options[iv]
        if len(op) == 1:
            result[iv] = st.slider(iv, 0, 1000, 150)
        else:
            result[iv] = st.selectbox(iv, op)
    submit = st.form_submit_button("Predict !!!")

if submit:
        # st.rerun()
    # if new_username and new_name:
    #     users.loc[len(users)] = [len(users), new_username, new_name]
    #     with pd.ExcelWriter(filename, mode='a', if_sheet_exists='replace') as f:
    #         users.to_excel(f, sheet_name="Users", index=False)
    st.write("")