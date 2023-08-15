#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model


# In[2]:


model = load_model('my_pipeline')


# In[3]:


st.title('Where in the UK App')

# Collect user input
mm = st.number_input('Month of the year', min_value=1, max_value=12)
tmax = st.number_input('Mean daily maximum temperature this month ºC', min_value=0.0, max_value=40.0)
tmin = st.number_input('Mean daily minimum temperature this month ºC', min_value=0.0, max_value=30.0)
af = st.number_input('Days of air frost this month', min_value=0.0, max_value=28.0)
rain = st.number_input('Total mm of rainfall this month', min_value=0.0, max_value=600.0)

# Predict the output
if st.button('Predict region of UK'):
  input_data = pd.DataFrame([[mm, tmax, tmin, af, rain]],
  columns=['mm', 'tmax', 'tmin', 'af', 'rain'])
  prediction = predict_model(model, data=input_data)
  st.write(f"I think you must be in the  {prediction['prediction_label'].iloc[0]}")


# In[ ]:




