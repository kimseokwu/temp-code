import os
import numpy as np
import streamlit as st

st.title('super-resolution')

uploaded_file = st.file_uploader('Choose a image')
if uploaded_file is not None:
     # To read file as bytes:
     with st.spinner('loading...'):
        st.image(uploaded_file)
        st.text('Done!')