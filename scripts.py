

''' upload a file ''''

import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image

st.set_page_config(page_title='First test')
st.header('First test')
st.subheader('Was the tutorial helpful?')


# import streamlit as st
import os.path
import pathlib
st.write("""
# File Picker
""")
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = uploaded_file.getvalue().decode('utf-8').splitlines()
    st.session_state["preview"] = ''
    for i in range(0, min(5, len(data))):
        st.session_state["preview"] += data[i]
preview = st.text_area("CSV Preview", "", height=150, key="preview")
upload_state = st.text_area("Upload State", "", key="upload_state")

# This is the place where to save the data you uploaded
def upload():
    if uploaded_file is None:
        st.session_state["upload_state"] = "Upload a file first!"
    else:
        data = uploaded_file.getvalue().decode('utf-8')
        parent_path = pathlib.Path(__file__).parent.parent.resolve()
        save_path = os.path.join(parent_path, "data")
        complete_name = os.path.join(save_path, uploaded_file.name)
        destination_file = open(complete_name, "w")
        destination_file.write(data)
        destination_file.close()
        st.session_state["upload_state"] = "Saved " + complete_name + " successfully!"
st.button("Upload file to Sandbox", on_click=upload)

import pathlib
from os import listdir
from os.path import isfile, join
st.write("""# Demo """)
parent_path = pathlib.Path(__file__).parent.parent.resolve()
data_path = os.path.join(parent_path, "data")
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
option = st.sidebar.selectbox('Pick a dataset', onlyfiles)
file_location=os.path.join(data_path, option)
# use `file_location` as a parameter to the main script

''' next 
  '''