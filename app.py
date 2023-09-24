import streamlit as st
from multiapp import MultiApp
from apps import SpamMessageClassification

app = MultiApp()

#!! Streamlit hide default texts
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;} 
    </style>
    """

#!! Streamlit Page config
st.set_page_config(page_title="DG NLP DOC")
st.header("DG NL DOC - header")
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


# Add all your application here
app.add_app("SpamMessageClassification", SpamMessageClassification.app) 


# The main app
app.run()