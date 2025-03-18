import streamlit as st
from sections.Data import data
from sections.Models import models
from sections.Preprocessing import preprocessing
from sections.HomePage import home
from streamlit_option_menu import option_menu

st.set_page_config(page_title="FutureVision", layout="wide",page_icon=":material/network_intelligence:")
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

st.markdown('''
<style>
.stApp [data-testid="stToolbar"]{
    display:none;
}
</style>
''', unsafe_allow_html=True)


with st.sidebar:
    styles={"container": {"background-color": "transparent"}}
    selected = option_menu('FutureVision',["Home","Data", 'Preprocessing','Models'],default_index=0,
                            icons=['house-door-fill','database-fill', 'cpu-fill','robot'],
                            menu_icon="eye-fill",
                            orientation="vertical",
                            styles=styles)
    st.divider()

if selected =='Data':
    data()

elif selected == 'Preprocessing':
    preprocessing()

elif selected == 'Models':
    models()

elif selected == 'Home':
    home()