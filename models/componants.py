import streamlit as st

def title_with_bt(title="title",bt_title="button",disabled=False):
    data_grid = st.columns([3, 1])
    with data_grid[0]:
        st.subheader(title)
    with data_grid[1]:
        return st.button(bt_title,disabled=disabled)