import streamlit as st
import time

def progress_bar(text0,text):
    progress_text = text0
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    st.success(text)


def progress_spin():
    with st.spinner('Wait for it...'):
        time.sleep(1.5)