##Created by Inderpreet RANA, Hina Hussain, 

import streamlit as st
from Pages import home, var_selection, pdp, ice, ale, shapley, model_perf, final_model
from pathlib import Path


PAGES = {
    "Home": home,
    "Model Performances": model_perf,
    "Variable Selection":var_selection,
    "PDP": pdp,
    "ICE": ice,
    "ALE": ale,
    "SHAP": shapley,
    "Final Model Selected": final_model,
}


def main():
    favicon = "📱"
    st.set_page_config(layout="wide",
    page_title='Customer Churn Prediction', page_icon = favicon, initial_sidebar_state = 'auto')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    } 
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)   
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    with st.spinner(f'Loading {selection} ...'):
        page.run()


if __name__ == "__main__":
    main()