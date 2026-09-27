"""
RetinaTriage — Streamlit entry point.

Streamlit Community Cloud looks for `streamlit_app.py` at the repository root,
so this file is the router and the two real pages live in `views/`.

`st.navigation` rather than a `pages/` folder, because that folder derives its
sidebar labels from filenames — the entry script shows up as "streamlit app",
which is not what the tool is called. Declaring the pages here lets them be
named properly.

    streamlit run streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="RetinaTriage",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

navigation = st.navigation(
    [
        st.Page("views/reader.py", title="Read an image", icon="👁️", default=True),
        st.Page("views/evidence.py", title="Model evidence", icon="📊"),
    ]
)
navigation.run()
