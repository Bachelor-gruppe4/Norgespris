import streamlit as st


def inject_dashboard_css():
    st.markdown(
        """
        <style>
        html {
            scroll-behavior: smooth;
        }

        .block-container {
            padding-top: 2.0rem !important;
        }

        :root {
            --aenergi-accent: #FFBBFC;
            --aenergi-deep: #3C000F;
            --aenergi-number: #7D283D;
            --aenergi-burgundy: #7D283D;
        }

        .stApp,
        .stApp * {
            color: var(--aenergi-burgundy) !important;
        }

        .logo-img {
            max-height: 50px;
        }

        .header-btn {
            padding: 10px 20px;
            border-radius: 10px;
            border: 1px solid #ccc;
            background-color: #f5f5f5;
        }

        div[data-testid="stMetric"] {
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0;
        }

        .st-key-consumption_box,
        .st-key-norgespris_box {
            background: #fff3fe;
            border: 1px solid #fff3fe;
            border-radius: 12px;
            padding: 0.55rem 0.75rem;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--aenergi-burgundy) !important;
        }

        div[data-testid="stMetricValue"] {
            color: var(--aenergi-burgundy) !important;
        }


        div[data-testid="stMultiSelect"] .stTags {
            background-color: #7D283D !important;
            color: #FFBBFC !important;
        }

        div[data-testid="stMultiSelect"] [role="button"][aria-selected="true"] {
            background-color: #7D283D !important;
            border-color: #7D283D !important;
            color: #FFBBFC !important;
        }

        div[data-testid="stMultiSelect"] > div > div {
            border-color: #7D283D !important;
        }

        div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:hover,
        div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
            border-color: #7D283D !important;
            box-shadow: 0 0 0 1px #7D283D !important;
        }

        span[data-baseweb="tag"] {
            background-color: #7D283D !important;
            color: #FFBBFC !important;
        }

        span[data-baseweb="tag"],
        span[data-baseweb="tag"] *,
        div[data-testid="stMultiSelect"] [data-baseweb="tag"],
        div[data-testid="stMultiSelect"] [data-baseweb="tag"] * {
            color: #FFBBFC !important;
            fill: #FFBBFC !important;
        }

        .station-info-box {
            background: #fff3fe;
            border: 1px solid #fff3fe;
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            margin: 0.75rem 0 1.25rem 0;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
