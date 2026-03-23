from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

def get_asset_path(path):
    return (BASE_DIR / "../assets" / path).resolve()

capgemini_logo = get_asset_path("images/Capgemini_201x_logo.svg")
a_energi_logo = get_asset_path("images/file.svg")

st.set_page_config(layout="wide")

# --- CSS for smooth scroll + styling ---
st.markdown("""
<style>
html {
    scroll-behavior: smooth;
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
</style>
""", unsafe_allow_html=True)


# --- HEADER ---
col1, col2 = st.columns([6, 1])

with col1:
    st.markdown("## Norgespris – Analyse av strømforbruk")

with col2:
    st.markdown("""
    <a href="#prosjektet">
        <button style="
            padding:10px 20px;
            border-radius:10px;
            border:1px solid #ccc;
            background-color:#f5f5f5;
        ">
            Prosjektet
        </button>
    </a>
    """, unsafe_allow_html=True)

st.markdown("---")


# --- FORKLAR GRAF ---
st.markdown("## Hva viser grafen?")
st.write("Forklaring av graf...")


# --- FILTER + GRAF ---
col_filter, col_graph = st.columns([1, 4])

with col_filter:
    st.markdown("### Filter")
    st.selectbox("Område", ["Samlet", "Breive", "Frikstad", "Hartevatn", "Timenes"], key="area")
    st.slider("Timer", 0, 24, (6, 18), key="hours")

with col_graph:
    st.line_chart([1, 3, 2, 5, 4, 6])


st.markdown("---")


# --- FORKLAR FUNN ---
st.markdown("## Hva fant vi?")
st.write("Forklaring av funn...")


st.markdown("---")


# --- PROSJEKTSEKSJON (ID FOR SCROLL) ---
st.markdown('<div id="prosjektet"></div>', unsafe_allow_html=True)

st.markdown("## Prosjektet")

col_img, col_text = st.columns([1, 3])

with col_img:
    st.image("https://via.placeholder.com/300")

with col_text:
    st.write("Om prosjektgruppen og samarbeid...")

st.markdown("---")

st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px;'>Samarbeidspartnere</div>",
    unsafe_allow_html=True
)

# 5 kolonner: tom - logo - logo - logo - tom
col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])

with col2:
    st.image(get_asset_path("images/file.svg"), width=60)

with col3:
    st.image(get_asset_path("images/Capgemini_201x_logo.svg"), width=100)

with col4:
    st.image(get_asset_path("images/uia.svg"), width=60)