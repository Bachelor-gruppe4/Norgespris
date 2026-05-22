from pathlib import Path
import sys

import streamlit as st
import pandas as pd
import altair as alt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dashboard.config import (
    STATION_WEATHER_BETAS_BEFORE,
    STATION_WEATHER_BETAS_AFTER,
    STATION_TEXTS,
)
from src.dashboard.queries import (
    apply_weather_control,
    get_consumption_season_summary,
    get_norgespris_user_stats,
    get_season_comparison,
    get_total_users,
    get_weather_season_covariates,
    get_weather_season_summary,
)
from src.dashboard.styles import inject_dashboard_css

st.set_page_config(layout="wide")
inject_dashboard_css()


# --- HEADER ---
col1, col2 = st.columns([6, 1])

with col1:
    st.markdown(
        "<h2>Norgespris – Analyse av strømforbruk</h2>",
        unsafe_allow_html=True,
    )


st.markdown("---")


# --- FORKLAR GRAF ---
st.header("Hva viser grafen?", anchor=False)
st.write("""
Grafen viser gjennomsnittlig døgnprofil for valgt trafostasjon, basert på aggregert forbruk per time. 
Den sammenligner forbruket før og etter innføringen av Norgespris, gitt de valgte filtrene. 
Ved å bruke gjennomsnittlige døgnprofiler tones enkeltstående avvik ned, slik at forskjeller i døgnrytme, belastningstopper og overordnede forbruksmønstre blir tydeligere.
""")


# --- FILTER + GRAF ---
col_filter, col_graph = st.columns([1, 4])

with col_filter:
    st.subheader("Filter", anchor=False)
    
    # Område-valg
    område_options = ["Breive", "Frikstad", "Hartevatn", "Timenes"]
    selected_område = st.selectbox("Område", område_options, key="area")
    
    selected_day_type = st.selectbox(
        "Dagtype",
        ["Alle", "Hverdag", "Helg", "Helligdag"],
        index=0,
        key="selected_day_type"
    )

    selected_month = st.selectbox(
        "Måned",
        ["Alle", "November", "Desember", "Januar"],
        index=0,
        key="selected_month"
    )

    # Consumption code filter
    consumption_code_option = st.selectbox(
        "Forbrukstype",
        ["Boliger", "Fritidsboliger", "Begge"],
        index=2,
        key="consumption_code"
    )
    if consumption_code_option == "Boliger":
        selected_consumption_codes = [35]
    elif consumption_code_option == "Fritidsboliger":
        selected_consumption_codes = [36]
    else:
        selected_consumption_codes = [35, 36]

    weather_control_enabled = st.checkbox(
        "Kontroller for vær (regresjon)",
        value=False,
        key="weather_control_checkbox",
        help="Justerer kurvene med beta-verdier fra regresjonsanalysen."
    )

    selected_weather_controls = []
    default_weather_betas_before = STATION_WEATHER_BETAS_BEFORE.get(
        selected_område,
        {"Temperatur": 0.0, "Vind": 0.0, "Nedbør": 0.0},
    )
    default_weather_betas_after = STATION_WEATHER_BETAS_AFTER.get(
        selected_område,
        {"Temperatur": 0.0, "Vind": 0.0, "Nedbør": 0.0},
    )
    weather_betas_before = default_weather_betas_before.copy()
    weather_betas_after = default_weather_betas_after.copy()

    if weather_control_enabled:
        st.caption(f"Stasjonskoeffisienter for {selected_område}. Du kan overstyre manuelt under.")
        selected_weather_controls = st.multiselect(
            "Aktive kontroller",
            ["Temperatur", "Vind", "Nedbør"],
            default=["Temperatur", "Vind", "Nedbør"],
            key="selected_weather_controls",
        )

        beta_col_before, beta_col_after = st.columns(2)

        with beta_col_before:
            st.caption("Før Norgespris")
            weather_betas_before["Temperatur"] = st.number_input(
                "Beta temperatur",
                value=float(default_weather_betas_before["Temperatur"]),
                format="%.6f",
                key=f"beta_temperature_before_{selected_område}",
            )
            weather_betas_before["Vind"] = st.number_input(
                "Beta vind",
                value=float(default_weather_betas_before["Vind"]),
                format="%.6f",
                key=f"beta_wind_before_{selected_område}",
            )
            weather_betas_before["Nedbør"] = st.number_input(
                "Beta nedbør",
                value=float(default_weather_betas_before["Nedbør"]),
                format="%.6f",
                key=f"beta_precipitation_before_{selected_område}",
            )

        with beta_col_after:
            st.caption("Etter Norgespris")
            weather_betas_after["Temperatur"] = st.number_input(
                "Beta temperatur",
                value=float(default_weather_betas_after["Temperatur"]),
                format="%.6f",
                key=f"beta_temperature_after_{selected_område}",
            )
            weather_betas_after["Vind"] = st.number_input(
                "Beta vind",
                value=float(default_weather_betas_after["Vind"]),
                format="%.6f",
                key=f"beta_wind_after_{selected_område}",
            )
            weather_betas_after["Nedbør"] = st.number_input(
                "Beta nedbør",
                value=float(default_weather_betas_after["Nedbør"]),
                format="%.6f",
                key=f"beta_precipitation_after_{selected_område}",
            )

with col_graph:
    # Hent data basert på filtre
    with st.spinner("Henter data..."):
        df = get_season_comparison(
            område=selected_område,
            day_type=selected_day_type,
            month_filter=selected_month,
            consumption_codes=selected_consumption_codes
        )
        consumption_summary_df = get_consumption_season_summary(
            område=selected_område,
            day_type=selected_day_type,
            month_filter=selected_month,
            consumption_codes=selected_consumption_codes
        )
        weather_cov_df = get_weather_season_covariates(
            område=selected_område,
            day_type=selected_day_type,
            month_filter=selected_month,
            consumption_codes=selected_consumption_codes
        )
        weather_summary_df = get_weather_season_summary(
            område=selected_område,
            day_type=selected_day_type,
            month_filter=selected_month,
            consumption_codes=selected_consumption_codes
        )
        temp_df = weather_cov_df[["hour", "season_label", "avg_temperature"]].copy() if not weather_cov_df.empty else pd.DataFrame()
    
    if not df.empty:
        st.subheader(f"Gjennomsnittlig døgnprofil - {selected_område.title()}", anchor=False)
        st.markdown(
            f"""
            <div class="station-info-box">
                {STATION_TEXTS.get(selected_område, "")}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write(
            f"Sammenligner dagtype: {selected_day_type.lower()} "
            f"og måned: {selected_month.lower()} for sesongene \"Før Norgespris\" (2024-2025) og \"Etter Norgespris\" (2025-2026)."
        )
        
        df = df.sort_values('hour')
        plot_df = df[["hour", "season_label", "avg_forbruk"]].copy()

        if weather_control_enabled:
            if selected_weather_controls and not weather_cov_df.empty:
                plot_df = apply_weather_control(
                    profile_df=plot_df,
                    weather_df=weather_cov_df,
                    before_betas=weather_betas_before,
                    after_betas=weather_betas_after,
                    controls=selected_weather_controls,
                )
            elif selected_weather_controls and weather_cov_df.empty:
                st.info("Fant ikke værdata for valgt filter, viser ukontrollert graf.")

        if weather_control_enabled and selected_weather_controls and not weather_cov_df.empty:
            summary_df = plot_df.groupby("season_label", as_index=False)["avg_forbruk"].mean()
        else:
            summary_df = consumption_summary_df

        pivot_df = plot_df.pivot(index='hour', columns='season_label', values='avg_forbruk')

        y_min = plot_df["avg_forbruk"].min()
        y_max = plot_df["avg_forbruk"].max()

        padding = (y_max - y_min) * 0.1

        chart = alt.Chart(plot_df).mark_line(point=True).encode(
            x=alt.X("hour:O", title="Time"),
            y=alt.Y(
                "avg_forbruk:Q",
                title="Forbruk (kWh)",
                scale=alt.Scale(domain=[y_min - padding, y_max + padding])
            ),
            color=alt.Color(
                "season_label:N",
                title="Sesong",
                scale=alt.Scale(
                    domain=["Før Norgespris", "Etter Norgespris"],
                    range=["#FFBBFC", "#7D283D"],
                ),
            ),
            tooltip=[
                    alt.Tooltip("hour:O", title="Time"),
                    alt.Tooltip("avg_forbruk:Q", title="Forbruk (kWh)", format=".2f"),
                    alt.Tooltip("season_label:N", title="Sesong")
]
        ).properties(height=400)

        st.altair_chart(chart, use_container_width=True)

        # Vis statistikk
        norgespris_users = get_norgespris_user_stats(
            område=selected_område,
            month_filter=selected_month
        )

        total_users = get_total_users(
            område=selected_område
        )
        
        # Bruk consumption_summary_df for gjennomsnittene (beregnet fra rådata)
        before_avg = None
        after_avg = None
        if not summary_df.empty:
            before_row = summary_df[summary_df["season_label"] == "Før Norgespris"]
            after_row = summary_df[summary_df["season_label"] == "Etter Norgespris"]
            if not before_row.empty:
                before_avg = before_row["avg_forbruk"].iloc[0]
            if not after_row.empty:
                after_avg = after_row["avg_forbruk"].iloc[0]

        stats_col_left, stats_col_right = st.columns([2.6, 1.4])

        with stats_col_left:
            with st.container(border=False, key="consumption_box"):
                st.markdown("**Gjennomsnittlig forbruk**")

                col_stats1, col_stats2, col_stats3 = st.columns(3)

                with col_stats1:
                    if before_avg is not None:
                        st.metric("Før Norgespris", f"{before_avg:.2f} kWh")

                with col_stats2:
                    if after_avg is not None:
                        st.metric("Etter Norgespris", f"{after_avg:.2f} kWh")
                with col_stats3:
                    if before_avg is not None and after_avg is not None:
                        if before_avg > 0:
                            change_pct = ((after_avg - before_avg) / before_avg) * 100
                            change_str = f"{change_pct:+.2f}%"
                        else:
                            change_str = "N/A"
                        st.metric("Prosentvis endring", change_str)

        with stats_col_right:
            with st.container(border=False, key="norgespris_box"):
                st.markdown("**Norgespris-brukere**")
                st.metric(
                    f"Brukere i {selected_område}",
                    f"{norgespris_users} av {total_users}"
                )

        if not temp_df.empty:
            st.subheader("Gjennomsnittlig temperatur (°C)", anchor=False)
            temp_df = temp_df.sort_values("hour")
            temp_pivot_df = temp_df.pivot(index="hour", columns="season_label", values="avg_temperature")

            temp_before_avg = None
            temp_after_avg = None
            if not weather_summary_df.empty:
                before_row = weather_summary_df[weather_summary_df["season_label"] == "Før Norgespris"]
                after_row = weather_summary_df[weather_summary_df["season_label"] == "Etter Norgespris"]
                if not before_row.empty:
                    temp_before_avg = before_row["avg_temperature"].iloc[0]
                if not after_row.empty:
                    temp_after_avg = after_row["avg_temperature"].iloc[0]
            else:
                temp_avg_per_season = temp_pivot_df.mean()
                temp_before_avg = temp_avg_per_season.get("Før Norgespris", None)
                temp_after_avg = temp_avg_per_season.get("Etter Norgespris", None)

            temp_plot_df = temp_pivot_df.reset_index().melt(
                id_vars="hour",
                var_name="season_label",
                value_name="avg_temperature"
            )

            temp_y_min = temp_plot_df["avg_temperature"].min()
            temp_y_max = temp_plot_df["avg_temperature"].max()
            temp_padding = (temp_y_max - temp_y_min) * 0.1

            chart_temp = alt.Chart(temp_plot_df).mark_line(point=True).encode(
                x=alt.X("hour:O", title="Time"),
                y=alt.Y(
                    "avg_temperature:Q",
                    title="Temperatur (°C)",
                    scale=alt.Scale(domain=[temp_y_min - temp_padding, temp_y_max + temp_padding])
                ),
                color=alt.Color(
                    "season_label:N",
                    title="Sesong",
                    scale=alt.Scale(
                        domain=["Før Norgespris", "Etter Norgespris"],
                        range=["#FFBBFC", "#7D283D"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("hour:O", title="Time"),
                    alt.Tooltip("avg_temperature:Q", title="Temperatur (°C)", format=".2f"),
                    alt.Tooltip("season_label:N", title="Sesong")
                ]
            ).properties(height=400)

            st.altair_chart(chart_temp, use_container_width=True)


            temp_stats_col_left, temp_stats_col_right = st.columns([2.6, 1.4])

            with temp_stats_col_left:
                st.markdown("**Gjennomsnittlig daglig temperatur**")
                temp_col1, temp_col2, temp_col3 = st.columns(3)
                with temp_col1:
                    if temp_before_avg is not None:
                        st.metric("Før Norgespris", f"{temp_before_avg:.2f} °C")
                with temp_col2:
                    if temp_after_avg is not None:
                        st.metric("Etter Norgespris", f"{temp_after_avg:.2f} °C")
                with temp_col3:
                    if temp_before_avg is not None and temp_after_avg is not None:
                        temp_diff = temp_after_avg - temp_before_avg
                        st.metric("Differanse (Etter - Før)", f"{temp_diff:+.2f} °C")
                    else:
                        st.metric("Differanse (Etter - Før)", "N/A")
        
            
    else:
        if selected_day_type == "Helligdag" and selected_month in ["November", "Januar"]:
            st.warning(
                "Ingen data funnet: Helligdager finnes kun i desember i dette datasettet. "
                "Velg måned Desember eller endre dagtype."
            )
        else:
            st.warning("Ingen data funnet for valgte filtre. Prøv å justere filtrene eller området.")
        
        # Vis eksempel på tilgjengelige data
        st.markdown("**Tilgjengelige områder:**")
        for område in område_options:
            try:
                test_df = get_forbruksdata(område, limit=1)
                if not test_df.empty:
                    st.write(f" {område.title()}: Data tilgjengelig")
                else:
                    st.write(f" {område.title()}: Ingen data")
            except:
                st.write(f" {område.title()}: Feil ved tilkobling")