import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import (
    VESSEL_PROFILES,
    SPECIES_CONFIG,
    HARBOR_COORDS,
    GRID_SIZE,
    FUEL_PRICE_INR_PER_LITER,
)
from src.data_manager import DataManager
from src.pfz_model import SpeciesPFZModel
from src.vessel_economics import VesselEconomicModel
from src.route_optimizer import RouteOptimizer
from src.evaluator import BaselineEvaluator

st.set_page_config(page_title="NavMeen Decision-Support System", layout="wide")

st.title("NavMeen: AI-Driven Fishing Route Decision-Support System")
st.caption(
    "Integrated Prototype: PyTorch Multi-Task Deep Learning + OpenCV PFZ Processing + "
    "Vessel Fuel & Safety Constrained Route Optimizer + Offline GPS Navigation"
)

# Sidebar Stage 1: Vessel Profile & Trip Form
st.sidebar.header("1. Vessel & Fuel Profile")
selected_profile_name = st.sidebar.selectbox("Preset Boat Profile", list(VESSEL_PROFILES.keys()))
base_vessel = VESSEL_PROFILES[selected_profile_name].copy()

target_species = st.sidebar.selectbox("Target Fish Species", list(SPECIES_CONFIG.keys()))
routing_algo = st.sidebar.radio("Route Optimization Algorithm", ["A*", "Dijkstra"], horizontal=True)
use_forecast_price = st.sidebar.checkbox("Use Historical Price Forecaster", value=False)

default_price = (
    VesselEconomicModel.forecast_species_price(target_species)
    if use_forecast_price
    else float(SPECIES_CONFIG[target_species]["market_price_inr_per_kg"])
)
market_price = st.sidebar.number_input("Sale Price (INR / kg)", 50.0, 1000.0, default_price, 5.0)

fuel_capacity = st.sidebar.number_input("Fuel Tank Capacity (L)", 30.0, 400.0, float(base_vessel["fuel_capacity_liters"]), 10.0)
current_fuel = st.sidebar.slider("Current Usable Fuel (L)", 15.0, float(fuel_capacity), float(base_vessel["current_fuel_liters"]), 5.0)
reserve_pct = st.sidebar.slider("Mandatory Return Reserve (%)", 10, 40, int(base_vessel["safety_reserve_fraction"] * 100), 5)
cruise_speed = st.sidebar.slider("Vessel Cruise Speed (km/h)", 10.0, 28.0, float(base_vessel["cruise_speed_kmh"]), 1.0)
fuel_burn = st.sidebar.slider("Base Fuel Consumption (L/km)", 0.30, 1.50, float(base_vessel["base_fuel_burn_l_per_km"]), 0.05)

st.sidebar.header("2. Weather, Offline Cache & GPS")
simulate_storm = st.sidebar.checkbox("SimulateIMD Offshore Wave/Wind Hazard", value=True)
cache_age_hours = st.sidebar.slider("Cached Forecast Age (Hours)", 0.5, 24.0, 2.0, 0.5)
offline_mode = st.sidebar.checkbox("Simulate Offshore Offline Mode (No Internet)", value=False)

vessel_profile = base_vessel.copy()
vessel_profile.update({
    "fuel_capacity_liters": fuel_capacity,
    "current_fuel_liters": current_fuel,
    "safety_reserve_fraction": reserve_pct / 100.0,
    "cruise_speed_kmh": cruise_speed,
    "base_fuel_burn_l_per_km": fuel_burn,
})

dm = DataManager()
if "weather_merged" not in st.session_state:
    st.session_state.weather_merged = False

if not offline_mode and not st.session_state.weather_merged:
    dm.generate_and_cache_layers(seed=42, simulate_storm=simulate_storm, hours_old=cache_age_hours)

df_hist = dm.generate_historical_training_data()
df_grid, meta = dm.load_offline_cache()

pfz_model = SpeciesPFZModel()
pfz_model.load_or_train(df_hist)
df_scored = pfz_model.predict_grid_suitability(df_grid, target_species)

econ_model = VesselEconomicModel(vessel_profile)
optimizer = RouteOptimizer(econ_model, algorithm=routing_algo)
top_routes_df, _ = optimizer.recommend_routes(
    df_scored,
    target_species,
    market_price_override=market_price,
    use_price_forecast=use_forecast_price,
    top_k=5,
)

# Status Banner (Methodology Section 6 & 8)
if meta["is_stale"]:
    st.error(
        f"STALE FORECAST WARNING: Cached layers are {meta['age_hours']} hrs old "
        f"(Validity limit: {meta['validity_hours']} hrs). Do not rely on outdated weather data."
    )
elif offline_mode:
    st.warning(
        f"OFFLINE GPS NAVIGATION MODE ACTIVE | Using Locally Cached Layers from {meta['issue_timestamp_utc']}"
    )
else:
    st.success(
        f"System Online & Cache Verified | Issued: {meta['issue_timestamp_utc']} "
        f"(Age: {meta['age_hours']} hrs | Validity: {meta['validity_hours']} hrs)"
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Route Dashboard & Offline GPS (Fig. 2)",
    "2. Deep Learning & OpenCV PFZ Analytics",
    "3. Baseline & Multi-Boat Evaluation (Sec. 9)",
    "4. Cached Layer Metadata & Economic Assumptions",
])

with tab1:
    if top_routes_df.empty:
        st.error("No feasible routes satisfy both the mandatory fuel reserve and IMD wave/wind safety thresholds.")
    else:
        best = top_routes_df.iloc[0]
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Recommended Zone", f"({int(best['target_row'])}, {int(best['target_col'])})")
        k2.metric("DL Suitability Score", f"{best['fish_suitability_score']:.2f}")
        k3.metric("Expected Catch", f"{best['expected_catch_kg']} kg")
        k4.metric("Round-Trip Distance", f"{best['round_trip_distance_km']} km")
        k5.metric("Est. Fuel (Reserve)", f"{best['estimated_fuel_liters']} L ({best['required_reserve_liters']} L)")
        k6.metric("Expected Net Return", f"INR {best['expected_net_return_inr']:,.0f}")

        col_map, col_details = st.columns([1.25, 1.0])

        with col_map:
            st.subheader(f"Personalized Route Map ({routing_algo} Optimizer)")
            waypoints = best["waypoints"]
            gps_step = st.slider(
                "Offline GPS Boat Progress Along Waypoints",
                0, len(waypoints) - 1, min(2, len(waypoints) - 1)
            )
            current_gps_wp = waypoints[gps_step]

            fig, ax = plt.subplots(figsize=(7, 5.8))
            suit_mat = df_scored.pivot(index="row", columns="col", values="fish_suitability_score").values
            im = ax.imshow(suit_mat, origin="lower", cmap="YlGnBu", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, label=f"{target_species} Suitability Probability")

            for _, r_cell in df_scored.iterrows():
                if not optimizer.is_cell_safe(r_cell):
                    ax.scatter(r_cell["col"], r_cell["row"], marker="x", color="red", s=50)

            wp_r = [p[0] for p in waypoints]
            wp_c = [p[1] for p in waypoints]
            ax.plot(wp_c, wp_r, color="#FF8C00", lw=3, label=f"NavMeen {routing_algo} Route")
            ax.scatter(HARBOR_COORDS[1], HARBOR_COORDS[0], color="lime", s=120, edgecolors="black", zorder=5, label="Harbor")
            ax.scatter(current_gps_wp[1], current_gps_wp[0], color="cyan", marker="^", s=150, edgecolors="black", zorder=6, label="Live Boat GPS")
            ax.scatter(best["target_col"], best["target_row"], color="gold", marker="*", s=220, edgecolors="black", zorder=5, label="Target PFZ")

            ax.set_xlabel("Grid Column (2 km/cell East-West)")
            ax.set_ylabel("Grid Row (2 km/cell Offshore)")
            ax.set_xlim(-0.5, GRID_SIZE - 0.5)
            ax.set_ylim(-0.5, GRID_SIZE - 0.5)
            ax.legend(loc="upper left", fontsize=8)
            st.pyplot(fig)

        with col_details:
            st.subheader("Top 5 Ranked Candidate Routes")
            st.dataframe(
                top_routes_df[[
                    "rank", "target_row", "target_col", "fish_suitability_score",
                    "expected_catch_kg", "round_trip_distance_km",
                    "estimated_fuel_liters", "expected_net_return_inr", "weather_status"
                ]],
                use_container_width=True,
            )

            st.subheader("Offline Dynamic Forecast Update & Route Recomputation (Sec. 8)")
            c_btn1, c_btn2 = st.columns(2)
            if c_btn1.button("Simulate New IMD Squall Update & Merge"):
                dm.merge_live_weather_update(new_storm_center=(4, 10))
                st.session_state.weather_merged = True
                st.rerun()
            if c_btn2.button("Reset Weather Cache to Default"):
                st.session_state.weather_merged = False
                dm.generate_and_cache_layers(seed=42, simulate_storm=simulate_storm, hours_old=cache_age_hours)
                st.rerun()

            st.info(
                "**Advisory Note:** High suitability scores indicate environmental preference, not a guaranteed catch. "
                "This decision-support tool complements—and does not replace—official IMD marine warnings and onboard navigation equipment."
            )

with tab2:
    st.subheader("PyTorch Deep Learning Multi-Task Metrics & OpenCV PFZ Layer")
    m_df = pd.DataFrame(pfz_model.metrics).T
    st.write("**Out-of-Time Test Split Metrics (Precision, Recall, F1, ROC-AUC, Calibration Loss):**")
    st.dataframe(m_df, use_container_width=True)

    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        st.markdown("**Epoch Training & Validation Loss / Accuracy Curves**")
        if target_species in pfz_model.training_history:
            hist = pfz_model.training_history[target_species]
            fig_h, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(8, 3.5))
            ax_l.plot(hist["train_loss"], label="Train Loss")
            ax_l.plot(hist["val_loss"], linestyle="--", label="Val Loss")
            ax_l.set_title(f"{target_species} Loss")
            ax_l.legend()

            ax_a.plot(hist["train_acc"], label="Train Acc")
            ax_a.plot(hist["val_acc"], linestyle="--", label="Val Acc")
            ax_a.set_title(f"{target_species} Accuracy")
            ax_a.legend()
            plt.tight_layout()
            st.pyplot(fig_h)

    with c_dl2:
        st.markdown("**OpenCV Processed PFZ Thermal-Front & Chlorophyll Index**")
        fig_cv, ax_cv = plt.subplots(figsize=(5.5, 3.8))
        cv_mat = df_scored.pivot(index="row", columns="col", values="opencv_pfz_index").values
        im_cv = ax_cv.imshow(cv_mat, origin="lower", cmap="magma")
        plt.colorbar(im_cv, ax=ax_cv, label="OpenCV PFZ Feature Score")
        ax_cv.set_title("OpenCV Smoothed & Sobel Front Layer")
        st.pyplot(fig_cv)

    st.write("**Sample Grid Predictions (Including Age, Gender Ratio, & Catch Annotations):**")
    st.dataframe(
        df_scored[[
            "row", "col", "sst_c", "chl_a_mg_m3", "opencv_pfz_index",
            "fish_suitability_score", "dl_expected_catch_kg",
            "estimated_female_ratio", "estimated_age_months"
        ]].head(10),
        use_container_width=True,
    )

with tab3:
    st.subheader("Section 9: Experimental Evaluation Against Baselines")
    evaluator = BaselineEvaluator(optimizer, econ_model)
    eval_df = evaluator.evaluate_baselines(df_scored, target_species, top_routes_df)
    st.dataframe(eval_df, use_container_width=True)

    st.subheader("Multi-Boat Profile Sensitivity Analysis")
    boat_df = evaluator.evaluate_multi_boat_scenarios(df_scored, target_species)
    st.dataframe(boat_df, use_container_width=True)

with tab4:
    st.subheader("Section 1 & 2: Cached Data Layer Registry (Timestamps, Units, & Quality Flags)")
    st.dataframe(pd.DataFrame(meta.get("layer_registry", [])), use_container_width=True)

    st.subheader("Section 4 & 5: Explicit Economic & Fuel Model Assumptions")
    st.markdown(f"""
    * **Objective Function:** $\\text{{Expected Net Return}} = \\text{{Gross Revenue}} - \\text{{Fuel Cost}} - \\text{{Other Trip Costs}} - \\text{{Weather Risk Penalty}}$
    * **Configured Fuel Price:** INR {FUEL_PRICE_INR_PER_LITER:.2f} / Liter
    * **Selected Species Price:** INR {market_price:.2f} / kg ({'Historical Linear Trend Forecast' if use_forecast_price else 'Latest Verified Harbor Price'})
    * **Fixed Trip Costs (Ice, Bait, Rations):** INR {vessel_profile['other_trip_costs_inr']:.2f}
    * **Mandatory Return Safety Reserve:** {reserve_pct}% of current fuel ({current_fuel * (reserve_pct / 100.0):.1f} Liters) reserved for return & contingencies.
    * **Uncertainty Assumption:** Catch volume and market sale price are probabilistic estimates; environmental suitability does not guarantee catch quantity.
    """)