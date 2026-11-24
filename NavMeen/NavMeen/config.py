import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
FIGURES_DIR = os.path.join(BASE_DIR, "output", "figures")

for d in [CACHE_DIR, RAW_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Spatial Grid Configuration (20x20 grid, 2 km per cell)
GRID_SIZE = 20
CELL_SIZE_KM = 2.0
HARBOR_COORDS = (0, 10)

# Preconfigured Vessel Profiles for Multi-Boat Evaluation (Section 9)
VESSEL_PROFILES = {
    "Small Artisanal Vallam (Default)": {
        "vessel_name": "TN-Vallam-01",
        "fuel_capacity_liters": 120.0,
        "current_fuel_liters": 85.0,
        "cruise_speed_kmh": 16.0,
        "base_fuel_burn_l_per_km": 0.65,
        "calibration_factor": 1.00,
        "safety_reserve_fraction": 0.20,
        "max_catch_capacity_kg": 250.0,
        "other_trip_costs_inr": 800.0,
    },
    "Motorized Gillnetter": {
        "vessel_name": "TN-Gillnet-02",
        "fuel_capacity_liters": 220.0,
        "current_fuel_liters": 160.0,
        "cruise_speed_kmh": 20.0,
        "base_fuel_burn_l_per_km": 0.95,
        "calibration_factor": 1.03,
        "safety_reserve_fraction": 0.20,
        "max_catch_capacity_kg": 450.0,
        "other_trip_costs_inr": 1500.0,
    },
    "Low-Fuel Coastal Boat (Constrained)": {
        "vessel_name": "TN-Coastal-Mini",
        "fuel_capacity_liters": 60.0,
        "current_fuel_liters": 32.0,
        "cruise_speed_kmh": 14.0,
        "base_fuel_burn_l_per_km": 0.55,
        "calibration_factor": 0.98,
        "safety_reserve_fraction": 0.25,
        "max_catch_capacity_kg": 150.0,
        "other_trip_costs_inr": 500.0,
    },
}

DEFAULT_VESSEL = VESSEL_PROFILES["Small Artisanal Vallam (Default)"].copy()

SAFETY_LIMITS = {
    "max_wave_height_m": 2.8,
    "max_wind_speed_knots": 25.0,
    "cache_validity_hours": 12.0,
}

SPECIES_CONFIG = {
    "Indian Mackerel": {
        "opt_sst_c": 28.0,
        "opt_chl_mg_m3": 1.8,
        "opt_salinity_psu": 34.0,
        "market_price_inr_per_kg": 180.0,
        "historical_prices_inr": [168.0, 172.0, 175.0, 179.0, 180.0, 184.0],
        "base_catch_yield_kg": 140.0,
    },
    "Yellowfin Tuna": {
        "opt_sst_c": 26.5,
        "opt_chl_mg_m3": 0.6,
        "opt_salinity_psu": 35.2,
        "market_price_inr_per_kg": 320.0,
        "historical_prices_inr": [300.0, 310.0, 315.0, 318.0, 320.0, 328.0],
        "base_catch_yield_kg": 110.0,
    },
    "Oil Sardine": {
        "opt_sst_c": 28.8,
        "opt_chl_mg_m3": 2.5,
        "opt_salinity_psu": 33.5,
        "market_price_inr_per_kg": 110.0,
        "historical_prices_inr": [102.0, 105.0, 108.0, 109.0, 110.0, 114.0],
        "base_catch_yield_kg": 210.0,
    },
}

FUEL_PRICE_INR_PER_LITER = 94.0