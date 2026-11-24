import numpy as np
from config import (
    SPECIES_CONFIG,
    FUEL_PRICE_INR_PER_LITER,
    SAFETY_LIMITS,
)


class VesselEconomicModel:
    """
    Computes calibrated segment fuel use, enforces return-fuel and safety reserve limits,
    supports latest price or forecasted market price, and calculates Expected Net Return.
    """

    def __init__(self, vessel_profile):
        self.vessel = vessel_profile

    @staticmethod
    def forecast_species_price(target_species):
        """
        Simple linear trend forecaster on historical landing prices
        (Methodology Section 4: Market and Expected-Return Estimation).
        """
        hist = SPECIES_CONFIG[target_species]["historical_prices_inr"]
        x = np.arange(len(hist))
        slope, intercept = np.polyfit(x, hist, 1)
        forecast_price = slope * len(hist) + intercept
        return round(float(forecast_price), 2)

    def calculate_segment_fuel(self, distance_km, wave_height_m, wind_speed_knots):
        """
        Estimates fuel consumption for a route segment using vessel speed profile,
        calibration factor, and sea-state resistance.
        """
        base_burn = self.vessel["base_fuel_burn_l_per_km"]
        cal_factor = self.vessel.get("calibration_factor", 1.0)
        speed_factor = (self.vessel["cruise_speed_kmh"] / 16.0) ** 1.15
        wave_factor = 1.0 + 0.12 * max(0.0, wave_height_m - 0.8)
        wind_factor = 1.0 + 0.01 * max(0.0, wind_speed_knots - 10.0)
        return distance_km * base_burn * cal_factor * speed_factor * wave_factor * wind_factor

    def evaluate_candidate_trip(
        self,
        route_metrics,
        suitability_score,
        target_species,
        market_price_override=None,
        use_price_forecast=False,
        dl_catch_estimate_kg=None,
    ):
        sp_cfg = SPECIES_CONFIG[target_species]
        if market_price_override is not None:
            price_per_kg = market_price_override
        elif use_price_forecast:
            price_per_kg = self.forecast_species_price(target_species)
        else:
            price_per_kg = sp_cfg["market_price_inr_per_kg"]

        loiter_fuel_liters = 1.5
        round_trip_distance_km = route_metrics["one_way_distance_km"] * 2.0
        round_trip_fuel_liters = (route_metrics["one_way_fuel_liters"] * 2.0) + loiter_fuel_liters
        travel_time_hours = round_trip_distance_km / max(1.0, self.vessel["cruise_speed_kmh"])

        current_fuel = self.vessel["current_fuel_liters"]
        required_reserve = current_fuel * self.vessel["safety_reserve_fraction"]
        max_usable_fuel = current_fuel - required_reserve
        fuel_feasible = round_trip_fuel_liters <= max_usable_fuel

        # Combine Deep Learning catch head with suitability score, bounded by boat capacity
        raw_catch = (
            0.5 * dl_catch_estimate_kg + 0.5 * (sp_cfg["base_catch_yield_kg"] * suitability_score)
            if dl_catch_estimate_kg is not None
            else sp_cfg["base_catch_yield_kg"] * suitability_score
        )
        expected_catch_kg = min(self.vessel["max_catch_capacity_kg"], raw_catch)

        gross_revenue_inr = expected_catch_kg * price_per_kg
        fuel_cost_inr = round_trip_fuel_liters * FUEL_PRICE_INR_PER_LITER
        other_costs_inr = self.vessel["other_trip_costs_inr"]

        max_wave = route_metrics["max_route_wave_m"]
        avg_wave = route_metrics["avg_route_wave_m"]
        wave_ratio = avg_wave / SAFETY_LIMITS["max_wave_height_m"]
        risk_penalty_inr = 1200.0 * (wave_ratio ** 2)

        expected_net_return_inr = (
            gross_revenue_inr - fuel_cost_inr - other_costs_inr - risk_penalty_inr
        )

        return {
            "round_trip_distance_km": round(round_trip_distance_km, 2),
            "travel_time_hours": round(travel_time_hours, 2),
            "estimated_fuel_liters": round(round_trip_fuel_liters, 2),
            "required_reserve_liters": round(required_reserve, 2),
            "remaining_fuel_after_trip_liters": round(current_fuel - round_trip_fuel_liters, 2),
            "fuel_feasible": bool(fuel_feasible),
            "expected_catch_kg": round(expected_catch_kg, 1),
            "market_price_inr_kg": round(price_per_kg, 2),
            "gross_revenue_inr": round(gross_revenue_inr, 2),
            "fuel_cost_inr": round(fuel_cost_inr, 2),
            "other_costs_inr": round(other_costs_inr, 2),
            "risk_penalty_inr": round(risk_penalty_inr, 2),
            "expected_net_return_inr": round(expected_net_return_inr, 2),
            "max_route_wave_m": round(max_wave, 2),
            "weather_status": "SAFE" if max_wave <= 2.0 else "MODERATE CAUTION",
        }