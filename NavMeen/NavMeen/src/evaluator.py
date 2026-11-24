import pandas as pd
from config import HARBOR_COORDS, VESSEL_PROFILES
from src.vessel_economics import VesselEconomicModel
from src.route_optimizer import RouteOptimizer


class BaselineEvaluator:
    """
    Implements Section 9 Evaluation Plan:
    1. Compares NavMeen against PFZ-Only and Shortest-Route baselines.
    2. Tests across multiple boat profiles, target species, weather hazards, and offline mode.
    """

    def __init__(self, optimizer, vessel_econ_model):
        self.optimizer = optimizer
        self.econ = vessel_econ_model

    def evaluate_baselines(self, df_scored_grid, target_species, navmeen_top_df):
        results = []

        if not navmeen_top_df.empty:
            best = navmeen_top_df.iloc[0]
            results.append({
                "Method": "NavMeen (Proposed)",
                "Target_Zone": f"({int(best['target_row'])}, {int(best['target_col'])})",
                "Suitability_Score": best["fish_suitability_score"],
                "Distance_km": best["round_trip_distance_km"],
                "Travel_Time_hrs": best["travel_time_hours"],
                "Fuel_Used_L": best["estimated_fuel_liters"],
                "Fuel_Feasible": best["fuel_feasible"],
                "Safety_Violations": int(best["safety_violations"]),
                "Expected_Net_Return_INR": best["expected_net_return_inr"],
            })

        G_unconstrained, _ = self.optimizer.build_navigation_graph(
            df_scored_grid, enforce_safety=False
        )

        # Baseline A: PFZ-Only Recommendation
        offshore_df = df_scored_grid[df_scored_grid["row"] >= 2].sort_values(
            by="fish_suitability_score", ascending=False
        )
        pfz_best_row = offshore_df.iloc[0]
        pfz_coord = (int(pfz_best_row["row"]), int(pfz_best_row["col"]))
        pfz_route = self.optimizer.compute_route_metrics(G_unconstrained, pfz_coord)

        if pfz_route:
            pfz_econ = self.econ.evaluate_candidate_trip(
                pfz_route,
                pfz_best_row["fish_suitability_score"],
                target_species,
                dl_catch_estimate_kg=pfz_best_row.get("dl_expected_catch_kg", None),
            )
            results.append({
                "Method": "PFZ-Only Baseline",
                "Target_Zone": f"{pfz_coord}",
                "Suitability_Score": pfz_best_row["fish_suitability_score"],
                "Distance_km": pfz_econ["round_trip_distance_km"],
                "Travel_Time_hrs": pfz_econ["travel_time_hours"],
                "Fuel_Used_L": pfz_econ["estimated_fuel_liters"],
                "Fuel_Feasible": pfz_econ["fuel_feasible"],
                "Safety_Violations": int(pfz_route["safety_violations"]),
                "Expected_Net_Return_INR": pfz_econ["expected_net_return_inr"],
            })

        # Baseline B: Shortest Feasible Route
        G_safe, _ = self.optimizer.build_navigation_graph(
            df_scored_grid, enforce_safety=True
        )
        moderate_cells = df_scored_grid[
            (df_scored_grid["row"] >= 2)
            & (df_scored_grid["fish_suitability_score"] >= 0.45)
        ].copy()

        shortest_candidate = None
        for _, row in moderate_cells.iterrows():
            coord = (int(row["row"]), int(row["col"]))
            if coord == HARBOR_COORDS or not self.optimizer.is_cell_safe(row):
                continue
            r_metrics = self.optimizer.compute_route_metrics(G_safe, coord)
            if r_metrics is None:
                continue
            econ = self.econ.evaluate_candidate_trip(
                r_metrics,
                row["fish_suitability_score"],
                target_species,
                dl_catch_estimate_kg=row.get("dl_expected_catch_kg", None),
            )
            if econ["fuel_feasible"]:
                if (
                    shortest_candidate is None
                    or econ["round_trip_distance_km"]
                    < shortest_candidate["Distance_km"]
                ):
                    shortest_candidate = {
                        "Method": "Shortest-Route Baseline",
                        "Target_Zone": f"{coord}",
                        "Suitability_Score": row["fish_suitability_score"],
                        "Distance_km": econ["round_trip_distance_km"],
                        "Travel_Time_hrs": econ["travel_time_hours"],
                        "Fuel_Used_L": econ["estimated_fuel_liters"],
                        "Fuel_Feasible": econ["fuel_feasible"],
                        "Safety_Violations": int(r_metrics["safety_violations"]),
                        "Expected_Net_Return_INR": econ["expected_net_return_inr"],
                    }

        if shortest_candidate:
            results.append(shortest_candidate)

        return pd.DataFrame(results)

    @staticmethod
    def evaluate_multi_boat_scenarios(df_scored_grid, target_species):
        """Evaluates different vessel profiles as required in Section 9."""
        rows = []
        for profile_name, v_cfg in VESSEL_PROFILES.items():
            econ_mod = VesselEconomicModel(v_cfg)
            opt = RouteOptimizer(econ_mod)
            top_df, _ = opt.recommend_routes(df_scored_grid, target_species, top_k=1)
            if not top_df.empty:
                b = top_df.iloc[0]
                rows.append({
                    "Boat_Profile": profile_name,
                    "Usable_Fuel_L": v_cfg["current_fuel_liters"],
                    "Recommended_Zone": f"({int(b['target_row'])}, {int(b['target_col'])})",
                    "Distance_km": b["round_trip_distance_km"],
                    "Fuel_Used_L": b["estimated_fuel_liters"],
                    "Reserve_Kept_L": b["required_reserve_liters"],
                    "Expected_Catch_kg": b["expected_catch_kg"],
                    "Net_Return_INR": b["expected_net_return_inr"],
                })
        return pd.DataFrame(rows)