import numpy as np
import pandas as pd
import networkx as nx
from config import GRID_SIZE, CELL_SIZE_KM, HARBOR_COORDS, SAFETY_LIMITS


class RouteOptimizer:
    """
    Builds a spatial navigation graph, filters out unsafe/restricted cells,
    supports both A* Search and Weighted Dijkstra pathfinding (Methodology Section 7),
    and ranks candidate routes by Expected Net Return.
    """

    def __init__(self, vessel_econ_model, safety_limits=SAFETY_LIMITS, algorithm="A*"):
        self.econ_model = vessel_econ_model
        self.safety_limits = safety_limits
        self.algorithm = algorithm  # "A*" or "Dijkstra"

    def is_cell_safe(self, cell_row):
        if int(cell_row["restricted_area"]) == 1:
            return False
        if int(cell_row["official_warning"]) == 1:
            return False
        if cell_row["wave_height_m"] > self.safety_limits["max_wave_height_m"]:
            return False
        if cell_row["wind_speed_knots"] > self.safety_limits["max_wind_speed_knots"]:
            return False
        return True

    def build_navigation_graph(self, df_grid, enforce_safety=True):
        G = nx.Graph()
        cell_lookup = {}

        for _, row in df_grid.iterrows():
            coord = (int(row["row"]), int(row["col"]))
            safe = self.is_cell_safe(row)
            cell_lookup[coord] = row.to_dict()
            cell_lookup[coord]["is_safe"] = safe

            if not enforce_safety or safe or coord == HARBOR_COORDS:
                G.add_node(coord, **cell_lookup[coord])

        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                u = (r, c)
                if u not in G:
                    continue
                for dr, dc in directions:
                    v = (r + dr, c + dc)
                    if v in G and u < v:
                        step_dist_km = CELL_SIZE_KM * np.hypot(dr, dc)
                        u_data = cell_lookup[u]
                        v_data = cell_lookup[v]

                        avg_wave = 0.5 * (u_data["wave_height_m"] + v_data["wave_height_m"])
                        avg_wind = 0.5 * (u_data["wind_speed_knots"] + v_data["wind_speed_knots"])

                        seg_fuel = self.econ_model.calculate_segment_fuel(
                            step_dist_km, avg_wave, avg_wind
                        )
                        edge_cost = seg_fuel * (1.0 + 0.25 * (avg_wave ** 2))

                        G.add_edge(
                            u,
                            v,
                            distance_km=step_dist_km,
                            fuel_liters=seg_fuel,
                            avg_wave_m=avg_wave,
                            weight=edge_cost,
                        )
        return G, cell_lookup

    @staticmethod
    def _heuristic(u, v):
        return np.hypot(u[0] - v[0], u[1] - v[1]) * CELL_SIZE_KM * 0.5

    def compute_route_metrics(self, G, target_coord):
        if target_coord not in G or not nx.has_path(G, HARBOR_COORDS, target_coord):
            return None

        if self.algorithm == "Dijkstra":
            path = nx.dijkstra_path(G, HARBOR_COORDS, target_coord, weight="weight")
        else:
            path = nx.astar_path(
                G, HARBOR_COORDS, target_coord, heuristic=self._heuristic, weight="weight"
            )

        total_dist = 0.0
        total_fuel = 0.0
        waves = []
        safety_violations = 0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = G[u][v]
            total_dist += edge["distance_km"]
            total_fuel += edge["fuel_liters"]
            waves.append(G.nodes[v]["wave_height_m"])
            if not G.nodes[v]["is_safe"]:
                safety_violations += 1

        return {
            "waypoints": path,
            "one_way_distance_km": total_dist,
            "one_way_fuel_liters": total_fuel,
            "max_route_wave_m": max(waves) if waves else 0.0,
            "avg_route_wave_m": float(np.mean(waves)) if waves else 0.0,
            "safety_violations": safety_violations,
        }

    def recommend_routes(
        self,
        df_scored_grid,
        target_species,
        market_price_override=None,
        use_price_forecast=False,
        top_k=5,
    ):
        G_safe, _ = self.build_navigation_graph(df_scored_grid, enforce_safety=True)
        candidates = []

        for _, row in df_scored_grid.iterrows():
            coord = (int(row["row"]), int(row["col"]))
            if coord == HARBOR_COORDS or coord[0] < 2:
                continue
            if not self.is_cell_safe(row):
                continue

            route_metrics = self.compute_route_metrics(G_safe, coord)
            if route_metrics is None:
                continue

            dl_catch = row.get("dl_expected_catch_kg", None)
            econ = self.econ_model.evaluate_candidate_trip(
                route_metrics=route_metrics,
                suitability_score=row["fish_suitability_score"],
                target_species=target_species,
                market_price_override=market_price_override,
                use_price_forecast=use_price_forecast,
                dl_catch_estimate_kg=dl_catch,
            )

            if not econ["fuel_feasible"]:
                continue

            candidates.append({
                "target_row": coord[0],
                "target_col": coord[1],
                "lat": row["lat"],
                "lon": row["lon"],
                "fish_suitability_score": row["fish_suitability_score"],
                "opencv_pfz_index": row.get("opencv_pfz_index", 0.0),
                "estimated_female_ratio": row.get("estimated_female_ratio", 0.5),
                "estimated_age_months": row.get("estimated_age_months", 18.0),
                "waypoints": route_metrics["waypoints"],
                "safety_violations": route_metrics["safety_violations"],
                **econ,
            })

        df_ranked = pd.DataFrame(candidates)
        if not df_ranked.empty:
            df_ranked = df_ranked.sort_values(
                by="expected_net_return_inr", ascending=False
            ).reset_index(drop=True)
            df_ranked["rank"] = df_ranked.index + 1
            return df_ranked.head(top_k), G_safe

        return df_ranked, G_safe