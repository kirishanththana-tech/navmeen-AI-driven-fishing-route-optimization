from config import DEFAULT_VESSEL
from src.data_manager import DataManager
from src.pfz_model import SpeciesPFZModel
from src.vessel_economics import VesselEconomicModel
from src.route_optimizer import RouteOptimizer
from src.evaluator import BaselineEvaluator


def run_navmeen_pipeline(target_species="Indian Mackerel"):
    print("=" * 78)
    print("NavMeen: AI-Driven Fishing Route Decision-Support System (Full Prototype)")
    print("=" * 78)

    dm = DataManager()
    dm.generate_and_cache_layers(seed=42, simulate_storm=True, hours_old=2.0)
    df_hist = dm.generate_historical_training_data()
    df_grid, meta = dm.load_offline_cache()

    print(f"\n[1] Offline Cache & OpenCV Satellite PFZ Layer Loaded:")
    print(f"    PFZ Source      : {meta['source_pfz']}")
    print(f"    Weather Source  : {meta['source_weather']}")
    print(f"    Coverage        : {meta['geographic_coverage']}")
    print(f"    Forecast Age    : {meta['age_hours']} hrs (Stale Warning: {meta['is_stale']})")

    pfz_model = SpeciesPFZModel()
    metrics = pfz_model.train_and_evaluate(df_hist)
    print(f"\n[2] Deep Learning Model Validation Metrics (Out-of-Time Test Split):")
    for sp, m in metrics.items():
        print(
            f"    - {sp:17s} | ROC-AUC: {m['ROC_AUC']:.3f} | "
            f"F1: {m['F1_Score']:.3f} | Precision: {m['Precision']:.3f} | "
            f"Recall: {m['Recall']:.3f} | Brier: {m['Brier_Calibration_Loss']:.3f}"
        )

    df_scored = pfz_model.predict_grid_suitability(df_grid, target_species)

    econ_model = VesselEconomicModel(DEFAULT_VESSEL)
    optimizer = RouteOptimizer(econ_model, algorithm="A*")
    top_routes_df, _ = optimizer.recommend_routes(
        df_scored, target_species=target_species, top_k=5
    )

    print(f"\n[3] Top 5 Personalized Routes for '{target_species}':")
    display_cols = [
        "rank", "target_row", "target_col", "fish_suitability_score",
        "expected_catch_kg", "round_trip_distance_km", "estimated_fuel_liters",
        "expected_net_return_inr", "weather_status"
    ]
    print(top_routes_df[display_cols].to_string(index=False))

    evaluator = BaselineEvaluator(optimizer, econ_model)
    eval_df = evaluator.evaluate_baselines(df_scored, target_species, top_routes_df)

    print(f"\n[4] Experimental Comparison Against Baselines (Section 9):")
    print(eval_df.to_string(index=False))

    boat_df = evaluator.evaluate_multi_boat_scenarios(df_scored, target_species)
    print(f"\n[5] Multi-Vessel Profile Sensitivity Evaluation:")
    print(boat_df.to_string(index=False))
    print("=" * 78)


if __name__ == "__main__":
    run_navmeen_pipeline(target_species="Indian Mackerel")