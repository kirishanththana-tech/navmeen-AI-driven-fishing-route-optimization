import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

from config import FIGURES_DIR, DEFAULT_VESSEL
from src.data_manager import DataManager
from src.pfz_model import SpeciesPFZModel
from src.vessel_economics import VesselEconomicModel
from src.route_optimizer import RouteOptimizer
from src.evaluator import BaselineEvaluator

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 300


def generate_all_report_figures():
    print("=" * 65)
    print("NavMeen Complete Report Figure Generator (Matplotlib & Seaborn)")
    print("=" * 65)

    dm = DataManager()
    df_grid, _ = dm.generate_and_cache_layers(seed=42, simulate_storm=True)
    df_hist = dm.generate_historical_training_data(n_samples=1800)

    pfz_model = SpeciesPFZModel()
    pfz_model.train_and_evaluate(df_hist, epochs=30)

    test_df = df_hist[df_hist["time_block"] >= 4]
    X_test = test_df[SpeciesPFZModel.FEATURE_COLS]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # 1. Deep Learning Training & Validation Loss / Accuracy Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for idx, (sp_name, hist) in enumerate(pfz_model.training_history.items()):
        epochs_range = range(1, len(hist["train_loss"]) + 1)
        ax1.plot(epochs_range, hist["train_loss"], color=colors[idx], lw=2, label=f"{sp_name} (Train)")
        ax1.plot(epochs_range, hist["val_loss"], color=colors[idx], lw=1.8, linestyle="--", label=f"{sp_name} (Val)")
        ax2.plot(epochs_range, hist["train_acc"], color=colors[idx], lw=2, label=f"{sp_name} (Train)")
        ax2.plot(epochs_range, hist["val_acc"], color=colors[idx], lw=1.8, linestyle="--", label=f"{sp_name} (Val)")

    ax1.set_title("Deep Learning Training & Validation Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Multi-Task Loss")
    ax1.legend(fontsize=8)

    ax2.set_title("Deep Learning Training & Validation Accuracy Curves")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Classification Accuracy")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    p1 = os.path.join(FIGURES_DIR, "fig1_training_loss_accuracy_curves.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"[+] Saved: {p1}")

    # 2. ROC Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, (sp_name, model) in enumerate(pfz_model.models.items()):
        y_test = test_df[f"label_{sp_name.replace(' ', '_')}"].values
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[idx], lw=2.2, label=f"{sp_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("NavMeen Deep Learning ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    p2 = os.path.join(FIGURES_DIR, "fig2_species_roc_curves.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"[+] Saved: {p2}")

    # 3. Calibration Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, (sp_name, model) in enumerate(pfz_model.models.items()):
        y_test = test_df[f"label_{sp_name.replace(' ', '_')}"].values
        y_prob = model.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=8)
        ax.plot(mean_pred, frac_pos, marker="o", color=colors[idx], lw=2, label=sp_name)

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Suitability Probability")
    ax.set_ylabel("Observed Presence Fraction")
    ax.set_title("NavMeen Probability Calibration Diagrams")
    ax.legend(loc="upper left")
    plt.tight_layout()
    p3 = os.path.join(FIGURES_DIR, "fig3_model_calibration_curves.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"[+] Saved: {p3}")

    # 4. Baseline Comparison Charts
    df_scored = pfz_model.predict_grid_suitability(df_grid, "Indian Mackerel")
    econ_model = VesselEconomicModel(DEFAULT_VESSEL)
    optimizer = RouteOptimizer(econ_model)
    top_routes_df, _ = optimizer.recommend_routes(df_scored, "Indian Mackerel", top_k=5)
    evaluator = BaselineEvaluator(optimizer, econ_model)
    eval_df = evaluator.evaluate_baselines(df_scored, "Indian Mackerel", top_routes_df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    methods = eval_df["Method"].values
    returns = eval_df["Expected_Net_Return_INR"].values
    fuels = eval_df["Fuel_Used_L"].values
    bar_colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]

    b1 = ax1.bar(methods, returns, color=bar_colors, width=0.5, edgecolor="black")
    ax1.set_title("Simulated Expected Net Return (INR)")
    ax1.set_ylabel("INR")
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h * 0.88, f"INR {h:,.0f}", ha="center", color="white", fontweight="bold")

    b2 = ax2.bar(methods, fuels, color=bar_colors, width=0.5, edgecolor="black")
    ax2.set_title("Estimated Round-Trip Fuel Consumption (L)")
    ax2.set_ylabel("Liters")
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h * 0.85, f"{h:.1f} L", ha="center", color="white", fontweight="bold")

    plt.tight_layout()
    p4 = os.path.join(FIGURES_DIR, "fig4_baseline_comparison.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"[+] Saved: {p4}")
    print("=" * 65)


if __name__ == "__main__":
    generate_all_report_figures()