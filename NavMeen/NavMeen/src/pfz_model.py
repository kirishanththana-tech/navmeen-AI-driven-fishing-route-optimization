import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)
from config import CACHE_DIR, SPECIES_CONFIG


class FishPotentialDNN(nn.Module):
    """
    Multi-Task PyTorch Deep Neural Network for NavMeen:
    - Head 1: Species Presence Suitability Probability [0, 1]
    - Head 2: Fish Gender Ratio Estimation [0, 1] (per System Requirements table)
    - Head 3: Normalized Expected Catch Factor [0, 1] (per System Requirements table)
    - Head 4: Estimated Fish Age Maturity Factor [0, 1] (per Dataset description)
    """

    def __init__(self, input_dim=6, scaler=None):
        super(FishPotentialDNN, self).__init__()
        self.scaler = scaler
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
        )
        self.suitability_head = nn.Sequential(nn.Linear(24, 1), nn.Sigmoid())
        self.gender_head = nn.Sequential(nn.Linear(24, 1), nn.Sigmoid())
        self.catch_head = nn.Sequential(nn.Linear(24, 1), nn.Sigmoid())
        self.age_head = nn.Sequential(nn.Linear(24, 1), nn.Sigmoid())

    def forward(self, x):
        features = self.shared_backbone(x)
        suitability = self.suitability_head(features)
        return suitability

    def forward_multitask(self, x):
        features = self.shared_backbone(x)
        return (
            self.suitability_head(features),
            self.gender_head(features),
            self.catch_head(features),
            self.age_head(features),
        )

    def predict_proba(self, X):
        """Scikit-learn compatible probability method for ROC and Calibration curves."""
        self.eval()
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        if self.scaler is not None:
            X_arr = self.scaler.transform(X_arr)

        device = next(self.parameters()).device
        with torch.no_grad():
            x_tensor = torch.tensor(X_arr, dtype=torch.float32).to(device)
            prob_pos = self.forward(x_tensor).cpu().numpy().flatten()

        prob_neg = 1.0 - prob_pos
        return np.column_stack([prob_neg, prob_pos])


class SpeciesPFZModel:
    """
    Trains, validates, and runs inference for NavMeen's PyTorch Deep Learning model.
    Saves epoch-by-epoch training/validation loss and accuracy curves for visualization.
    """

    FEATURE_COLS = [
        "sst_c",
        "chl_a_mg_m3",
        "salinity_psu",
        "ssh_m",
        "current_knots",
        "opencv_pfz_index",
    ]

    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.training_history = {}
        self.history_file = os.path.join(cache_dir, "dl_training_history.json")
        self.metrics_file = os.path.join(cache_dir, "dl_validation_metrics.json")

    def train_and_evaluate(self, df_historical, epochs=30, batch_size=32, lr=0.004):
        """
        Separates Train (time_block 0..3), Validation (time_block 4), and Test (time_block 5)
        to prevent data leakage (Methodology Section 2).
        """
        df_hist = df_historical.copy()
        if "opencv_pfz_index" not in df_hist.columns:
            df_hist["opencv_pfz_index"] = np.clip(df_hist["chl_a_mg_m3"] / 4.0, 0.0, 1.0)
        if "fish_gender" not in df_hist.columns:
            df_hist["fish_gender"] = 1
        if "fish_age_months" not in df_hist.columns:
            df_hist["fish_age_months"] = 18.0

        train_df = df_hist[df_hist["time_block"] < 4]
        val_df = df_hist[df_hist["time_block"] == 4]
        test_df = df_hist[df_hist["time_block"] >= 4]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[self.FEATURE_COLS].values)
        X_val = scaler.transform(val_df[self.FEATURE_COLS].values)
        X_test = scaler.transform(test_df[self.FEATURE_COLS].values)
        self.scalers["global"] = scaler

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        for sp_name, sp_cfg in SPECIES_CONFIG.items():
            sp_key = sp_name.replace(" ", "_")
            y_train_suit = train_df[f"label_{sp_key}"].values.astype(np.float32)
            y_train_gen = train_df["fish_gender"].values.astype(np.float32)
            y_train_age = np.clip(train_df["fish_age_months"].values / 36.0, 0.0, 1.0).astype(np.float32)

            if f"catch_kg_{sp_key}" in train_df.columns:
                y_train_catch = np.clip(
                    train_df[f"catch_kg_{sp_key}"].values / sp_cfg["base_catch_yield_kg"], 0.0, 1.0
                ).astype(np.float32)
            else:
                y_train_catch = y_train_suit.copy()

            y_val_suit = val_df[f"label_{sp_key}"].values.astype(np.float32)
            y_test_suit = test_df[f"label_{sp_key}"].values.astype(np.float32)

            dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train_suit).unsqueeze(1),
                torch.tensor(y_train_gen).unsqueeze(1),
                torch.tensor(y_train_catch).unsqueeze(1),
                torch.tensor(y_train_age).unsqueeze(1),
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            model = FishPotentialDNN(input_dim=len(self.FEATURE_COLS), scaler=scaler).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

            history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

            for _ in range(epochs):
                model.train()
                batch_losses = []
                correct = 0
                total = 0

                for bx, by_s, by_g, by_c, by_a in loader:
                    bx = bx.to(device)
                    by_s, by_g, by_c, by_a = (
                        by_s.to(device),
                        by_g.to(device),
                        by_c.to(device),
                        by_a.to(device),
                    )
                    optimizer.zero_grad()
                    p_s, p_g, p_c, p_a = model.forward_multitask(bx)

                    loss = (
                        bce_loss(p_s, by_s)
                        + 0.25 * bce_loss(p_g, by_g)
                        + 0.35 * mse_loss(p_c, by_c)
                        + 0.15 * mse_loss(p_a, by_a)
                    )
                    loss.backward()
                    optimizer.step()

                    batch_losses.append(float(loss.item()))
                    preds = (p_s >= 0.5).float()
                    correct += int((preds == by_s).sum().item())
                    total += int(by_s.size(0))

                # Validation evaluation per epoch
                model.eval()
                with torch.no_grad():
                    vx = torch.tensor(X_val, dtype=torch.float32).to(device)
                    vy = torch.tensor(y_val_suit, dtype=torch.float32).unsqueeze(1).to(device)
                    vp_s = model(vx)
                    v_loss = float(bce_loss(vp_s, vy).item())
                    v_acc = float(((vp_s >= 0.5).float() == vy).float().mean().item())

                history["train_loss"].append(round(float(np.mean(batch_losses)), 4))
                history["val_loss"].append(round(v_loss, 4))
                history["train_acc"].append(round(correct / max(1, total), 4))
                history["val_acc"].append(round(v_acc, 4))

            # Out-of-time Test Evaluation
            model.eval()
            with torch.no_grad():
                tx = torch.tensor(X_test, dtype=torch.float32).to(device)
                y_prob = model(tx).cpu().numpy().flatten()

            y_pred = (y_prob >= 0.5).astype(int)
            self.models[sp_name] = model
            self.training_history[sp_name] = history
            self.metrics[sp_name] = {
                "Precision": round(float(precision_score(y_test_suit, y_pred, zero_division=0)), 3),
                "Recall": round(float(recall_score(y_test_suit, y_pred, zero_division=0)), 3),
                "F1_Score": round(float(f1_score(y_test_suit, y_pred, zero_division=0)), 3),
                "ROC_AUC": round(float(roc_auc_score(y_test_suit, y_prob)), 3),
                "Brier_Calibration_Loss": round(float(brier_score_loss(y_test_suit, y_prob)), 3),
            }

            model_path = os.path.join(self.cache_dir, f"dl_pfz_model_{sp_key}.pt")
            torch.save({"model_state": model.state_dict()}, model_path)

        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.training_history, f, indent=2)
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

        return self.metrics

    def load_or_train(self, df_historical):
        df_hist = df_historical.copy()
        if "opencv_pfz_index" not in df_hist.columns:
            df_hist["opencv_pfz_index"] = np.clip(df_hist["chl_a_mg_m3"] / 4.0, 0.0, 1.0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler = StandardScaler()
        scaler.fit(df_hist[df_hist["time_block"] < 4][self.FEATURE_COLS].values)
        self.scalers["global"] = scaler

        if os.path.exists(self.history_file) and os.path.exists(self.metrics_file):
            with open(self.history_file, "r", encoding="utf-8") as f:
                self.training_history = json.load(f)
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

        for sp_name in SPECIES_CONFIG.keys():
            sp_key = sp_name.replace(" ", "_")
            model_path = os.path.join(self.cache_dir, f"dl_pfz_model_{sp_key}.pt")
            if os.path.exists(model_path):
                model = FishPotentialDNN(input_dim=len(self.FEATURE_COLS), scaler=scaler).to(device)
                try:
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint["model_state"])
                    model.eval()
                    self.models[sp_name] = model
                except Exception:
                    self.train_and_evaluate(df_hist)
                    break
            else:
                self.train_and_evaluate(df_hist)
                break

    def predict_grid_suitability(self, df_grid, target_species):
        if target_species not in self.models:
            raise ValueError(f"Unknown species: {target_species}")

        df_out = df_grid.copy()
        if "opencv_pfz_index" not in df_out.columns:
            df_out["opencv_pfz_index"] = np.clip(df_out["chl_a_mg_m3"] / 4.0, 0.0, 1.0).round(3)

        model = self.models[target_species]
        model.eval()
        scaler = self.scalers["global"]
        X_scaled = scaler.transform(df_out[self.FEATURE_COLS].values)

        device = next(model.parameters()).device
        with torch.no_grad():
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            p_suit, p_gen, p_catch, p_age = model.forward_multitask(x_tensor)

        suit_arr = p_suit.cpu().numpy().flatten()
        gen_arr = p_gen.cpu().numpy().flatten()
        catch_arr = p_catch.cpu().numpy().flatten()
        age_arr = p_age.cpu().numpy().flatten()

        base_yield = SPECIES_CONFIG[target_species]["base_catch_yield_kg"]
        df_out["fish_suitability_score"] = np.round(suit_arr, 3)
        df_out["dl_expected_catch_kg"] = np.round(catch_arr * base_yield, 1)
        df_out["estimated_female_ratio"] = np.round(gen_arr, 2)
        df_out["estimated_age_months"] = np.round(age_arr * 36.0, 1)
        return df_out