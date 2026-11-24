import os
import re
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from config import GRID_SIZE, CELL_SIZE_KM, CACHE_DIR, SAFETY_LIMITS, SPECIES_CONFIG


class RealDataIngestor:
    """
    Loads, aligns, and preprocesses real INCOIS PFZ CSV advisories and
    NOAA/INCOIS/IMD NetCDF (.nc) marine datasets into the NavMeen 20x20 spatial grid.
    """

    def __init__(
        self,
        origin_lat=8.00,
        origin_lon=77.50,
        grid_size=GRID_SIZE,
        cell_size_km=CELL_SIZE_KM,
        cache_dir=CACHE_DIR,
    ):
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.grid_size = grid_size
        self.cell_size_km = cell_size_km
        self.cache_dir = cache_dir
        self.deg_step = cell_size_km / 111.32  # Approx degrees per km near equator

    def build_target_grid_skeleton(self):
        """Creates the base 20x20 coastal coordinate grid."""
        rows = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rows.append({
                    "row": r,
                    "col": c,
                    "lat": round(self.origin_lat + r * self.deg_step, 4),
                    "lon": round(self.origin_lon + c * self.deg_step, 4),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def parse_dms_to_decimal(coord_val):
        """
        Converts INCOIS DMS strings (e.g., '08° 12\\' 30\" N' or '8 12.5 N')
        or numeric strings into Decimal Degrees.
        """
        if pd.isna(coord_val):
            return np.nan
        if isinstance(coord_val, (int, float)):
            return float(coord_val)

        val_str = str(coord_val).strip().upper()
        if not val_str:
            return np.nan

        try:
            return float(val_str)
        except ValueError:
            pass

        parts = re.findall(r"[\d.]+", val_str)
        if not parts:
            return np.nan

        deg = float(parts[0])
        minutes = float(parts[1]) if len(parts) > 1 else 0.0
        seconds = float(parts[2]) if len(parts) > 2 else 0.0
        decimal = deg + (minutes / 60.0) + (seconds / 3600.0)

        if "S" in val_str or "W" in val_str:
            decimal = -decimal
        return round(decimal, 4)

    @staticmethod
    def project_bearing_distance(start_lat, start_lon, bearing_deg, distance_km):
        """
        Computes target (lat, lon) when an INCOIS advisory provides
        Landing Center coordinates + Bearing (deg) + Distance (km).
        """
        R_earth_km = 6371.0
        lat1 = np.radians(start_lat)
        lon1 = np.radians(start_lon)
        brng = np.radians(bearing_deg)
        d_div_r = distance_km / R_earth_km

        lat2 = np.arcsin(
            np.sin(lat1) * np.cos(d_div_r)
            + np.cos(lat1) * np.sin(d_div_r) * np.cos(brng)
        )
        lon2 = lon1 + np.arctan2(
            np.sin(brng) * np.sin(d_div_r) * np.cos(lat1),
            np.cos(d_div_r) - np.sin(lat1) * np.sin(lat2),
        )
        return round(np.degrees(lat2), 4), round(np.degrees(lon2), 4)

    def ingest_ocean_netcdf(self, df_grid, nc_path):
        """
        Extracts SST, Chlorophyll-a, Salinity, SSH, and Currents from a NetCDF file
        and interpolates them onto the NavMeen target grid coordinates.
        """
        ds = xr.open_dataset(nc_path)

        if "time" in ds.dims:
            ds = ds.isel(time=-1)

        lat_name = "lat" if "lat" in ds.coords else "latitude"
        lon_name = "lon" if "lon" in ds.coords else "longitude"

        target_lats = xr.DataArray(df_grid["lat"].values, dims="points")
        target_lons = xr.DataArray(df_grid["lon"].values, dims="points")

        interp_ds = ds.interp(
            {lat_name: target_lats, lon_name: target_lons}, method="linear"
        )

        df_out = df_grid.copy()
        df_out["sst_c"] = interp_ds["sst"].values
        df_out["chl_a_mg_m3"] = interp_ds["chlor_a"].values
        df_out["salinity_psu"] = (
            interp_ds["salinity"].values if "salinity" in interp_ds else 34.2
        )
        df_out["ssh_m"] = (
            interp_ds["ssh"].values if "ssh" in interp_ds else 0.05
        )

        if "u_current" in interp_ds and "v_current" in interp_ds:
            speed_ms = np.hypot(interp_ds["u_current"].values, interp_ds["v_current"].values)
            df_out["current_knots"] = speed_ms * 1.94384
        else:
            df_out["current_knots"] = 0.6

        ds.close()

        # Fill any coastal boundary NaNs using Pandas (no bottleneck needed)
        df_out = df_out.ffill().bfill()
        df_out["sst_c"] = df_out["sst_c"].round(2)
        df_out["chl_a_mg_m3"] = df_out["chl_a_mg_m3"].round(2)
        df_out["salinity_psu"] = df_out["salinity_psu"].round(2)
        df_out["ssh_m"] = df_out["ssh_m"].round(3)
        df_out["current_knots"] = df_out["current_knots"].round(2)

        return df_out

    def ingest_weather_netcdf(self, df_grid, nc_path):
        """
        Extracts IMD / NOAA Wave Height (swh) and Wind Speed (u10, v10 or wind_knots)
        from a marine weather NetCDF file and maps official safety warnings.
        """
        ds = xr.open_dataset(nc_path)
        if "time" in ds.dims:
            ds = ds.isel(time=-1)

        lat_name = "lat" if "lat" in ds.coords else "latitude"
        lon_name = "lon" if "lon" in ds.coords else "longitude"

        target_lats = xr.DataArray(df_grid["lat"].values, dims="points")
        target_lons = xr.DataArray(df_grid["lon"].values, dims="points")

        interp_ds = ds.interp(
            {lat_name: target_lats, lon_name: target_lons}, method="linear"
        )

        df_out = df_grid.copy()
        df_out["wave_height_m"] = interp_ds["swh"].values

        if "u10" in interp_ds and "v10" in interp_ds:
            wind_ms = np.hypot(interp_ds["u10"].values, interp_ds["v10"].values)
            df_out["wind_speed_knots"] = wind_ms * 1.94384
        else:
            df_out["wind_speed_knots"] = interp_ds["wind_speed_knots"].values

        # Fill boundary NaNs using Pandas before evaluating integer safety flags
        df_out = df_out.ffill().bfill()
        df_out["wave_height_m"] = df_out["wave_height_m"].round(2)
        df_out["wind_speed_knots"] = df_out["wind_speed_knots"].round(1)

        if "official_warning" in interp_ds:
            warn_vals = pd.Series(interp_ds["official_warning"].values).ffill().bfill()
            df_out["official_warning"] = warn_vals.round().astype(int)
        else:
            df_out["official_warning"] = (
                (df_out["wave_height_m"] > SAFETY_LIMITS["max_wave_height_m"])
                | (df_out["wind_speed_knots"] > SAFETY_LIMITS["max_wind_speed_knots"])
            ).astype(int)

        df_out["restricted_area"] = 0
        ds.close()
        return df_out

    def overlay_incois_pfz_csv(self, df_grid, pfz_csv_path, influence_radius_km=4.5):
        """
        Reads an INCOIS PFZ advisory CSV containing either direct (lat, lon)
        or (landing_lat, landing_lon, bearing_deg, distance_km) and tags nearby grid cells.
        """
        df_pfz = pd.read_csv(pfz_csv_path)
        pfz_coords = []

        for _, row in df_pfz.iterrows():
            plat, plon = np.nan, np.nan
            if "pfz_lat" in row and "pfz_lon" in row and pd.notna(row["pfz_lat"]) and str(row["pfz_lat"]).strip() != "":
                plat = self.parse_dms_to_decimal(row["pfz_lat"])
                plon = self.parse_dms_to_decimal(row["pfz_lon"])
            elif all(k in row for k in ["landing_lat", "landing_lon", "bearing_deg", "distance_km"]):
                llat = self.parse_dms_to_decimal(row["landing_lat"])
                llon = self.parse_dms_to_decimal(row["landing_lon"])
                plat, plon = self.project_bearing_distance(
                    llat, llon, float(row["bearing_deg"]), float(row["distance_km"])
                )

            if not np.isnan(plat) and not np.isnan(plon):
                pfz_coords.append((plat, plon))

        df_out = df_grid.copy()
        if not pfz_coords:
            df_out["incois_pfz_flag"] = 0
            return df_out

        tree = cKDTree(np.array(pfz_coords))
        grid_pts = df_out[["lat", "lon"]].values
        dists_deg, _ = tree.query(grid_pts, k=1)
        dists_km = dists_deg * 111.32

        df_out["incois_pfz_flag"] = (dists_km <= influence_radius_km).astype(int)
        df_out.loc[df_out["incois_pfz_flag"] == 1, "chl_a_mg_m3"] = (
            df_out.loc[df_out["incois_pfz_flag"] == 1, "chl_a_mg_m3"] + 0.65
        ).round(2)
        return df_out

    def build_and_cache_from_files(self, ocean_nc, weather_nc, pfz_csv):
        """
        Complete pipeline: builds skeleton grid, aligns Ocean NetCDF, Weather NetCDF,
        and INCOIS PFZ CSV, then saves to NavMeen's offline cache.
        """
        df_grid = self.build_target_grid_skeleton()
        df_grid = self.ingest_ocean_netcdf(df_grid, ocean_nc)
        df_grid = self.ingest_weather_netcdf(df_grid, weather_nc)
        df_grid = self.overlay_incois_pfz_csv(df_grid, pfz_csv)

        grid_file = os.path.join(self.cache_dir, "cached_ocean_weather_grid.csv")
        meta_file = os.path.join(self.cache_dir, "cache_metadata.json")
        df_grid.to_csv(grid_file, index=False)

        now_ts = datetime.now(timezone.utc).timestamp()
        metadata = {
            "source_pfz": f"INCOIS Advisory CSV ({os.path.basename(pfz_csv)}) + Ocean NetCDF",
            "source_weather": f"IMD Marine Forecast NetCDF ({os.path.basename(weather_nc)})",
            "issue_timestamp_utc": datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),
            "issue_epoch": now_ts,
            "validity_hours": SAFETY_LIMITS["cache_validity_hours"],
            "prices_inr_per_kg": {
                sp: cfg["market_price_inr_per_kg"] for sp, cfg in SPECIES_CONFIG.items()
            },
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return df_grid, metadata