import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import rasterio
from crop_yield.config import settings

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts spatial, texture, temporal, and auxiliary weather features
    from preprocessed SAR geotiff images and weather records.
    """

    def __init__(self):
        logger.info("Feature Extractor initialized.")

    def extract_backscatter(self, image_path: Path) -> Dict[str, float]:
        """
        Extracts spatial statistical features (mean, std) for VV and VH bands,
        and computes the cross-polarization ratio (VV - VH in dB).
        """
        logger.info(f"Extracting backscatter statistics from: {image_path.name}")

        with rasterio.open(image_path) as src:
            vv_band = src.read(1, out_shape=(512, 512))
            # Handle optional VH band
            if src.count >= 2:
                vh_band = src.read(2, out_shape=(512, 512))
            else:
                logger.warning("VH band missing. Using VV fallback.")
                vh_band = vv_band.copy()

            # Mask out any fill or nodata values (e.g. 0.0 if not masked)
            # In dB scale, typical backscatters are negative. 0.0 is usually nodata
            vv_valid = vv_band[vv_band != 0.0]
            vh_valid = vh_band[vh_band != 0.0]

            if len(vv_valid) == 0:
                vv_valid = vv_band
            if len(vh_valid) == 0:
                vh_valid = vh_band

            vv_mean = float(np.mean(vv_valid))
            vv_std = float(np.std(vv_valid))
            vh_mean = float(np.mean(vh_valid))
            vh_std = float(np.std(vh_valid))

            # Ratio in dB scale is the difference
            vv_vh_ratio = vv_mean - vh_mean

        return {
            "vv_mean": vv_mean,
            "vv_std": vv_std,
            "vh_mean": vh_mean,
            "vh_std": vh_std,
            "vv_vh_ratio": vv_vh_ratio,
        }

    def extract_texture(self, image_path: Path) -> Dict[str, float]:
        """
        Calculates Gray Level Co-occurrence Matrix (GLCM) texture metrics
        (Contrast, Homogeneity, Energy, Entropy) using pure NumPy.
        """
        logger.info(f"Extracting GLCM texture metrics from: {image_path.name}")

        with rasterio.open(image_path) as src:
            # Extract texture from the VV band (standard for structural analysis)
            band = src.read(1, out_shape=(512, 512))
            small = band[:256, :256]

        # Use only small crop for texture analysis

        b_min, b_max = small.min(), small.max()

        if b_max - b_min < 1e-5:
            img_quant = np.zeros_like(small, dtype=np.int32)

        else:
            img_quant = np.digitize(small, np.linspace(b_min, b_max, 8)) - 1

            img_quant = np.clip(img_quant, 0, 7)

        # Build horizontal GLCM
        h, w = img_quant.shape
        glcm = np.zeros((8, 8), dtype=np.float32)
        for r in range(h):
            for c in range(w - 1):
                i = img_quant[r, c]
                j = img_quant[r, c + 1]
                glcm[i, j] += 1

        # Normalize the co-occurrence matrix to get probabilities
        glcm_sum = glcm.sum()
        if glcm_sum > 0:
            glcm /= glcm_sum

        # Calculate GLCM properties
        contrast = 0.0
        homogeneity = 0.0
        energy = 0.0
        entropy = 0.0

        for i in range(8):
            for j in range(8):
                p_ij = glcm[i, j]
                contrast += ((i - j) ** 2) * p_ij
                homogeneity += p_ij / (1.0 + (i - j) ** 2)
                energy += p_ij**2
                if p_ij > 0:
                    entropy -= p_ij * np.log2(p_ij)

        return {
            "texture_contrast": float(contrast),
            "texture_homogeneity": float(homogeneity),
            "texture_energy": float(energy),
            "texture_entropy": float(entropy),
        }

    def extract_temporal_metrics(
        self, time_series_paths: List[Path]
    ) -> Dict[str, float]:
        """
        Computes multi-temporal aggregate metrics (min, max, mean, and trend slope)
        across a series of processed SAR image paths.
        """
        logger.info(
            f"Extracting temporal statistics across {len(time_series_paths)} scenes."
        )

        if not time_series_paths:
            return {
                "temporal_vv_min": -15.0,
                "temporal_vv_max": -10.0,
                "temporal_vv_mean": -12.5,
                "temporal_vv_slope": 0.0,
            }

        vv_means = []
        for idx, path in enumerate(time_series_paths):
            with rasterio.open(path) as src:
                band = src.read(
                    1,
                    out_shape=(256, 256)
                )
                vv_means.append(float(np.mean(band)))

        vv_means_arr = np.array(vv_means)

        # Calculate temporal slope (rate of change over time indices)
        x = np.arange(len(vv_means_arr))
        if len(x) > 1:
            slope, _ = np.polyfit(x, vv_means_arr, 1)
        else:
            slope = 0.0

        return {
            "temporal_vv_min": float(np.min(vv_means_arr)),
            "temporal_vv_max": float(np.max(vv_means_arr)),
            "temporal_vv_mean": float(np.mean(vv_means_arr)),
            "temporal_vv_slope": float(slope),
        }

    def run_pipeline(
        self,
        preprocessed_image: Path,
        weather_data: Path,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Extracts spatial, texture, temporal, and meteorological features,
        and saves them into a unified tabular CSV file.
        """
        logger.info(
            f"Running feature extraction pipeline. Image: {preprocessed_image.name}, Weather: {weather_data.name}"
        )

        # 1. Backscatter Stats
        backscatter = self.extract_backscatter(preprocessed_image)

        # 2. Textures
        texture = self.extract_texture(preprocessed_image)

        # 3. Temporal (simulate multiple scenes using the current preprocessed image)
        temporal = self.extract_temporal_metrics([preprocessed_image])

        # 4. Combine all features
        features = {**backscatter, **texture, **temporal}

        # 5. Integrate weather features
        try:
            weather_df = pd.read_csv(weather_data)
            features["mean_temperature"] = float(weather_df["temperature"].mean())
            features["max_temperature"] = float(weather_df["temperature"].max())
            features["min_temperature"] = float(weather_df["temperature"].min())
            features["total_rainfall"] = float(weather_df["rainfall"].sum())
            features["mean_soil_moisture"] = float(weather_df["soil_moisture"].mean())
        except Exception as e:
            logger.warning(f"Error parsing weather data: {e}. Using defaults.")
            features["mean_temperature"] = 24.0
            features["max_temperature"] = 30.0
            features["min_temperature"] = 18.0
            features["total_rainfall"] = 150.0
            features["mean_soil_moisture"] = 0.3

        # Ground truth crop yield label (simulated targets based on weather & soil variables)
        # Yield is typically correlated with moderate temperatures, rainfall, and backscatter growth curves
        base_yield = 2.0
        temp_factor = 0.05 * (30.0 - features["mean_temperature"])
        rain_factor = 0.005 * features["total_rainfall"]
        soil_factor = 1.2 * features["mean_soil_moisture"]
        sar_factor = 0.1 * (
            features["vv_mean"] + 15.0
        )  # less negative dB = higher biomass

        simulated_yield = (
            base_yield + temp_factor + rain_factor + soil_factor + sar_factor
        )
        features["target_yield"] = max(0.5, round(float(simulated_yield), 2))

        # Save to tabular output CSV file
        out_path = output_dir or settings.FEATURES_DIR
        out_path.mkdir(parents=True, exist_ok=True)

        features_file = out_path / "engineered_features.csv"

        # Write/append to dataset
        if features_file.exists():
            try:
                existing_df = pd.read_csv(features_file)
                # Keep appending row
                df_new = pd.DataFrame([features])
                df_combined = pd.concat([existing_df, df_new], ignore_index=True)
                df_combined.to_csv(features_file, index=False)
            except Exception:
                df = pd.DataFrame([features])
                df.to_csv(features_file, index=False)
        else:
            df = pd.DataFrame([features])
            df.to_csv(features_file, index=False)

        logger.info(f"Successfully compiled and saved features to: {features_file}")
        return features_file


if __name__ == "__main__":
    from crop_yield.data.ingestion import Sentinel1Downloader, WeatherDataIngester

    from crop_yield.data.preprocessing import SARPreprocessor

    downloader = Sentinel1Downloader()

    weather_ingester = WeatherDataIngester()

    test_aoi = {
        "type": "Polygon",
        "coordinates": [
            [[77.0, 28.5], [77.5, 28.5], [77.5, 28.8], [77.0, 28.8], [77.0, 28.5]]
        ],
    }

    # -----------------------------------------
    # 1. Sentinel ingestion
    # -----------------------------------------

    ingestion_result = downloader.download_by_aoi(
        geojson_aoi=test_aoi, start_date="2024-01-01", end_date="2024-01-31"
    )

    vv_path = Path(ingestion_result["vv_path"]) if isinstance(ingestion_result, dict) else Path(ingestion_result)

    # -----------------------------------------
    # 2. SAR preprocessing
    # -----------------------------------------

    preprocessor = SARPreprocessor()

    processed_file = preprocessor.run_pipeline(vv_path)

    # -----------------------------------------
    # 3. Weather ingestion
    # -----------------------------------------

    weather_csv = weather_ingester.fetch_weather_for_aoi(
        geojson_aoi=test_aoi, start_date="2024-01-01", end_date="2024-01-31"
    )

    # -----------------------------------------
    # 4. Feature extraction
    # -----------------------------------------

    extractor = FeatureExtractor()

    features_file = extractor.run_pipeline(
        preprocessed_image=processed_file, weather_data=weather_csv
    )

    print("\nFEATURE ENGINEERING COMPLETE")

    print(f"Features File: {features_file}")
