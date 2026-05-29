import logging
import numpy as np
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

features_path = Path("data/features/engineered_features.csv")

df = pd.read_csv(features_path)

# Temporary dataset expansion for prototype training

if len(df) < 10:
    logger.warning("Dataset too small. Expanding synthetic samples.")

    synthetic_rows = []

    for _ in range(50):
        noisy_row = df.iloc[0].copy()

        for col in df.columns:
            if col != "target_yield":
                noise = np.random.normal(0, 0.05)

                noisy_row[col] += noise

        noisy_row["target_yield"] += np.random.normal(0, 0.1)

        synthetic_rows.append(noisy_row)

    df = pd.DataFrame(synthetic_rows)

X = df.drop(columns=["target_yield"])

y = df["target_yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

rmse = mean_squared_error(y_test, predictions) ** 0.5

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)
print("\nMODEL PERFORMANCE")

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

model_dir = Path("models")

model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_dir / "random_forest.pkl")

print("\nMODEL SAVED")

print("Saved model to: models/random_forest.pkl")
