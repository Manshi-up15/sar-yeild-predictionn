import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from crop_yield.config import settings

logger = logging.getLogger(__name__)

class BaselineModelTrainer:
    """
    Manages data splitting, model fitting, metric evaluation, saving,
    loading, and running inference for baseline models: Random Forest, XGBoost, and Linear Regression.
    """
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type.lower()
        self.model = None
        self._initialize_model()
        logger.info(f"Baseline Model Trainer initialized with type: {self.model_type}")

    def _initialize_model(self) -> None:
        """Initializes the underlying scikit-learn or xgboost regressor."""
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=settings.RANDOM_STATE,
                n_jobs=-1
            )
        elif self.model_type == "xgboost":
            self.model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=settings.RANDOM_STATE,
                n_jobs=-1
            )
        elif self.model_type == "linear_regression":
            self.model = LinearRegression()
        else:
            raise ValueError(
                f"Unknown model type: '{self.model_type}'. Choose from 'random_forest', 'xgboost', 'linear_regression'."
            )

    def train(self, features_df: pd.DataFrame, target_col: str = "target_yield") -> Dict[str, Any]:
        """
        Splits features_df into train/test, fits the selected model,
        evaluates on test data, and returns performance metrics.
        """
        logger.info(f"Preparing datasets for training baseline {self.model_type} model.")
        if target_col not in features_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in training data.")

        # Split features and target
        X = features_df.drop(columns=[target_col])
        y = features_df[target_col]

        # Handle very small datasets (common in unit tests/early prototyping)
        if len(features_df) < 5:
            logger.warning(
                f"Dataset size ({len(features_df)}) too small for train-test split. Training and testing on entire dataset."
            )
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE
            )

        logger.info(f"Dataset split completed. Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Predict and evaluate
        from crop_yield.evaluation.metrics import evaluate_predictions
        y_pred = self.model.predict(X_test)
        
        metrics = evaluate_predictions(y_test.to_numpy(), y_pred)
        
        logger.info(f"Model {self.model_type} training complete. Test metrics: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        return {
            "status": "trained",
            "model_type": self.model_type,
            "metrics": metrics,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "features_used": list(X.columns)
        }

    def predict(self, features_df: pd.DataFrame, target_col: str = "target_yield") -> Any:
        """
        Generates predictions for the given features dataset.
        """
        logger.info(f"Generating predictions with {self.model_type} model.")
        if target_col in features_df.columns:
            x = features_df.drop(columns=[target_col])
        else:
            x = features_df
            
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
            
        return self.model.predict(x)

    def save_model(self, output_dir: Optional[Path] = None) -> Path:
        """
        Saves the trained model to disk as a pickle file.
        """
        out_dir = output_dir or settings.MODELS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        model_file = out_dir / f"baseline_{self.model_type}.pkl"
        
        with open(model_file, "wb") as f:
            pickle.dump(self.model, f)
            
        logger.info(f"Successfully saved {self.model_type} model to: {model_file}")
        return model_file

    def load_model(self, model_path: Path) -> None:
        """
        Loads a pre-trained model pickle file from disk.
        """
        logger.info(f"Loading {self.model_type} model from: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
