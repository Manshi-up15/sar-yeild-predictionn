import pytest
import numpy as np
import pandas as pd
from crop_yield.models.baseline import BaselineModelTrainer

@pytest.fixture
def mock_dataset():
    """Generates a mock tabular features dataset for training tests."""
    np.random.seed(42)
    n_samples = 30
    df = pd.DataFrame({
        "vv_mean": np.random.normal(-12.0, 1.5, n_samples),
        "vv_std": np.random.uniform(1.0, 2.5, n_samples),
        "vh_mean": np.random.normal(-18.0, 2.0, n_samples),
        "vh_std": np.random.uniform(0.8, 2.0, n_samples),
        "vv_vh_ratio": np.random.normal(6.0, 1.0, n_samples),
        "texture_contrast": np.random.uniform(0.1, 0.9, n_samples),
        "texture_homogeneity": np.random.uniform(0.3, 0.8, n_samples),
        "mean_temperature": np.random.normal(24.0, 3.0, n_samples),
        "total_rainfall": np.random.uniform(100.0, 500.0, n_samples),
        "mean_soil_moisture": np.random.uniform(0.15, 0.45, n_samples),
        "target_yield": np.random.uniform(1.5, 5.0, n_samples)
    })
    return df

def test_model_initialization():
    """Verify that only supported models can be initialized."""
    # Valid types
    assert BaselineModelTrainer(model_type="random_forest")
    assert BaselineModelTrainer(model_type="xgboost")
    assert BaselineModelTrainer(model_type="linear_regression")
    
    # Case insensitivity
    assert BaselineModelTrainer(model_type="XgBoOsT")
    
    # Invalid types
    with pytest.raises(ValueError, match="Unknown model type"):
        BaselineModelTrainer(model_type="unsupported_deep_net")


def test_small_dataset_fallback(mock_env):
    """Test that training succeeds even on very small datasets."""
    df_small = pd.DataFrame({
        "feature1": [1.0, 2.0],
        "feature2": [10.0, 20.0],
        "target_yield": [3.0, 5.0]
    })
    
    trainer = BaselineModelTrainer(model_type="linear_regression")
    summary = trainer.train(df_small)
    
    assert summary["train_size"] == 2
    assert summary["test_size"] == 2
    assert "rmse" in summary["metrics"]


def test_baseline_training_pipeline(mock_dataset, mock_env):
    """Verify train, predict, save, and load workflow for baseline regressors."""
    for model_type in ["random_forest", "xgboost", "linear_regression"]:
        trainer = BaselineModelTrainer(model_type=model_type)
        
        # 1. Train model
        summary = trainer.train(mock_dataset)
        assert summary["model_type"] == model_type
        assert summary["train_size"] > 0
        assert summary["test_size"] > 0
        
        metrics = summary["metrics"]
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        
        # 2. Predict
        preds = trainer.predict(mock_dataset)
        assert len(preds) == len(mock_dataset)
        
        # 3. Save
        model_file = trainer.save_model()
        assert model_file.exists()
        
        # 4. Load & Predict Compare
        new_trainer = BaselineModelTrainer(model_type=model_type)
        new_trainer.load_model(model_file)
        new_preds = new_trainer.predict(mock_dataset)
        
        # Predictions should match exactly
        np.testing.assert_array_almost_equal(preds, new_preds, decimal=5)
