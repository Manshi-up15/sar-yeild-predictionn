import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from crop_yield.evaluation.metrics import (
    evaluate_predictions,
    plot_predicted_vs_actual,
    plot_residuals,
    plot_model_comparison
)

def test_metrics_evaluation():
    """Verify regression metrics outputs."""
    y_true = np.array([3.5, 4.0, 4.5, 5.0])
    y_pred = np.array([3.4, 4.1, 4.3, 5.2])
    
    metrics = evaluate_predictions(y_true, y_pred)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] > 0.0
    assert metrics["r2"] > 0.5


def test_visualization_generation(tmp_path):
    """Test scatter and residual plotting saves valid PNG files to disk."""
    y_true = np.array([3.5, 4.0, 4.5, 5.0, 5.5])
    y_pred = np.array([3.4, 4.1, 4.3, 5.2, 5.7])
    
    # 1. Test Predicted vs Actual Plot
    scatter_file = tmp_path / "scatter.png"
    result_scatter = plot_predicted_vs_actual(y_true, y_pred, scatter_file)
    assert result_scatter.exists()
    assert result_scatter.stat().st_size > 0
    
    # 2. Test Residual Plot
    residuals_file = tmp_path / "residuals.png"
    result_residuals = plot_residuals(y_true, y_pred, residuals_file)
    assert result_residuals.exists()
    assert result_residuals.stat().st_size > 0


def test_comparison_plot_generation(tmp_path):
    """Verify that multi-model metric comparisons compile chart graphics."""
    df_comparison = pd.DataFrame([
        {"model": "Linear Regression", "r2": 0.65, "rmse": 0.42},
        {"model": "Random Forest", "r2": 0.81, "rmse": 0.31},
        {"model": "XGBoost", "r2": 0.86, "rmse": 0.26},
        {"model": "CropCNN", "r2": 0.89, "rmse": 0.22}
    ])
    
    comparison_file = tmp_path / "comparison.png"
    result_comp = plot_model_comparison(df_comparison, comparison_file)
    
    assert result_comp.exists()
    assert result_comp.stat().st_size > 0
