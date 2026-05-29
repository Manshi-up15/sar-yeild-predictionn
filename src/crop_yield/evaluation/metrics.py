import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from crop_yield.config import settings

logger = logging.getLogger(__name__)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates key regression metrics (RMSE, MAE, R2 Score) to evaluate performance.
    """
    logger.info("Evaluating predictions against ground truth.")
    
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Empty predictions or ground truth array provided.")
        return {"rmse": 0.0, "mae": 0.0, "r2": 0.0}
        
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }
    
    logger.info(f"Evaluation results: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    return metrics


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> Path:
    """
    Generates and saves a Predicted vs. Actual scatter plot with a 1:1 line.
    """
    logger.info(f"Generating Predicted vs. Actual plot: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, color='#3b82f6', alpha=0.7, edgecolors='none', label='Data Points')
    
    # 1:1 reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#ef4444', linestyle='--', linewidth=2, label='1:1 Perfect Fit')
    
    plt.xlabel('Ground Truth Yield (tons/ha)', fontsize=12)
    plt.ylabel('Predicted Yield (tons/ha)', fontsize=12)
    plt.title('Predicted vs. Actual Crop Yield', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> Path:
    """
    Generates and saves a Residuals vs. Predicted values plot to diagnose variance.
    """
    logger.info(f"Generating Residual plot: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    residuals = y_true - y_pred
    
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, color='#8b5cf6', alpha=0.7, edgecolors='none')
    plt.axhline(y=0.0, color='#10b981', linestyle='-', linewidth=2)
    
    plt.xlabel('Predicted Yield (tons/ha)', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.title('Residual Analysis Plot', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_model_comparison(comparison_df: pd.DataFrame, output_path: Path) -> Path:
    """
    Generates a bar chart comparing performance metrics across multiple models.
    Expects comparison_df to have index or column 'model' and metric columns.
    """
    logger.info(f"Generating Model Comparison plot: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if 'model' in comparison_df.columns:
        comparison_df = comparison_df.set_index('model')
        
    # We want to display R2 and RMSE (side-by-side or in subplots)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R2 Subplot
    if 'r2' in comparison_df.columns:
        comparison_df['r2'].plot(kind='bar', ax=axes[0], color='#10b981', alpha=0.85)
        axes[0].set_title('R² Coefficient of Determination (Higher is Better)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('R² Score')
        axes[0].grid(axis='y', linestyle=':', alpha=0.6)
        
    # RMSE Subplot
    if 'rmse' in comparison_df.columns:
        comparison_df['rmse'].plot(kind='bar', ax=axes[1], color='#f59e0b', alpha=0.85)
        axes[1].set_title('RMSE Error (Lower is Better)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('RMSE (tons/ha)')
        axes[1].grid(axis='y', linestyle=':', alpha=0.6)
        
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel('Model Architecture')
        
    plt.suptitle('Predictive Performance Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
