import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from crop_yield.data.ingestion import Sentinel1Downloader
from crop_yield.models.deep_learning import SARDataset, CropCNN, CropCNNLSTM, DeepLearningModelTrainer

@pytest.fixture
def dummy_raster_files(mock_env):
    """Creates mock rasters for dataset testing."""
    downloader = Sentinel1Downloader()
    path1 = downloader.download_by_aoi({}, "2023-01-01", "2023-01-02")
    path2 = downloader.download_by_aoi({}, "2023-01-03", "2023-01-04")
    return [path1, path2]

def test_sar_dataset_static(dummy_raster_files):
    """Test SARDataset returning standard spatial image shapes for CNN."""
    targets = [3.5, 4.2]
    dataset = SARDataset(
        image_paths=dummy_raster_files,
        targets=targets,
        is_temporal=False,
        img_size=16
    )
    
    assert len(dataset) == 2
    img, target = dataset[0]
    
    # 2 channels (VV, VH), 16x16 size
    assert img.shape == (2, 16, 16)
    assert target.shape == (1,)
    assert target.item() == 3.5


def test_sar_dataset_temporal(dummy_raster_files):
    """Test SARDataset returning sequence shapes for CNN-LSTM."""
    targets = [3.5, 4.2]
    dataset = SARDataset(
        image_paths=dummy_raster_files,
        targets=targets,
        is_temporal=True,
        seq_len=4,
        img_size=16
    )
    
    assert len(dataset) == 2
    seq_img, target = dataset[0]
    
    # [sequence_length, channels, height, width]
    assert seq_img.shape == (4, 2, 16, 16)
    assert target.shape == (1,)


def test_cnn_training_loop(dummy_raster_files, mock_env):
    """Test that DeepLearningModelTrainer compiles and trains CNN using real DataLoaders."""
    targets = [3.5, 4.2]
    dataset = SARDataset(dummy_raster_files, targets, is_temporal=False, img_size=16)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    trainer = DeepLearningModelTrainer(model_name="cnn")
    history = trainer.train_model(train_loader=loader, val_loader=loader, epochs=3)
    
    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) == 3
    assert history["train_loss"][-1] >= 0.0


def test_cnn_lstm_training_loop(dummy_raster_files, mock_env):
    """Test that DeepLearningModelTrainer compiles and trains CNN-LSTM using real DataLoaders."""
    targets = [3.5, 4.2]
    dataset = SARDataset(dummy_raster_files, targets, is_temporal=True, seq_len=3, img_size=16)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    trainer = DeepLearningModelTrainer(model_name="cnn_lstm")
    history = trainer.train_model(train_loader=loader, val_loader=None, epochs=2)
    
    assert len(history["train_loss"]) == 2
    
    # Save checkpoint
    chk_file = trainer.save_checkpoint()
    assert chk_file.exists()
    
    # Load and predict
    new_trainer = DeepLearningModelTrainer(model_name="cnn_lstm")
    new_trainer.load_checkpoint(chk_file)
    preds = new_trainer.predict(loader)
    
    assert preds.shape == (2, 1)
