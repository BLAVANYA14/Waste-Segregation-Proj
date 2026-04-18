"""Tests for model evaluation functions."""
import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import load_model, evaluate


CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
MODEL_NAME = 'efficientnet_b0'
MODEL_PATH = 'models/best_efficientnet_b0.pth'


class TestLoadModel:
    """Tests for load_model function."""
    
    @pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model checkpoint not found")
    def test_load_model_returns_model(self):
        """Test that load_model returns a model."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(MODEL_NAME, len(CLASSES), MODEL_PATH, device)
        
        assert model is not None
    
    @pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model checkpoint not found")
    def test_loaded_model_is_eval_mode(self):
        """Test that loaded model is in eval mode."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(MODEL_NAME, len(CLASSES), MODEL_PATH, device)
        
        assert not model.training
    
    @pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model checkpoint not found")
    def test_loaded_model_on_correct_device(self):
        """Test that model is on the correct device."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(MODEL_NAME, len(CLASSES), MODEL_PATH, device)
        
        # Check first parameter's device
        param = next(model.parameters())
        assert str(param.device).startswith(device)


class TestEvaluateFunction:
    """Tests for evaluate function."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        import timm
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
        model.eval()
        return model
    
    @pytest.fixture
    def mock_loader(self):
        """Create a mock data loader."""
        # Create fake data
        batch_size = 4
        num_batches = 3
        
        data = []
        for _ in range(num_batches):
            images = torch.randn(batch_size, 3, 224, 224)
            labels = torch.randint(0, len(CLASSES), (batch_size,))
            data.append((images, labels))
        
        return data
    
    def test_evaluate_returns_arrays(self, mock_model, mock_loader):
        """Test that evaluate returns numpy arrays."""
        device = 'cpu'
        mock_model.to(device)
        
        ys, ps = evaluate(mock_model, mock_loader, device)
        
        assert isinstance(ys, np.ndarray)
        assert isinstance(ps, np.ndarray)
    
    def test_evaluate_returns_correct_length(self, mock_model, mock_loader):
        """Test that evaluate returns correct number of predictions."""
        device = 'cpu'
        mock_model.to(device)
        
        ys, ps = evaluate(mock_model, mock_loader, device)
        
        expected_length = sum(batch[1].shape[0] for batch in mock_loader)
        assert len(ys) == expected_length
        assert len(ps) == expected_length
    
    def test_evaluate_predictions_in_valid_range(self, mock_model, mock_loader):
        """Test that predictions are valid class indices."""
        device = 'cpu'
        mock_model.to(device)
        
        ys, ps = evaluate(mock_model, mock_loader, device)
        
        assert (ps >= 0).all()
        assert (ps < len(CLASSES)).all()


class TestEvaluationMetrics:
    """Tests for evaluation metrics calculation."""
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        from sklearn.metrics import accuracy_score
        
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3, 4, 5])
        
        accuracy = accuracy_score(y_true, y_pred)
        assert accuracy == 1.0
    
    def test_zero_accuracy(self):
        """Test metrics with all wrong predictions."""
        from sklearn.metrics import accuracy_score
        
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5, 0])
        
        accuracy = accuracy_score(y_true, y_pred)
        assert accuracy == 0.0
    
    def test_confusion_matrix_shape(self):
        """Test confusion matrix has correct shape."""
        from sklearn.metrics import confusion_matrix
        
        y_true = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 1, 2, 3])
        
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (len(CLASSES), len(CLASSES))
    
    def test_precision_recall_f1(self):
        """Test precision, recall, and F1 calculation."""
        from sklearn.metrics import precision_recall_fscore_support
        
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
