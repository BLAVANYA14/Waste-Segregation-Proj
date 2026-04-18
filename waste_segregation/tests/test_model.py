"""Tests for model loading and inference."""
import pytest
import torch
import timm
from PIL import Image
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transforms import get_transforms


# Constants matching app.py
MODEL_NAME = 'efficientnet_b0'
MODEL_PATH = 'models/best_efficientnet_b0.pth'
IMG_SIZE = 224
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


class TestModelArchitecture:
    """Test model architecture and creation."""
    
    def test_model_creation(self):
        """Test that model can be created with correct number of classes."""
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
        assert model is not None
        
    def test_model_output_shape(self):
        """Test that model outputs correct number of classes."""
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (1, len(CLASSES)), f"Expected shape (1, {len(CLASSES)}), got {output.shape}"
    
    def test_model_batch_processing(self):
        """Test that model can process batches."""
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
        model.eval()
        
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (batch_size, len(CLASSES))


class TestModelLoading:
    """Test model loading from checkpoint."""
    
    @pytest.fixture
    def model_path(self):
        return Path(MODEL_PATH)
    
    def test_checkpoint_exists(self, model_path):
        """Test that checkpoint file exists."""
        assert model_path.exists(), f"Model checkpoint not found at {model_path}"
    
    @pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model checkpoint not found")
    def test_load_checkpoint(self, model_path):
        """Test loading checkpoint file."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(model_path, map_location=device)
        
        # Should be either a state dict or a dict with 'model_state_dict'
        assert isinstance(ckpt, dict), "Checkpoint should be a dictionary"
    
    @pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model checkpoint not found")
    def test_load_model_weights(self, model_path):
        """Test that model weights can be loaded successfully."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
        
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt
        
        # This should not raise an error
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        
        # Verify model works after loading
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (1, len(CLASSES))


class TestModelInference:
    """Test model inference functionality."""
    
    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
        model.eval()
        return model
    
    @pytest.fixture
    def transform(self):
        """Get validation transform."""
        return get_transforms('torchvision', 'val', IMG_SIZE)
    
    def test_inference_with_random_image(self, model, transform):
        """Test inference with a random PIL image."""
        # Create a random image
        random_array = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        image = Image.fromarray(random_array, 'RGB')
        
        # Apply transform
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)[0]
        
        # Check output properties
        assert probs.shape == (len(CLASSES),)
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert (probs >= 0).all() and (probs <= 1).all()
    
    def test_softmax_probabilities_sum_to_one(self, model, transform):
        """Test that softmax probabilities sum to 1."""
        random_array = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        image = Image.fromarray(random_array, 'RGB')
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
        
        # Sum should be very close to 1
        assert torch.isclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5).all()
    
    def test_prediction_output_format(self, model, transform):
        """Test that prediction returns valid class index."""
        random_array = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        image = Image.fromarray(random_array, 'RGB')
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(img_tensor)
            pred_idx = logits.argmax(dim=1).item()
        
        assert 0 <= pred_idx < len(CLASSES)
        assert CLASSES[pred_idx] in CLASSES


class TestDeviceCompatibility:
    """Test model works on different devices."""
    
    def test_cpu_inference(self):
        """Test model inference on CPU."""
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
        model.to('cpu')
        model.eval()
        
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to('cpu')
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_inference(self):
        """Test model inference on CUDA if available."""
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
        model.to('cuda')
        model.eval()
        
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to('cuda')
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.device.type == 'cuda'
