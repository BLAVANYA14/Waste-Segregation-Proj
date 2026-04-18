"""Tests for image transforms."""
import pytest
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transforms import (
    get_transforms,
    get_torchvision_transforms,
    mean, std
)


IMG_SIZE = 224


class TestTransformConstants:
    """Test transform constants."""
    
    def test_mean_values(self):
        """Test ImageNet mean values."""
        assert mean == (0.485, 0.456, 0.406)
    
    def test_std_values(self):
        """Test ImageNet std values."""
        assert std == (0.229, 0.224, 0.225)


class TestGetTransforms:
    """Test get_transforms function."""
    
    def test_get_torchvision_train_transforms(self):
        """Test getting torchvision train transforms."""
        transform = get_transforms('torchvision', 'train', IMG_SIZE)
        assert transform is not None
    
    def test_get_torchvision_val_transforms(self):
        """Test getting torchvision val transforms."""
        transform = get_transforms('torchvision', 'val', IMG_SIZE)
        assert transform is not None
    
    def test_invalid_backend_raises_error(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError):
            get_transforms('invalid_backend', 'train', IMG_SIZE)
    
    def test_backend_case_insensitive(self):
        """Test that backend parameter is case insensitive."""
        transform1 = get_transforms('TORCHVISION', 'train', IMG_SIZE)
        transform2 = get_transforms('TorchVision', 'train', IMG_SIZE)
        
        assert transform1 is not None
        assert transform2 is not None


class TestTorchvisionTransforms:
    """Test torchvision transform pipelines."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample PIL image."""
        return Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8),
            'RGB'
        )
    
    def test_train_transform_output_type(self, sample_image):
        """Test that train transform outputs a tensor."""
        transform = get_torchvision_transforms('train', IMG_SIZE)
        output = transform(sample_image)
        
        assert isinstance(output, torch.Tensor)
    
    def test_train_transform_output_shape(self, sample_image):
        """Test that train transform outputs correct shape."""
        transform = get_torchvision_transforms('train', IMG_SIZE)
        output = transform(sample_image)
        
        assert output.shape == (3, IMG_SIZE, IMG_SIZE)
    
    def test_val_transform_output_type(self, sample_image):
        """Test that val transform outputs a tensor."""
        transform = get_torchvision_transforms('val', IMG_SIZE)
        output = transform(sample_image)
        
        assert isinstance(output, torch.Tensor)
    
    def test_val_transform_output_shape(self, sample_image):
        """Test that val transform outputs correct shape."""
        transform = get_torchvision_transforms('val', IMG_SIZE)
        output = transform(sample_image)
        
        assert output.shape == (3, IMG_SIZE, IMG_SIZE)
    
    def test_transform_normalizes_values(self, sample_image):
        """Test that transforms normalize image values."""
        transform = get_torchvision_transforms('val', IMG_SIZE)
        output = transform(sample_image)
        
        # Normalized values should not be in [0, 255] range
        assert output.max() < 10  # After normalization
        assert output.min() > -10
    
    def test_different_image_sizes(self):
        """Test transforms work with different input image sizes."""
        sizes = [(100, 100), (300, 300), (500, 500), (200, 400)]
        transform = get_torchvision_transforms('val', IMG_SIZE)
        
        for size in sizes:
            img = Image.fromarray(
                np.random.randint(0, 255, (*size, 3), dtype=np.uint8),
                'RGB'
            )
            output = transform(img)
            
            assert output.shape == (3, IMG_SIZE, IMG_SIZE), f"Failed for size {size}"


class TestTransformAugmentation:
    """Test augmentation behavior."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a deterministic sample image."""
        np.random.seed(42)
        return Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8),
            'RGB'
        )
    
    def test_train_transforms_are_stochastic(self, sample_image):
        """Test that train transforms produce different outputs."""
        transform = get_torchvision_transforms('train', IMG_SIZE)
        
        outputs = [transform(sample_image) for _ in range(5)]
        
        # At least some outputs should be different
        all_same = all(torch.equal(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Train transforms should be stochastic"
    
    def test_val_transforms_are_deterministic(self, sample_image):
        """Test that val transforms produce same outputs."""
        transform = get_torchvision_transforms('val', IMG_SIZE)
        
        outputs = [transform(sample_image) for _ in range(5)]
        
        # All outputs should be the same
        all_same = all(torch.equal(outputs[0], o) for o in outputs[1:])
        assert all_same, "Val transforms should be deterministic"


class TestDifferentImageSizes:
    """Test transforms with different target sizes."""
    
    @pytest.fixture
    def sample_image(self):
        return Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8),
            'RGB'
        )
    
    def test_custom_image_size_128(self, sample_image):
        """Test transforms with 128x128 output size."""
        transform = get_transforms('torchvision', 'val', 128)
        output = transform(sample_image)
        
        assert output.shape == (3, 128, 128)
    
    def test_custom_image_size_256(self, sample_image):
        """Test transforms with 256x256 output size."""
        transform = get_transforms('torchvision', 'val', 256)
        output = transform(sample_image)
        
        assert output.shape == (3, 256, 256)
    
    def test_custom_image_size_384(self, sample_image):
        """Test transforms with 384x384 output size."""
        transform = get_transforms('torchvision', 'val', 384)
        output = transform(sample_image)
        
        assert output.shape == (3, 384, 384)


class TestAlbumentationsTransforms:
    """Test albumentations transform pipelines if available."""
    
    @pytest.fixture
    def sample_image(self):
        return Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8),
            'RGB'
        )
    
    @pytest.mark.skipif(True, reason="Albumentations may not be installed")
    def test_albumentations_train_transforms(self, sample_image):
        """Test albumentations train transforms."""
        try:
            from src.transforms import get_albumentations_transforms
            import albumentations as A
            
            transform = get_albumentations_transforms('train', IMG_SIZE)
            img_np = np.array(sample_image)
            output = transform(image=img_np)['image']
            
            assert isinstance(output, torch.Tensor)
            assert output.shape == (3, IMG_SIZE, IMG_SIZE)
        except ImportError:
            pytest.skip("Albumentations not installed")
