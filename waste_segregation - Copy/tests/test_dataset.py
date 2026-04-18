"""Tests for WasteDataset class."""
import pytest
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import WasteDataset
from src.transforms import get_transforms


CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = 224


class TestWasteDatasetStructure:
    """Test dataset initialization and structure."""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary dataset directory with sample images."""
        temp_dir = tempfile.mkdtemp()
        
        # Create train split with all classes
        for split in ['train', 'val', 'test']:
            for cls in CLASSES:
                cls_dir = Path(temp_dir) / split / cls
                cls_dir.mkdir(parents=True, exist_ok=True)
                
                # Create sample images (3 per class for train, 1 for val/test)
                num_images = 3 if split == 'train' else 1
                for i in range(num_images):
                    img = Image.fromarray(
                        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
                        'RGB'
                    )
                    img.save(cls_dir / f"sample_{i}.jpg")
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_dataset_initialization(self, temp_dataset_dir):
        """Test that dataset initializes correctly."""
        dataset = WasteDataset(temp_dataset_dir, split='train')
        
        assert dataset is not None
        assert len(dataset.classes) == len(CLASSES)
        assert len(dataset) > 0
    
    def test_dataset_classes(self, temp_dataset_dir):
        """Test that dataset correctly identifies classes."""
        dataset = WasteDataset(temp_dataset_dir, split='train')
        
        assert set(dataset.classes) == set(CLASSES)
        assert all(cls in dataset.class_to_idx for cls in CLASSES)
    
    def test_dataset_sample_count(self, temp_dataset_dir):
        """Test that dataset has correct number of samples."""
        dataset = WasteDataset(temp_dataset_dir, split='train')
        
        # 3 images per class, 6 classes
        expected_samples = 3 * len(CLASSES)
        assert len(dataset) == expected_samples
    
    def test_dataset_splits(self, temp_dataset_dir):
        """Test different dataset splits."""
        train_ds = WasteDataset(temp_dataset_dir, split='train')
        val_ds = WasteDataset(temp_dataset_dir, split='val')
        test_ds = WasteDataset(temp_dataset_dir, split='test')
        
        # Train has 3 images per class, val/test have 1
        assert len(train_ds) == 3 * len(CLASSES)
        assert len(val_ds) == 1 * len(CLASSES)
        assert len(test_ds) == 1 * len(CLASSES)


class TestWasteDatasetOutput:
    """Test dataset output format."""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary dataset directory."""
        temp_dir = tempfile.mkdtemp()
        
        for cls in CLASSES:
            cls_dir = Path(temp_dir) / 'train' / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            
            img = Image.fromarray(
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
                'RGB'
            )
            img.save(cls_dir / "sample.jpg")
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_getitem_returns_tuple(self, temp_dataset_dir):
        """Test that __getitem__ returns (image, label) tuple."""
        dataset = WasteDataset(temp_dataset_dir, split='train')
        
        item = dataset[0]
        
        assert isinstance(item, tuple)
        assert len(item) == 2
    
    def test_getitem_image_is_tensor(self, temp_dataset_dir):
        """Test that image is a tensor."""
        dataset = WasteDataset(temp_dataset_dir, split='train')
        
        img, label = dataset[0]
        
        assert isinstance(img, torch.Tensor)
    
    def test_getitem_image_shape(self, temp_dataset_dir):
        """Test that image has correct shape [C, H, W]."""
        transform = get_transforms('torchvision', 'train', IMG_SIZE)
        dataset = WasteDataset(temp_dataset_dir, split='train', transform=transform)
        
        img, label = dataset[0]
        
        assert img.shape == (3, IMG_SIZE, IMG_SIZE)
    
    def test_getitem_label_is_valid(self, temp_dataset_dir):
        """Test that label is a valid class index."""
        dataset = WasteDataset(temp_dataset_dir, split='train')
        
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            assert 0 <= label < len(CLASSES)


class TestWasteDatasetWithTransforms:
    """Test dataset with different transforms."""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary dataset directory."""
        temp_dir = tempfile.mkdtemp()
        
        cls_dir = Path(temp_dir) / 'train' / 'cardboard'
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            'RGB'
        )
        img.save(cls_dir / "sample.jpg")
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_torchvision_transforms(self, temp_dataset_dir):
        """Test dataset with torchvision transforms."""
        transform = get_transforms('torchvision', 'train', IMG_SIZE)
        dataset = WasteDataset(temp_dataset_dir, split='train', transform=transform)
        
        img, label = dataset[0]
        
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, IMG_SIZE, IMG_SIZE)
    
    def test_val_transforms(self, temp_dataset_dir):
        """Test dataset with validation transforms."""
        transform = get_transforms('torchvision', 'val', IMG_SIZE)
        dataset = WasteDataset(temp_dataset_dir, split='train', transform=transform)
        
        img, label = dataset[0]
        
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, IMG_SIZE, IMG_SIZE)


class TestWasteDatasetWithDataLoader:
    """Test dataset works with PyTorch DataLoader."""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary dataset directory."""
        temp_dir = tempfile.mkdtemp()
        
        for cls in CLASSES:
            cls_dir = Path(temp_dir) / 'train' / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(4):
                img = Image.fromarray(
                    np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
                    'RGB'
                )
                img.save(cls_dir / f"sample_{i}.jpg")
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_dataloader_iteration(self, temp_dataset_dir):
        """Test that DataLoader can iterate over dataset."""
        from torch.utils.data import DataLoader
        
        transform = get_transforms('torchvision', 'train', IMG_SIZE)
        dataset = WasteDataset(temp_dataset_dir, split='train', transform=transform)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        batch = next(iter(loader))
        
        assert len(batch) == 2
        images, labels = batch
        assert images.shape == (4, 3, IMG_SIZE, IMG_SIZE)
        assert labels.shape == (4,)
    
    def test_dataloader_all_batches(self, temp_dataset_dir):
        """Test iterating through all batches."""
        from torch.utils.data import DataLoader
        
        transform = get_transforms('torchvision', 'train', IMG_SIZE)
        dataset = WasteDataset(temp_dataset_dir, split='train', transform=transform)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        total_samples = 0
        for images, labels in loader:
            total_samples += len(labels)
        
        assert total_samples == len(dataset)


class TestRealDataset:
    """Tests for the real dataset if it exists."""
    
    @pytest.fixture
    def data_dir(self):
        return Path('data/processed')
    
    @pytest.mark.skipif(not Path('data/processed/train').exists(), 
                        reason="Real dataset not found")
    def test_real_train_dataset(self, data_dir):
        """Test loading real training dataset."""
        dataset = WasteDataset(str(data_dir), split='train')
        
        assert len(dataset) > 0
        assert len(dataset.classes) == len(CLASSES)
    
    @pytest.mark.skipif(not Path('data/processed/val').exists(), 
                        reason="Real dataset not found")
    def test_real_val_dataset(self, data_dir):
        """Test loading real validation dataset."""
        dataset = WasteDataset(str(data_dir), split='val')
        
        assert len(dataset) > 0
    
    @pytest.mark.skipif(not Path('data/processed/test').exists(), 
                        reason="Real dataset not found")
    def test_real_test_dataset(self, data_dir):
        """Test loading real test dataset."""
        dataset = WasteDataset(str(data_dir), split='test')
        
        assert len(dataset) > 0
