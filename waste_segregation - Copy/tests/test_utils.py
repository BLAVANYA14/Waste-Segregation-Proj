"""Tests for utility functions."""
import pytest
import tempfile
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import ensure_dir, list_image_files


class TestEnsureDir:
    """Test ensure_dir function."""
    
    def test_creates_single_directory(self):
        """Test creating a single directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / 'new_folder'
            
            assert not new_dir.exists()
            ensure_dir(new_dir)
            assert new_dir.exists()
            assert new_dir.is_dir()
    
    def test_creates_nested_directories(self):
        """Test creating nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / 'level1' / 'level2' / 'level3'
            
            assert not nested_dir.exists()
            ensure_dir(nested_dir)
            assert nested_dir.exists()
            assert nested_dir.is_dir()
    
    def test_existing_directory_no_error(self):
        """Test that existing directory doesn't raise error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise any error
            ensure_dir(temp_dir)
            assert Path(temp_dir).exists()
    
    def test_with_string_path(self):
        """Test with string path instead of Path object."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, 'new_folder')
            
            ensure_dir(new_dir)
            assert os.path.exists(new_dir)


class TestListImageFiles:
    """Test list_image_files function."""
    
    @pytest.fixture
    def temp_image_dir(self):
        """Create a temporary directory with image files."""
        temp_dir = tempfile.mkdtemp()
        
        # Create various files
        files = [
            'image1.jpg',
            'image2.jpeg',
            'image3.png',
            'image4.JPG',  # uppercase
            'document.txt',
            'data.json',
            'subfolder/nested_image.jpg',
        ]
        
        for f in files:
            file_path = Path(temp_dir) / f
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()
        
        yield temp_dir
        
        shutil.rmtree(temp_dir)
    
    def test_finds_jpg_files(self, temp_image_dir):
        """Test finding .jpg files."""
        files = list(list_image_files(temp_image_dir))
        
        jpg_files = [f for f in files if f.lower().endswith('.jpg')]
        assert len(jpg_files) >= 2  # image1.jpg, image4.JPG, nested
    
    def test_finds_jpeg_files(self, temp_image_dir):
        """Test finding .jpeg files."""
        files = list(list_image_files(temp_image_dir))
        
        jpeg_files = [f for f in files if f.lower().endswith('.jpeg')]
        assert len(jpeg_files) == 1  # image2.jpeg
    
    def test_finds_png_files(self, temp_image_dir):
        """Test finding .png files."""
        files = list(list_image_files(temp_image_dir))
        
        png_files = [f for f in files if f.lower().endswith('.png')]
        assert len(png_files) == 1  # image3.png
    
    def test_excludes_non_image_files(self, temp_image_dir):
        """Test that non-image files are excluded."""
        files = list(list_image_files(temp_image_dir))
        
        txt_files = [f for f in files if f.endswith('.txt')]
        json_files = [f for f in files if f.endswith('.json')]
        
        assert len(txt_files) == 0
        assert len(json_files) == 0
    
    def test_finds_nested_images(self, temp_image_dir):
        """Test that nested images are found."""
        files = list(list_image_files(temp_image_dir))
        
        nested = [f for f in files if 'nested' in f]
        assert len(nested) == 1
    
    def test_returns_generator(self, temp_image_dir):
        """Test that function returns a generator."""
        result = list_image_files(temp_image_dir)
        
        # Should be a generator
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')
    
    def test_empty_directory(self):
        """Test with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            files = list(list_image_files(temp_dir))
            assert len(files) == 0
    
    def test_custom_extensions(self, temp_image_dir):
        """Test with custom extensions."""
        # Add a file with different extension
        (Path(temp_image_dir) / 'test.bmp').touch()
        
        # Default should not find .bmp
        default_files = list(list_image_files(temp_image_dir))
        bmp_default = [f for f in default_files if f.endswith('.bmp')]
        assert len(bmp_default) == 0
        
        # Custom extensions should find .bmp
        custom_files = list(list_image_files(temp_image_dir, exts=('.bmp',)))
        bmp_custom = [f for f in custom_files if f.endswith('.bmp')]
        assert len(bmp_custom) == 1
    
    def test_returns_absolute_paths(self, temp_image_dir):
        """Test that returned paths are absolute."""
        files = list(list_image_files(temp_image_dir))
        
        for f in files:
            assert os.path.isabs(f), f"Path should be absolute: {f}"


class TestListImageFilesEdgeCases:
    """Edge case tests for list_image_files."""
    
    def test_nonexistent_directory(self):
        """Test with nonexistent directory."""
        with pytest.raises(Exception):
            list(list_image_files('/nonexistent/path/that/should/not/exist'))
    
    def test_file_instead_of_directory(self):
        """Test with file path instead of directory."""
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
            # Should handle gracefully or raise appropriate error
            try:
                files = list(list_image_files(temp_file.name))
                assert len(files) == 0
            except (NotADirectoryError, OSError):
                pass  # Expected behavior
