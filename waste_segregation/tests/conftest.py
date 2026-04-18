"""Pytest configuration and shared fixtures."""
import pytest
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "requires_model: marks tests that require trained model"
    )


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return data directory."""
    return project_root / "data"


@pytest.fixture(scope="session")
def models_dir(project_root):
    """Return models directory."""
    return project_root / "models"


@pytest.fixture(scope="session")
def device():
    """Return available device (cuda or cpu)."""
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def sample_classes():
    """Return list of waste classes."""
    return ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
