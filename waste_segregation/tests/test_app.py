"""Tests for Flask application endpoints."""
import pytest
import io
import json
from PIL import Image
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
MODEL_PATH = 'models/best_efficientnet_b0.pth'


@pytest.fixture
def app():
    """Create test Flask application."""
    from app import app as flask_app
    flask_app.config['TESTING'] = True
    return flask_app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


def create_test_image(size=(224, 224), color='RGB'):
    """Create a test image and return as bytes."""
    img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, color)
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes


class TestIndexRoute:
    """Test the index route."""
    
    def test_index_returns_200(self, client):
        """Test that index route returns 200."""
        response = client.get('/')
        assert response.status_code == 200
    
    def test_index_returns_html(self, client):
        """Test that index route returns HTML content."""
        response = client.get('/')
        assert b'html' in response.data.lower() or response.content_type == 'text/html; charset=utf-8'


@pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model checkpoint not found")
class TestPredictRoute:
    """Test the predict API endpoint."""
    
    @pytest.fixture(autouse=True)
    def setup_model(self, app):
        """Load model before tests."""
        from app import load_model
        load_model()
    
    def test_predict_returns_200(self, client):
        """Test that predict route returns 200 with valid image."""
        img_bytes = create_test_image()
        
        response = client.post(
            '/predict',
            data={'file': (img_bytes, 'test.jpg')},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 200
    
    def test_predict_returns_json(self, client):
        """Test that predict route returns JSON."""
        img_bytes = create_test_image()
        
        response = client.post(
            '/predict',
            data={'file': (img_bytes, 'test.jpg')},
            content_type='multipart/form-data'
        )
        
        assert response.content_type == 'application/json'
    
    def test_predict_response_structure(self, client):
        """Test that predict response has correct structure."""
        img_bytes = create_test_image()
        
        response = client.post(
            '/predict',
            data={'file': (img_bytes, 'test.jpg')},
            content_type='multipart/form-data'
        )
        
        data = json.loads(response.data)
        
        assert 'predicted_class' in data
        assert 'confidence' in data
        assert 'probabilities' in data
    
    def test_predict_class_is_valid(self, client):
        """Test that predicted class is one of the valid classes."""
        img_bytes = create_test_image()
        
        response = client.post(
            '/predict',
            data={'file': (img_bytes, 'test.jpg')},
            content_type='multipart/form-data'
        )
        
        data = json.loads(response.data)
        
        assert data['predicted_class'] in CLASSES
    
    def test_predict_confidence_range(self, client):
        """Test that confidence is between 0 and 1."""
        img_bytes = create_test_image()
        
        response = client.post(
            '/predict',
            data={'file': (img_bytes, 'test.jpg')},
            content_type='multipart/form-data'
        )
        
        data = json.loads(response.data)
        
        assert 0 <= data['confidence'] <= 1
    
    def test_predict_probabilities_sum_to_one(self, client):
        """Test that probabilities sum to approximately 1."""
        img_bytes = create_test_image()
        
        response = client.post(
            '/predict',
            data={'file': (img_bytes, 'test.jpg')},
            content_type='multipart/form-data'
        )
        
        data = json.loads(response.data)
        prob_sum = sum(data['probabilities'].values())
        
        assert abs(prob_sum - 1.0) < 0.01
    
    def test_predict_all_classes_in_probabilities(self, client):
        """Test that all classes are in probabilities."""
        img_bytes = create_test_image()
        
        response = client.post(
            '/predict',
            data={'file': (img_bytes, 'test.jpg')},
            content_type='multipart/form-data'
        )
        
        data = json.loads(response.data)
        
        for cls in CLASSES:
            assert cls in data['probabilities']


class TestPredictRouteErrors:
    """Test error handling in predict route."""
    
    def test_predict_no_file(self, client):
        """Test predict with no file returns error."""
        response = client.post('/predict')
        
        # Should return 400 or error message
        assert response.status_code in [400, 500] or b'error' in response.data.lower()
    
    def test_predict_empty_filename(self, client):
        """Test predict with empty filename."""
        response = client.post(
            '/predict',
            data={'file': (io.BytesIO(b''), '')},
            content_type='multipart/form-data'
        )
        
        assert response.status_code in [400, 500] or b'error' in response.data.lower()


class TestPredictDifferentImageFormats:
    """Test prediction with different image formats."""
    
    @pytest.fixture(autouse=True)
    def setup_model(self, app):
        """Load model before tests."""
        if Path(MODEL_PATH).exists():
            from app import load_model
            load_model()
    
    @pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model checkpoint not found")
    def test_predict_jpeg(self, client):
        """Test prediction with JPEG image."""
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        response = client.post(
            '/predict',
            data={'file': (img_bytes, 'test.jpg')},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 200
    
    @pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model checkpoint not found")
    def test_predict_png(self, client):
        """Test prediction with PNG image."""
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            '/predict',
            data={'file': (img_bytes, 'test.png')},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 200


class TestAppConfiguration:
    """Test application configuration."""
    
    def test_max_content_length(self, app):
        """Test max content length is set."""
        assert app.config['MAX_CONTENT_LENGTH'] == 16 * 1024 * 1024  # 16MB
