"""Flask web application for waste classification.

Run with:
    python app.py

Or with Flask CLI:
    flask run --host=0.0.0.0 --port=5000
"""
import os
from pathlib import Path
from io import BytesIO

import torch
import timm
from PIL import Image
from flask import Flask, request, jsonify, render_template

from src.transforms import get_transforms

# Configuration
MODEL_NAME = 'efficientnet_b2'
MODEL_PATH = 'models/best_1efficientnet_b2.pth'
IMG_SIZE = 224
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Global model variable
model = None
device = None
transform = None


def load_model():
    """Load the trained model into memory."""
    global model, device, transform
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model architecture
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
    
    # Load checkpoint
    ckpt = torch.load(MODEL_PATH, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    # Load validation transforms (no augmentation)
    transform = get_transforms('torchvision', 'val', IMG_SIZE)
    
    print(f"Model loaded successfully from {MODEL_PATH}")


def predict_image(image: Image.Image) -> dict:
    """Run inference on a PIL Image and return predictions."""
    global model, device, transform
    
    # Ensure RGB
    image = image.convert('RGB')
    
    # Apply transforms
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    
    # Get top prediction
    top_prob, top_idx = probabilities.max(0)
    predicted_class = CLASSES[top_idx.item()]
    confidence = top_prob.item()
    
    # Get all class probabilities
    all_probs = {cls: float(probabilities[i]) for i, cls in enumerate(CLASSES)}
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': all_probs
    }


@app.route('/')
def index():
    """Render the main upload page."""
    return render_template('index.html', classes=CLASSES)


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for image classification.
    
    Accepts:
        - File upload with key 'file'
        - Or JSON with base64 encoded image
    
    Returns:
        JSON with prediction results
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in allowed_extensions:
        return jsonify({'error': f'Invalid file type. Allowed: {allowed_extensions}'}), 400
    
    try:
        # Read and process image
        image = Image.open(BytesIO(file.read()))
        result = predict_image(image)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model': MODEL_NAME,
        'device': str(device),
        'classes': CLASSES
    })


@app.route('/api/classes')
def get_classes():
    """Return the list of classification classes."""
    return jsonify({'classes': CLASSES})


# Load model on startup
with app.app_context():
    load_model()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
