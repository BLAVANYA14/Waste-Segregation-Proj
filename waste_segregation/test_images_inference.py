"""Test inference on images in Test_images folder."""
import torch
import timm
from PIL import Image
from pathlib import Path
from src.transforms import get_transforms

# Constants
MODEL_NAME = 'efficientnet_b2'
MODEL_PATH = 'models/best_1efficientnet_b2.pth'
IMG_SIZE = 224
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
    ckpt = torch.load(MODEL_PATH, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device

def predict(model, image_path, device):
    transform = get_transforms('torchvision', 'val', IMG_SIZE)
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    return CLASSES[pred_idx], confidence, probs[0].cpu().numpy()

def main():
    print("Loading model...")
    model, device = load_model()
    print(f"Model loaded on {device}\n")
    
    test_dir = Path('Test_images')
    images = list(test_dir.glob('*.[jp][pn][g]'))  # jpg, png
    
    print(f"Found {len(images)} images to test:\n")
    print("-" * 60)
    
    for img_path in images:
        pred_class, confidence, all_probs = predict(model, img_path, device)
        print(f"\nImage: {img_path.name}")
        print(f"  Predicted: {pred_class} ({confidence*100:.1f}% confidence)")
        print("  All probabilities:")
        for cls, prob in zip(CLASSES, all_probs):
            bar = "█" * int(prob * 20)
            print(f"    {cls:10s}: {prob*100:5.1f}% {bar}")

if __name__ == '__main__':
    main()
