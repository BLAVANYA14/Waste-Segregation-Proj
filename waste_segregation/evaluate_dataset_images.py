"""Evaluate trained model on dataset images and show detailed results."""
import torch
import timm
from PIL import Image
from pathlib import Path
from collections import defaultdict
import numpy as np

from src.transforms import get_transforms
from src.dataset import WasteDataset

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

def predict_image(model, image_path, transform, device):
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    return pred_idx, confidence

def main():
    print("Loading model...")
    model, device = load_model()
    print(f"Model loaded on {device}\n")
    
    transform = get_transforms('torchvision', 'val', IMG_SIZE)
    
    # Evaluate on test set
    data_dir = Path('data/processed')
    split = 'test'  # Change to 'train' or 'val' if needed
    
    print(f"Evaluating {split} set...")
    print("=" * 80)
    
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)
    misclassified = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = data_dir / split / class_name
        if not class_dir.exists():
            continue
        
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        for img_path in images:
            pred_idx, confidence = predict_image(model, img_path, transform, device)
            total_per_class[class_name] += 1
            
            if pred_idx == class_idx:
                correct_per_class[class_name] += 1
            else:
                misclassified.append({
                    'path': str(img_path),
                    'true_class': class_name,
                    'pred_class': CLASSES[pred_idx],
                    'confidence': confidence
                })
    
    # Print per-class accuracy
    print("\nPER-CLASS ACCURACY:")
    print("-" * 50)
    for class_name in CLASSES:
        total = total_per_class[class_name]
        correct = correct_per_class[class_name]
        acc = correct / total * 100 if total > 0 else 0
        print(f"  {class_name:12s}: {correct:3d}/{total:3d} = {acc:5.1f}%")
    
    total_correct = sum(correct_per_class.values())
    total_samples = sum(total_per_class.values())
    print("-" * 50)
    print(f"  {'OVERALL':12s}: {total_correct:3d}/{total_samples:3d} = {total_correct/total_samples*100:.1f}%")
    
    # Print misclassified images
    print(f"\n\nMISCLASSIFIED IMAGES ({len(misclassified)} total):")
    print("=" * 80)
    
    # Group by true class
    by_true_class = defaultdict(list)
    for item in misclassified:
        by_true_class[item['true_class']].append(item)
    
    for true_class in CLASSES:
        items = by_true_class[true_class]
        if items:
            print(f"\n{true_class.upper()} misclassified as:")
            for item in items:
                filename = Path(item['path']).name
                print(f"  {filename:25s} -> {item['pred_class']:10s} ({item['confidence']*100:.1f}%)")
    
    # Summary of confusion patterns
    print("\n\nCONFUSION PATTERNS (True -> Predicted):")
    print("-" * 50)
    confusion_counts = defaultdict(int)
    for item in misclassified:
        key = f"{item['true_class']} -> {item['pred_class']}"
        confusion_counts[key] += 1
    
    for pattern, count in sorted(confusion_counts.items(), key=lambda x: -x[1]):
        print(f"  {pattern:25s}: {count} images")

if __name__ == '__main__':
    main()
