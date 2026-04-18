from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class WasteDataset(Dataset):
    """Simple dataset expecting a folder structure: root/<split>/<class>/*.jpg

    Example: data/processed/train/plastic/xxx.jpg
    """

    def __init__(self, root, split='train', transform=None):
        self.root = Path(root) / split
        self.transform = transform
        self.samples = []  # list of (path, label_idx)
        self.classes = []
        self.class_to_idx = {}
        self._scan()
        if self.transform is None:
            # lazily import default torchvision transform to avoid heavy import at module load
            import torchvision.transforms as T

            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def _scan(self):
        classes = [p.name for p in self.root.iterdir() if p.is_dir()]
        classes.sort()
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            for img in (self.root / c).iterdir():
                if img.is_file():
                    self.samples.append((str(img), self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    @property
    def labels(self):
        """Return list of all labels for computing class weights."""
        return [label for _, label in self.samples]

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')

        # Support albumentations (expects numpy array) and torchvision transforms
        is_alb = False
        try:
            import albumentations as A

            is_alb = isinstance(self.transform, A.core.composition.Compose)
        except Exception:
            is_alb = False

        if is_alb:
            img_np = np.array(img)
            img = self.transform(image=img_np)['image']
        else:
            img = self.transform(img)

        return img, label
