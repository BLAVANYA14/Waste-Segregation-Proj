from pathlib import Path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def list_image_files(folder, exts=('.jpg', '.jpeg', '.png')):
    p = Path(folder)
    for f in p.rglob('*'):
        if f.is_file() and f.suffix.lower() in exts:
            yield str(f)
