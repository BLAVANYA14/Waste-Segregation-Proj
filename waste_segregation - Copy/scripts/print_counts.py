from pathlib import Path

root = Path('data/processed')
if not root.exists():
    print('No processed folder found')
    raise SystemExit(0)

for split in ['train', 'val', 'test']:
    sp = root / split
    print('Split:', split)
    if not sp.exists():
        print('  (missing)')
        continue
    classes = sorted([p for p in sp.iterdir() if p.is_dir()])
    if not classes:
        print('  (no class folders)')
        continue
    for cls in classes:
        n = sum(1 for _ in cls.rglob('*') if _.is_file())
        print(f'  {cls.name} : {n}')
