import numpy as np, glob, os
from PIL import Image
root='_inbox/task4/segmentation'
def stat(d,pat='*.jpg'):
    fs=sorted(glob.glob(os.path.join(root,d,pat)))
    print(f'--- {d}: {len(fs)} files')
    for f in fs[:3]:
        im=Image.open(f); a=np.asarray(im.convert('RGB'))
        print(f'   {os.path.basename(f):8s} size={im.size} mode={im.mode} mean={a.mean():.1f}')
    return fs
tr=stat('train/image'); lb=stat('train/label'); te=stat('test/image')
print()
print('sizes train:', sorted(set(Image.open(f).size for f in tr)))
print('sizes label:', sorted(set(Image.open(f).size for f in lb)))
print('sizes test :', sorted(set(Image.open(f).size for f in te)))
print()
for f in lb[:6]:
    a=np.asarray(Image.open(f).convert('L'))
    u,c=np.unique(a,return_counts=True)
    frac=(a>127).mean()
    print(f'{os.path.basename(f):8s} label uniq={len(u)} min={a.min()} max={a.max()} frac>127={frac:.4f} top={list(zip(u[:4],c[:4]))[:4]}')
print()
for f in tr[:4]:
    a=np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
    print(f'{os.path.basename(f):8s} per-channel mean={a.reshape(-1,3).mean(0).round(1)}')
