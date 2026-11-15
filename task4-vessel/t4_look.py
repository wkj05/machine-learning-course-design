import numpy as np
from PIL import Image
root='_inbox/task4/segmentation'
ids=['1','10','12','17']
W,H=565,584
canvas=Image.new('RGB',(W*len(ids), H*2),(0,0,0))
for k,i in enumerate(ids):
    im=np.asarray(Image.open(f'{root}/train/image/{i}.jpg').convert('RGB'))
    lb=np.asarray(Image.open(f'{root}/train/label/{i}.jpg').convert('L'))>127
    g=im[:,:,1]
    canvas.paste(Image.fromarray(im),(k*W,0))
    # label 上色：血管标红，叠在绿通道上
    vis=np.stack([np.where(lb,255,g//2), np.where(lb,0,g), np.where(lb,0,g//2)],-1).astype(np.uint8)
    canvas.paste(Image.fromarray(vis),(k*W,H))
canvas=canvas.resize((canvas.width//2, canvas.height//2), Image.LANCZOS)
canvas.save('_inbox/t4_look/pairs.png') if __import__('os').path.isdir('_inbox/t4_look') else None
import os
os.makedirs('_inbox/t4_look',exist_ok=True)
canvas.save('_inbox/t4_look/pairs.png')
print('saved', canvas.size)
