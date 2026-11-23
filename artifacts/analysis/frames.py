import numpy as np
from PIL import Image
from bigym.loco import make_gym
g=make_gym("move_plate"); demos=g.get_demos(3)
rows=[]
for di in [0,1]:
    d=demos[di]; T=len(d['action'])
    ts=[0,150,229,300,420,560,T-1]
    row=[]
    for t in ts:
        im=d['obs']['rgb'][t]  # (3,3,84,84)
        row.append(np.concatenate([c.transpose(1,2,0) for c in im],axis=0))
    rows.append(np.concatenate(row,axis=1))
img=np.concatenate(rows,axis=0)
Image.fromarray(img).resize((img.shape[1]*2,img.shape[0]*2),Image.NEAREST).save('/tmp/claude-1000/-home-keyu/e383bcf2-afc3-4d98-a857-a69ee81b8bef/scratchpad/frames.png')
print(img.shape)
