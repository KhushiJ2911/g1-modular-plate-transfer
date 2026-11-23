import numpy as np
from bigym.loco import make_gym
env = make_gym("move_plate"); demos = env.get_demos(60, decode_images=False)
d=demos[0]
def show(k,v,ind=''):
    if isinstance(v,dict):
        for kk,vv in v.items(): show(kk,vv,ind+k+'.')
    else:
        a=np.asarray(v) if not isinstance(v,(str,bytes)) else v
        print(ind+k, getattr(a,'shape',None), getattr(a,'dtype',type(v)))
for k,v in d.items(): show(k,v)
L=[len(np.asarray(x['actions'])) if 'actions' in x else None for x in demos]
print('lengths', L)
