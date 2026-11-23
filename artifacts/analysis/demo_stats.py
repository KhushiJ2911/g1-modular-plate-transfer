import numpy as np, json
from bigym.loco import make_gym
env = make_gym("move_plate"); demos = env.get_demos(60, decode_images=False)
L=np.array([len(d['action']) for d in demos]); print('len min/med/max', L.min(), np.median(L), L.max())
print('success flags', sum(bool(d['success']) for d in demos))
A=np.concatenate([d['action'] for d in demos]); print('action min', A.min(0).round(2)); print('action max', A.max(0).round(2)); print('action std', A.std(0).round(3))
# gripper timing: dims 18,19
for name,i in [('g18',18),('g19',19)]:
    on=[(np.argmax(d['action'][:,i]>0.0) if (d['action'][:,i]>0).any() else -1) for d in demos]
    off=[(len(d['action'])-np.argmax(d['action'][::-1,i]>0.0)) if (d['action'][:,i]>0).any() else -1 for d in demos]
    print(name,'close step med',np.median(on),'open step med',np.median(off), 'n used', sum(o>=0 for o in on))
S=[d['obs']['state'] for d in demos]
disp=np.array([s[-1,18:22]-s[0,18:22] for s in S]); print('pelvis xy z yaw displacement mean',disp.mean(0).round(3),'std',disp.std(0).round(3))
print('start pelvis', np.array([s[0,18:22] for s in S]).mean(0).round(3))
print('meta keys', list(demos[0].keys()))
