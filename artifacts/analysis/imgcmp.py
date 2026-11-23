import numpy as np
from advaitic_eval.environment import make_environment
z=np.load('/home/keyu/advaitic/data/human60/ep_0.npz'); seed=int(z['seed'])
env=make_environment(); ts=env.reset(seed=seed)
a=np.asarray(ts.rgb_obs).astype(float); b=z['rgb'][0].astype(float)
for c,n in enumerate(['head','rwrist','lwrist']):
    print(n,'mean abs diff %.1f'%np.abs(a[c]-b[c]).mean(),'means eval %.1f demo %.1f'%(a[c].mean(),b[c].mean()))
print('state maxdiff', np.abs(np.asarray(ts.low_dim_obs)-z['state'][0]).max())
from PIL import Image
Image.fromarray(np.concatenate([np.concatenate([x.transpose(1,2,0) for x in im],1) for im in [a.astype(np.uint8),b.astype(np.uint8)]],0)).resize((504,336)).save('/tmp/claude-1000/-home-keyu/e383bcf2-afc3-4d98-a857-a69ee81b8bef/scratchpad/imgcmp.png')
