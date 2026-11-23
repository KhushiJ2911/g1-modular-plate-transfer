# Blind replay of demo i's actions on a NEW evaluator-style seed with the SAME source slot.
import sys, json, numpy as np
from advaitic_eval.environment import make_environment, observed_source_slot
from advaitic_eval.protocol import source_slot_for_seed
Z=np.load("/home/keyu/advaitic/analysis/demos_lowdim.npz")
R=[json.loads(l) for l in open(sys.argv[3])]; slot_of={r['demo']:r['slot']-1 for r in R}
env=make_environment(); out=open(sys.argv[2],'a')
for i in [int(x) for x in sys.argv[1].split(',')]:
    seed=900000+i*1000
    while source_slot_for_seed(seed)!=slot_of[i]: seed+=1
    ts=env.reset(seed=seed); A=Z[f"a{i}"]
    for k in range(1700):
        ts=env.step(A[min(k,len(A)-1)])
        if ts.last(): break
    r=dict(demo=i,new_seed=seed,slot=slot_of[i]+1,success=bool(env.episode_succeeded() and not env.episode_fell()),term=env.episode_termination())
    print(json.dumps(r),file=out,flush=True)
