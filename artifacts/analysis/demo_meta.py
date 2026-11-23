import json, numpy as np
from advaitic_eval.environment import make_environment, observed_source_slot
Z=np.load("demos_lowdim.npz"); env=make_environment(); inner=env.inner_env
NOM_S=np.array([0.55,0.24,0.70]); NOM_T=np.array([0.55,-0.24,0.70])
meta=[]
for i in range(60):
    seed=int(Z[f"seed{i}"]); env.reset(seed=seed)
    rs=inner.rack_start.get_position(); rt=inner.rack_target.get_position()
    m=dict(demo=i,seed=seed,slot=observed_source_slot(env),
      start_rack_offset=(rs-NOM_S).tolist(),target_rack_offset=(rt-NOM_T).tolist(),
      start_sites=[inner.data.bind(s).xpos.tolist() for s in inner.rack_start.sites],
      target_sites=[inner.data.bind(s).xpos.tolist() for s in inner.rack_target.sites],
      plate_pose_after_warmup=np.asarray(inner.plates[0].get_pose()).ravel().tolist())
    S=Z[f"s{i}"]; A=Z[f"a{i}"]; g=int(np.argmax(A[:,18]>0)); r=len(A)-int(np.argmax(A[::-1,18]>0))
    m.update(grasp_step=g,release_step=r,pelvis_at_grasp=S[g,18:22].tolist(),pelvis_at_release=S[r,18:22].tolist(),pelvis_start=S[0,18:22].tolist())
    meta.append(m)
json.dump(meta,open("demo_meta.json","w"),indent=1)
P=np.array([[*m['start_rack_offset'][:2],*m['target_rack_offset'][:2]] for m in meta])
G=np.array([m['pelvis_at_grasp'][:2] for m in meta]); R=np.array([m['pelvis_at_release'][:2] for m in meta])
src=np.array([m['start_sites'][m['slot']][:2] for m in meta]); plate=np.array([m['plate_pose_after_warmup'][:2] for m in meta])
print("sites spacing (start rack, y):", np.diff(np.array(meta[0]['start_sites'])[:,:2],axis=0).round(3).tolist())
print("plate xy - pelvis_at_grasp: mean",(plate-G).mean(0).round(3),"std",(plate-G).std(0).round(3))
print("corr(pelvis_grasp_x, plate_x)=%.2f corr(pelvis_grasp_y, plate_y)=%.2f"%(np.corrcoef(G[:,0],plate[:,0])[0,1],np.corrcoef(G[:,1],plate[:,1])[0,1]))
print("corr(pelvis_release_y, target_rack_y)=%.2f corr(x)=%.2f"%(np.corrcoef(R[:,1],P[:,3])[0,1],np.corrcoef(R[:,0],P[:,2])[0,1]))
print("plate xy spread std", plate.std(0).round(3), "pelvis grasp std", G.std(0).round(3))
