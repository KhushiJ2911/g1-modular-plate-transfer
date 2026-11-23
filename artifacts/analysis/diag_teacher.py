import sys, json, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'src')
from method.teacher.teacher import Teacher
from advaitic_eval.environment import make_environment
from pyquaternion import Quaternion
env=make_environment(); inner=env.inner_env; T=Teacher()
for seed in [int(s) for s in sys.argv[1].split(',')]:
    ts=env.reset(seed=seed); T.reset(env); k=None; rows=[]
    step=0
    while not ts.last():
        a=T.act(ts); ts=env.step(a); step+=1
        if k is None: k=T.chosen_demo; m=T.meta[k]
        if step%25==0 or ts.last():
            p=np.asarray(inner.plates[0].get_pose()).ravel(); pos=p[:3]
            d=min(np.linalg.norm(pos-inner.data.bind(s).xpos) for s in inner.rack_target.sites)
            up=Quaternion(p[3:7]).rotate([0,0,1]); ang=np.degrees(np.arccos(np.clip(np.dot(up,[0,-1,0]),-1,1)))
            held=inner.robot.is_gripper_holding_object(inner.plates[0], next(iter(inner.robot.grippers)))
            rows.append((step,T.demo_step,round(pos[0],3),round(pos[1],3),round(pos[2],3),round(d,3),round(ang,1),
                bool(inner.plates[0].is_colliding(inner.rack_target)),bool(inner.plates[0].is_colliding(inner.table)),held))
    print(f"== seed {seed} demo {k} grasp_step {m['grasp_step']} release {m['release_step']} term {env.episode_termination()}")
    for r in rows: print(r)
