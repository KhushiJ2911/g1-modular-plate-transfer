import sys, time, numpy as np, pickle
from bigym.loco import make_gym
t=time.time()
env = make_gym("move_plate")
demos = env.get_demos(60, decode_images=False)
print("load s", time.time()-t, type(demos), len(demos))
d0 = demos[0]
print(type(d0)); print([a for a in dir(d0) if not a.startswith('_')][:40])
