"""Fit three perception heads on frozen public embeddings; split by whole episode."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from submission.head_features import head_inputs

PICKUP = ("APPROACH", "PREGRASP", "GRASP")
CONTACT = ("CARRY","ALIGN","LOWER","RELEASE","SETTLE","RETREAT")


def world(rel, state):
    x,y,z = rel.T
    c,s = np.cos(state[:,21]),np.sin(state[:,21])
    return np.stack([c*x-s*y,s*x+c*y,z],axis=1)+state[:,18:21]


def batch(files, teacher=False, estimator=None):
    rows=[]
    for path in files:
        with np.load(path) as f:
            state, truth = f["state"], f["truth"]
            if teacher:
                # Recreate the frozen encoder output for historical teacher RGB.
                from submission.model import PROPRIO_DIMS
                rgb = torch.as_tensor(f["rgb"],device="cuda")
                prop = (state[:,list(PROPRIO_DIMS)]-estimator.stats["proprio_mean"])/estimator.stats["proprio_std"]
                feats=[]
                hook=estimator.model.head[0].register_forward_pre_hook(
                    lambda m,a: feats.append(a[0].detach().cpu().numpy()))
                with torch.inference_mode():
                    y=torch.cat([estimator.model(im,pr) for im,pr in zip(rgb.split(32),torch.as_tensor(prop,device="cuda").split(32),strict=True)]).cpu()
                hook.remove()
                physical=y[:,:7].numpy()*estimator.stats["target_std"]+estimator.stats["target_mean"]
                pred=np.c_[physical,y[:,7].sigmoid().numpy()]
                features=np.concatenate(feats)
                phase=np.full(len(y),"teacher")
            else:
                pred,features,phase=f["baseline"],f["features"],f["phase"]
            plate,site=world(pred[:,:3],state),world(pred[:,3:6],state)
            x=head_inputs(features,state,plate,site,pred[:,7],plate[:1])
            labels=np.c_[(world(truth[:,:3],state)-plate)/0.025,truth[:,7],truth[:,15]]
            rows.append((x,labels,phase))
    return tuple(np.concatenate(column) for column in zip(*rows,strict=True))


def train(root, out, base, steps=1500, smoke=False):
    from submission.estimator import Estimator
    torch.manual_seed(773)
    files = [root / "astra-dagger/sol" / f"{s}.npz" for s in range(3110000,3110001 if smoke else 3110300)]
    trainfiles=[p for p in files if int(p.stem)%10>=2] if not smoke else files
    validation=[p for p in files if int(p.stem)%10==1] if not smoke else files
    heldout=[p for p in files if int(p.stem)%10==0] if not smoke else files
    teacher=sorted((root/"est_v2/teacher").glob("*.npz"))
    assert len(teacher)==60
    if smoke:
        teacher=teacher[:1]
    teacherbatch=batch(teacher,True,Estimator(base))
    fresh=batch(trainfiles)
    x,y,phase=[np.concatenate(pair) for pair in zip(fresh,teacherbatch,strict=True)]
    avg=x.mean(0)
    sd=x.std(0).clip(0.001)
    def tensor(a):
        return torch.as_tensor(a,device="cuda",dtype=torch.float32)
    def norm(x):
        return tensor(((x-avg)/sd).clip(-12,12))
    X,Y=norm(x),tensor(y)
    def masks(ph):
        return [torch.as_tensor(m,device="cuda") for m in
                (np.isin(ph,(*PICKUP,"teacher")),np.isin(ph,CONTACT))]
    pickup,contact=masks(phase)
    vx,vy,vphase=batch(validation)
    VX,VY=norm(vx),tensor(vy)
    vp,vc=masks(vphase)
    model=nn.Sequential(nn.Linear(X.shape[1],96),nn.ReLU(),nn.Linear(96,5)).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=0.001)
    def objective(p,y,pick,ct):
        z=F.binary_cross_entropy_with_logits(p[:,3],y[:,3])
        if pick.any():
            z=z+F.smooth_l1_loss(p[pick,:3],y[pick,:3])
        if ct.any():
            z=z+F.binary_cross_entropy_with_logits(p[ct,4],y[ct,4])
        return z
    counts={p:(phase==p).sum() for p in set(phase)}
    w=tensor(np.array([1/counts[p] for p in phase]))
    best=float("inf")
    out.parent.mkdir(parents=True,exist_ok=True)
    with np.load(root/"astra-r3/contact_v1.npz") as f:
        original=dict(f)
    for step in range(1,steps+1):
        idx=torch.multinomial(w,256,replacement=True)
        opt.zero_grad()
        score=objective(model(X[idx]),Y[idx],pickup[idx],contact[idx])
        score.backward()
        opt.step()
        if step%100==0 or step==steps:
            with torch.no_grad():
                err=float(objective(model(VX),VY,vp,vc))
            if err<best:
                best=err
                def cpu(t):
                    return t.detach().cpu().numpy()
                np.savez_compressed(out,**original,x_mean=avg,x_std=sd,hidden_weight=cpu(model[0].weight.T),hidden_bias=cpu(model[0].bias),out_weight=cpu(model[2].weight.T),out_bias=cpu(model[2].bias))
    tx,ty,ph=batch(heldout)
    with np.load(out) as f:
        hidden=np.maximum(0,((tx-avg)/sd).clip(-12,12)@f["hidden_weight"]+f["hidden_bias"])
        yhat=hidden@f["out_weight"]+f["out_bias"]
    metrics={}
    for p in np.unique(ph):
        m=ph==p
        metrics[p]={"frames":int(m.sum()),"held_error":float(np.mean((yhat[m,3]>=0)!=ty[m,3])),
                    "xyz_residual_cm":float(np.linalg.norm(yhat[m,:3]-ty[m,:3],axis=1).mean()*2.5),
                    "contact_error":float(np.mean((yhat[m,4]>=0)!=ty[m,4]))}
    return dict(sha256=hashlib.sha256(out.read_bytes()).hexdigest(), heldout=metrics, train_seeds=[int(p.stem) for p in trainfiles],
                validation_seeds=[int(p.stem) for p in validation],test_seeds=[int(p.stem) for p in heldout],
                teacher_count=len(teacher), train_frames=len(X), validation_loss=best)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    print(json.dumps(train(Path("/data"),a.out,"/base/est_v1b/last.pt",3 if a.smoke else 1500,a.smoke)))
