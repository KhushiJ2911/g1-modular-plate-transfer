"""Geometric registration of the visible plate lip using the head and left wrist."""

import numpy as np

from .orientation_feedback import LEFT_GRASP_NORMAL

# G1 head and wrist mounts in bigym/robots/configs/g1.py. Each camera is 84 px
# square with a 60 degree vertical field of view. Use the grip site as origin.
FOCAL = 42 / np.tan(np.pi / 6)
WRIST_POS = np.array([-0.028, 0.0, 0.040])
HEAD_POS = np.array([0.12, 0.0, 0.37])


def mount(pitch):
    c, s = np.cos(pitch), np.sin(pitch)
    z, w = np.cos(-1.57), np.sin(-1.57)
    return np.array([[c*z, -c*w, s], [w, z, 0.], [-s*z, s*w, c]])


WRIST_ROT = mount(-0.87)
HEAD_ROT = mount(-0.52)
# Outer lip height and radius, in metres, from the fixed plate_01 visual mesh.
RADIUS = 0.114
LIP_HEIGHT = 0.0174
THETA = np.linspace(0, 2*np.pi, 64, endpoint=False)


def rings(poses):
    center = poses[:, :3]
    normal = np.c_[poses[:,3], -np.ones(len(poses)), poses[:,4]]
    normal /= np.linalg.norm(normal,axis=1,keepdims=True)
    u = np.cross(normal,[0,0,1])
    u /= np.linalg.norm(u,axis=1,keepdims=True)
    v = np.cross(normal,u)
    return (center + LIP_HEIGHT*normal)[:,None,:] + RADIUS*(
        u[:,None,:]*np.cos(THETA)[None,:,None] + v[:,None,:]*np.sin(THETA)[None,:,None]
    )


def _image_gradient(image):
    color = np.asarray(image,dtype=float)
    # Preserve object edges even against an equally bright but coloured background.
    gy, gx = np.gradient(color, axis=(1,2))
    return gx, gy


def _outline(points, pos, rot, grad, wrist):
    p = (points - pos) @ rot
    z = -p[...,2]
    uv = 41.5 + FOCAL*p[...,:2]/np.maximum(z,0.001)[...,None]*[1,-1]
    u,v = uv[...,0], uv[...,1]
    valid = (z > .028) & (v>1) & (v < (25 if wrist else 81)) & (u>1) & (u<81)
    # Image gradient should be perpendicular to the projected lip. Occlusion
    # by a finger makes some contour segments unavailable in either view.
    tangent = np.roll(uv,-1,axis=1) - np.roll(uv,1,axis=1)
    tangent /= np.maximum(np.linalg.norm(tangent,axis=2,keepdims=True),1e-6)
    gx,gy=grad
    x=np.clip(u,0,82.99).astype(int)
    y=np.clip(v,0,82.99).astype(int)
    wx=np.clip(u-x,0,1)
    wy=np.clip(v-y,0,1)
    def interp(a):
        return ((1-wx)*((1-wy)*a[:,y,x]+wy*a[:,y+1,x])
                + wx*((1-wy)*a[:,y,x+1]+wy*a[:,y+1,x+1]))
    magnitude = np.mean(np.abs(interp(gx)*tangent[...,1] - interp(gy)*tangent[...,0]),axis=0)
    strength=np.minimum(magnitude,45)/45
    return (strength*valid).sum(axis=1)/np.maximum(valid.sum(axis=1),10)


def fit_views(rgb, head_rgb, torso_pos, torso_rot, estimated_center):
    """Return a measured hand-frame unit normal. All arguments are observable."""
    hpos=torso_pos+torso_rot@HEAD_POS
    hrot=torso_rot@HEAD_ROT
    wh,hh=_image_gradient(rgb),_image_gradient(head_rgb)
    prior=LEFT_GRASP_NORMAL
    x=np.r_[estimated_center,prior[[0,2]]/-prior[1]]
    p0=x.copy()
    # Coarse-to-fine search of a shared metric circle in the two camera views.
    # Penalize depth shifts that leave the observation-based starting vicinity.
    def score(choices):
        edges=rings(choices)
        h=_outline(edges,hpos,hrot,hh,False)
        w=_outline(edges,WRIST_POS,WRIST_ROT,wh,True)
        shift=(choices-p0)/[.045,.035,.045,.33,.33]
        return h + .55*w - .035*np.sum(shift*shift,axis=1)
    dirs=np.r_[np.zeros((1,5)),np.eye(5),-np.eye(5)]
    for step in [1.,.5,.25]:
        for _ in range(10):
            candidates=x+dirs*np.array([.012,.008,.012,.085,.085])*step
            which=np.argmax(score(candidates))
            if which==0: break
            x=candidates[which]
    n=np.array([x[3],-1.,x[4]])
    return n / np.linalg.norm(n),x[:3]
