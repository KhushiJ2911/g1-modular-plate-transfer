"""V2 checkpoint selection, reset context and the 6-D rotation convention."""

import numpy as np
import pytest

torch=pytest.importorskip("torch")

from submission.estimator import Estimator
from submission.estimator_v2 import PlateEstimatorV2,from_initial_frame,rotation_matrix
from submission.model import PROPRIO_DIMS


def test_reset_transform_and_column_order():
    start=torch.zeros(1,50)
    current=start.clone()
    start[0,21]=np.pi/2
    current[0,18]=1.
    answer=from_initial_frame(torch.tensor([[1.,0.,2.]]),start,current)
    torch.testing.assert_close(answer,torch.tensor([[-1.,1.,2.]]))
    q=rotation_matrix(torch.tensor([[1.,0.,0.,0.,0.,1.]]))[0]
    torch.testing.assert_close(q[:,2],torch.tensor([0.,-1.,0.]))
    torch.testing.assert_close(q.T@q,torch.eye(3))


def test_checkpoint_and_public_reset(tmp_path):
    model=PlateEstimatorV2()
    path=tmp_path/"test.pt"
    torch.save(dict(version=2,model=model.state_dict(),config=dict(proprio_dims=PROPRIO_DIMS)),path)
    estimator=Estimator(path,"cpu")
    obs=dict(rgb=np.zeros((3,3,84,84),np.uint8),state=np.zeros(50,np.float32))
    first=estimator(obs)
    moved={"state":obs["state"].copy(),"rgb":np.full_like(obs["rgb"],125)}
    moved["state"][18]=.35
    second=estimator(moved)
    np.testing.assert_allclose(first["target_site_rel"]-[.35,0,0],second["target_site_rel"],atol=1e-6)
    estimator.reset()
    again=estimator(obs)
    for k in ("plate_rel","plate_rot_rel","target_site_rel"):
        np.testing.assert_allclose(again[k],first[k])
    assert set(again["contacts"])=={"source_rack","target_rack","table","floor"}
