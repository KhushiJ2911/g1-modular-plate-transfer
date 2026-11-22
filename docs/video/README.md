# Plate transfer from six source slots

The 57.15-second reel shows all six source slots on frozen candidate `876a00f`. Of the eight selected development episodes, 7 succeeded. In slot 3, seed 1044006 reopened the gripper at 0:25.24 in the reel and went on to complete the transfer. [Play the video](plate_transfer_876a00f.mp4)

| Reel time | Development seed | Source slot (0–5) | Outcome | Footage |
|---|---:|---:|---|---|
| 0:00.00 | 1044003 | 0 | Success | Full episode, 2× |
| 0:06.55 | 1044004 | 2 | Success | Full episode, 2× |
| 0:12.10 | 1044005 | 5 | Success | Full episode, 2× |
| 0:17.70 | 1044006 | 3 | Grasp retry, then success | Full episode, 1× |
| 0:35.20 | 1044007 | 4 | Success | Full episode, 2× |
| 0:40.40 | 1044008 | 1 | Success | Full episode, 2× |
| 0:45.95 | 1044009 | 1 | Success | Full episode, 2× |
| 0:51.15 | 1044010 | 2 | Timeout | Final 6 s, 1× |

The recovery chapter starts at 0:17.70. The action trace shows a close at step 120, an open at 377 and another close at 438, before base transport began at 515. The evaluator recorded success after 879 steps. Step numbers here are zero-based action indices, except for the evaluator's episode length.

The three panels show the native head, right-wrist and left-wrist camera observations, in that order. The reel adds magnification, captions and chapter markers. The eight original 10 fps MP4s contain each complete episode at simulator speed. This development block was selected to illustrate recovery. [Original evaluator report](originals/report.json) · [Clip and report checksums](checksums.sha256)

## Retrieve the footage

The files are on the `chinmaykarkar7` Modal account in `advaitic-runs:/modular/deliverable_video_876a00f_dev_1044003`. From the repository root with access to that account, run the following in a checkout without the downloaded files.

```bash
uv run --frozen --with modal modal volume get advaitic-runs /modular/deliverable_video_876a00f_dev_1044003/plate_transfer_876a00f.mp4 docs/video/plate_transfer_876a00f.mp4
ops_modular/lead_video_download.sh deliverable_video_876a00f_dev_1044003 docs/video/originals
(cd docs/video && sha256sum -c checksums.sha256)
```

The recording used one T4, the unchanged evaluator, policy `submission.policy_modular_round4:make_measured_policy` and head `64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff`. The evaluator submission SHA-256 was `b93718baa4c6f29aed446d7a2d6c050cf390de8e6ae285b673ddeab5b16912a2`, matching the archived development evaluation. FFprobe counted 1188 native frames and 1143 presentation frames. FFmpeg decoded all nine videos end to end.
