# Plate transfer from six source slots

The 56.9-second reel covers all six source slots on frozen candidate `aa1e273`. The evaluator recorded 7 successes in eight selected development episodes. In slot 3, seed 1044006 reopened the gripper at 0:25.29 and completed the transfer on its next attempt. [Play the video](plate_transfer_aa1e273.mp4)

| Reel time | Development seed | Source slot (0–5) | Outcome | Footage |
|---|---:|---:|---|---|
| 0:00.00 | 1044003 | 0 | Success | Full episode, 2× |
| 0:06.55 | 1044004 | 2 | Success | Full episode, 2× |
| 0:12.10 | 1044005 | 5 | Success | Full episode, 2× |
| 0:17.75 | 1044006 | 3 | Grasp retry, then success | Full episode, 1× |
| 0:34.85 | 1044007 | 4 | Success | Full episode, 2× |
| 0:40.05 | 1044008 | 1 | Success | Full episode, 2× |
| 0:45.70 | 1044009 | 1 | Success | Full episode, 2× |
| 0:50.90 | 1044010 | 2 | Timeout | Final 6 s, 1× |

The recovery chapter begins at 0:17.75. The first close, reopening and second close occurred at action indices 120, 377 and 441. Base transport began at index 516. The evaluator recorded success after 858 steps.

The three panels show the native head, right-wrist and left-wrist camera observations, in that order. The reel adds magnification, captions and chapter markers. The eight original 10 fps MP4s contain each complete episode at simulator speed. We reused the eight development seeds from the previous candidate’s video so the starts can be compared. [Original evaluator report](originals/report.json) · [Clip and report checksums](checksums.sha256)

## Files

The reel and the eight original evaluator clips are committed in this folder. Verify them with:

```bash
(cd docs/video && sha256sum -c checksums.sha256)
```

The recording used one T4, the unchanged evaluator, policy `method.modular.vertical_clearance:make_policy` and head `64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff`. The evaluator submission SHA-256 was `1b05190d2013d5d08f230e9253615c13dd8fc5b009686e72a55bfa7fa5586f25`, matching the digest computed independently from the frozen files mounted for evaluation. FFprobe counted 1187 native frames and 1138 presentation frames. FFmpeg decoded all nine videos end to end.
