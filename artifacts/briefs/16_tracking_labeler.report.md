Implemented the [labeler](/home/keyu/advaitic/wt-teacher/method/teacher/track_label.py) and [report with sanity numbers and exact commands](/home/keyu/advaitic/wt-teacher/method/teacher/track_label_report.md).

- All 120 episodes: **102,230 states verified bit-for-bit**.
- Cost threshold 4 masks **16.12%**, retaining **85,747 fully labeled steps**.
- Teacher arm differences: mean **0.00194–0.00237**, maximum **0.06085**; worst differences match the requested `t*+1` indexing.
- **48 tests**, Ruff, and the independent archive audit passed.

Phase matching uses arm/gripper observations; aligned demo plate geometry is unavailable. Phase stalls are documented.

Data is staged in [results/dagger3_track](/home/keyu/advaitic/wt-teacher/results/dagger3_track). The sandbox prevents writing the requested external destination. Git metadata is read-only, so files remain uncommitted.