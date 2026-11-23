# Brief #14: maintain the project research log

You work in `/home/keyu/advaitic/research_log` and write ONLY `RESEARCH_LOG.md` there (plus
`.log_state.json`). Everything else is read-only.

## Sources (read-only; the facts must come from these, never invent numbers)
- `/home/keyu/advaitic/robot-learning-assignment/EXPERIMENTS.md`: the master chronological log
  (every run, failure, number, cost). This is the primary source.
- `/home/keyu/advaitic/robot-learning-assignment/README.md` (the task and rules), `PLAN.md`, `METHODS.md`.
- `/home/keyu/advaitic/briefs/*.md` (what each Codex task was asked to build) and `*.report.md`
  (what was built), and the `11_muon.md` report at `/home/keyu/advaitic/wt-muon/method/vla/11_muon.md`.

## What to write: `RESEARCH_LOG.md`
A readable research narrative of the whole project, from the start to now, that someone could use to
write a detailed blog post and the final report. Structure:
1. **Task & constraints**: one short section (G1 humanoid plate transfer in BiGym2, 90% target, the
   observation contract, dev / confirm / test seed discipline, budget/thermal constraints).
2. **Timeline by phase**: e.g. BC baseline → privileged teacher → teacher data & DART → BC students →
   gripper BCE (bce1) → DAgger attempts & strict relabel → microVLA → Modal migration → benchmark
   (larger from-scratch + Muon vs pretrained VLAs, fine-tuned and zero-shot) → post-training (SFT, data,
   harness). For each phase: the question, what we tried, the result with the exact numbers and
   n (e.g. 57/90), what we concluded, and what it changed next.
3. **Results table**: every evaluated policy with dev success (k/n), seeds, key config, cost.
4. **What worked / what didn't**: two honest lists, including ops failures (idle GPUs, bad launches,
   label bugs, timeouts) and how each was fixed.
5. **Open questions & next steps.**
Rules: cite numbers exactly as in EXPERIMENTS.md; when a result is uncertain or n is small, say so; mark
anything inferred as such. Neutral, factual tone; no hype. Keep the user's personal data out.

## Incremental updates
Record in `.log_state.json` how far into EXPERIMENTS.md you have incorporated (line count + last
heading). On later runs ("update"), read only what is new since then, integrate it into the right
sections (don't just append), update the results table, and bump the state file. If this file
already exists, this run is an update.

Report (the `-o` file): 5-10 lines on what you added or changed this run.
