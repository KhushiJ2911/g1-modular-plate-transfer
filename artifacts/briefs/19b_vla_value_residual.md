# Brief #19b (FAST, ~40 min): value residual learning in the microVLA transformer
Worktree /home/keyu/advaitic/wt-muon (branch muon, HEAD 28a180d: method/vla/train.py with AdamW/Muon, configs
microvla_v1.yaml and microvla_large.yaml; the model may live under method/vla/vendor/). Leave files
uncommitted (read-only git metadata; I commit). Tiny CPU checks only, ADV_THROTTLE=1.
Implement ResFormer-style value residuals (Zhou et al., "Value Residual Learning"): in every transformer layer
l >= 1 (backbone/prefix AND action expert stacks, each stack separately), replace the attention values V_l with
V_l' = lambda_l * V_l + (1 - lambda_l) * V_1, where V_1 are the first layer's values of that stack (same token
positions) and lambda_l is a learnable scalar per layer initialised to 0.5 (sigmoid-parametrised to stay in (0,1)).
Behind a config flag `value_residual: false` (default false => bit-identical to the current model; prove with an
equality test on a fixed seed). Make sure KV caching / prefix-attention paths used at inference
(submission/policy_vla.py in /home/keyu/advaitic/wt-vla is the inference wrapper; check the model's own
inference function) carry V_1 correctly. Also accept the flag through the yaml so a run can use
`--config method/vla/microvla_v1.yaml` plus an override. Tests: default-off equality; on: forward/backward,
lambda receives gradients, one 2-step CPU train. Report the exact override syntax for train.py.
