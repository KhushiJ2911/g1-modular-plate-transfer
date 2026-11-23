"""Spawn calls on a DEPLOYED Modal app (server-side; independent of this laptop's client).

python spawn.py APP FUNCTION 'JSON list of arg lists' ['JSON with_options']
"""
import json
import sys

import modal

app, fn, args = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
opts = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
f = modal.Function.from_name(app, fn)
if opts:
    f = f.with_options(**opts)
for a in args:
    call = f.spawn(*a)
    print(fn, a[:2], call.object_id, flush=True)
