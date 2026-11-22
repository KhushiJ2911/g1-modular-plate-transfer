#!/usr/bin/env bash
# Fetch the original evidence bundle into a new local directory.
set -euo pipefail
if [[ $# -ne 2 || ! $1 =~ ^[A-Za-z0-9_-]{1,80}$ ]]; then
    echo "Usage: $0 RUN_NAME NEW_LOCAL_DIRECTORY" >&2
    exit 2
fi
dest=${2%/}
if [[ -z $dest || -e $dest || -L $dest ]]; then
    echo "Choose a new local directory: $2" >&2
    exit 2
fi
parent=$(dirname -- "$dest")
mkdir -p -- "$parent"
stage=$(mktemp -d -- "$parent/.lead-video.XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
uv run --frozen --with modal modal volume get advaitic-runs "/modular/$1/eval" "$stage"
test -s "$stage/eval/report.json"
mv -nT -- "$stage/eval" "$dest"
test ! -e "$stage/eval"
