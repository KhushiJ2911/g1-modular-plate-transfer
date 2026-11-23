#!/bin/bash
# wait_results.sh NAME... : poll until every run has a report; print them.
while true; do out=$(/home/keyu/advaitic/ops/collect.sh "$@"); echo "$out" | /usr/bin/grep -q pending || break; sleep 180; done
echo "$out"
