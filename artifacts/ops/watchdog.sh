#!/bin/bash
# Thermal / RAM guard for long jobs on this laptop.
# Only touches processes started with ADV_THROTTLE=1 in their environment
# (data generation, training). Official evaluator runs are never marked, so
# they are never paused (pausing them could cause a fake policy timeout).
# Pause:  CPU >= 85C or GPU >= 82C or MemAvailable < 2000 MB
# Resume: CPU <= 72C and GPU <= 70C and MemAvailable >= 3000 MB
LOG=$HOME/advaitic/ops/watchdog.log
paused=0
marked() {
  for d in /proc/[0-9]*; do
    p=${d#/proc/}
    [ "$p" = "$$" ] && continue
    tr '\0' '\n' < "$d/environ" 2>/dev/null | grep -qx 'ADV_THROTTLE=1' && echo "$p"
  done
}
last_ok=0
while true; do
  t=$(( $(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1) / 1000 ))
  g=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null || echo 0)
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ $paused -eq 0 ] && { [ "$t" -ge 85 ] || [ "$g" -ge 82 ] || [ "$avail" -lt 2000 ]; }; then
    pids=$(marked)
    [ -n "$pids" ] && kill -STOP $pids
    paused=1
    echo "$(date +%F_%T) PAUSE cpu=${t}C gpu=${g}C avail=${avail}MB pids=$(echo $pids)" >> "$LOG"
  elif [ $paused -eq 1 ] && [ "$t" -le 72 ] && [ "$g" -le 70 ] && [ "$avail" -ge 3000 ]; then
    pids=$(marked)
    [ -n "$pids" ] && kill -CONT $pids
    paused=0
    echo "$(date +%F_%T) RESUME cpu=${t}C gpu=${g}C avail=${avail}MB" >> "$LOG"
  fi
  now=$(date +%s)
  if [ $((now - last_ok)) -ge 300 ]; then
    echo "$(date +%F_%T) ok cpu=${t}C gpu=${g}C avail=${avail}MB paused=$paused" >> "$LOG"
    last_ok=$now
  fi
  sleep 10
done
