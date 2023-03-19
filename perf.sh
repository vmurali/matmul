#!/bin/bash

make
cp a.out perf

for i in {5..15}
do
  m=$((1<<i))
  n=$m
  k=$m
  for maxt in 1 #2 4 6 8 16 32
  do
    ./perf $m $n $k 0 $maxt
    ./perf $((m+1)) $((n+1)) $((k+1)) 0 $maxt
    ./perf $((m-1)) $((n-1)) $((k-1)) 0 $maxt
  done
done

