#!/bin/bash

clang++ -Wno-deprecated-register -mavx512f -lpthread -O3 Kernel.cc Test.cc -o ./perf

for i in {1..14}
do
  n=$((1<<i))
  nadd=$((n+1))
  nsub=$((n-1))
  ./perf $n $n $n 0 1
  ./perf $nadd $nadd $nadd 0 10
  ./perf $nsub $nsub $nsub 0 10
done
