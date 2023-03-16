#!/bin/bash

clang++ -Wno-deprecated-register -mavx512f -lpthread -O3 Kernel.cc Test.cc -o ./perf

./perf 384 128 512 0 10
./perf 384 512 128 0 10
./perf 384 384 32 0 10
./perf 384 32 384 0 10
./perf 384 128 128 0 10
./perf 384 512 384 0 10
./perf 384 2 512 0 10

for i in {1..14}
do
  n=$((1<<i))
  nadd=$((n+1))
  nsub=$((n-1))
  ./perf $n $n $n 0 10
  ./perf $nadd $nadd $nadd 0 10
  ./perf $nsub $nsub $nsub 0 10
done

