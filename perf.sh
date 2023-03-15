#!/bin/bash

clang++ -Wno-deprecated-register -mavx512f -lpthread -O3 kernel.cc test.cc -o ./perf.out

for i in {1..14}
do
  n=$((1<<i))
  nadd=$((n+1))
  nsub=$((n-1))
  ./perf.out $n $n $n 0 1
  ./perf.out $nadd $nadd $nadd 0 10
  ./perf.out $nsub $nsub $nsub 0 10
done
