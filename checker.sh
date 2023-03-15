#!/bin/bash

clang++ -Wno-deprecated-register -mavx512f -lpthread -O3 Kernel.cc Test.cc -o ./checker

for k in {1..2}
do
  for j in {1..256}
  do
    for i in {1..256}
    do
      for t in {1..256}
      do
        echo "$i $j $k 1 1 $t"
        ./checker $i $j $k 1 1 $t
        error=$?
        if [[ $error -ne 0 ]]; then
          echo "Wrong($error): $i $j $k $t"
          exit 1
        fi
      done
    done
  done
done
