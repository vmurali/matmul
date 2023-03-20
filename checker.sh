#!/bin/bash

make
cp a.out checker

for t in 1
  do
  for j in {1..33}
  do
    for i in {1..33}
    do
      for k in {1..20}
      do
        echo "$i $j $k 1 $t"
        ./checker $i $j $k 1 $t
        error=$?
        if [[ $error -ne 0 ]]; then
          echo "Wrong($error): $i $j $k $t"
          exit 1
        fi
      done
    done
  done
done
