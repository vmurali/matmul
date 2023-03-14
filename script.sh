#!/bin/bash

make

for k in {1..2}
do
  for j in {1..256}
  do
    for i in {1..256}
    do
      for t in {1..256}
      do
        echo "$i $j $k $t"
        ./a.out $i $j $k $t
        error=$?
        if [[ $error -ne 0 ]]; then
          echo "Wrong($error): $i $j $k $t"
          exit 1
        fi
      done
    done
  done
done
