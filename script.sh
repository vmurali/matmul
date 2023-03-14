#!/bin/bash

for i in {1..256}
do
  for j in {1..256}
  do
    for k in {1..256}
    do
      echo "$i $j $k"
      ./a.out $i $j $k
      error=$?
      if [[ $error -ne 0 ]]; then
        echo "Wrong($error): $i $j $k"
        exit 1
      fi
    done
  done
done
