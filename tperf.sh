#!/bin/bash
# MIT License
# Copyright (c) 2025 Youcef Lemsafer
# See LICENSE file fore more details.
# Creation date: 2025.03.21

# This script used for generating the performances table in the readme file
# For each range below run the sieve 6 times remove the largest and smallest duration
# and compute the average.


ranges=(0 $((10**9))\
 0 $((2*(2**31-1)+1))\
 $((10**12)) $((10**12+10**10))\
 $((10**15)) $((10**15+10**10))\
 $((10**18)) $((10**18+10**10))\
 18446744063709551616 18446744073709551615\
)

threads_c=(1 4 8 16 32)

compute_avg() {
  local values=("$@")
  local count=${#values[@]}

  IFS=$'\n' sorted=($(printf "%s\n" "${values[@]}" | sort -n))
  unset IFS

  sorted=("${sorted[@]:1:$(($count-2))}")
  local sum=0
  for v in "${sorted[@]}"; do
    sum=$(echo "$sum + $v" | bc)
  done
  local avg=$(echo "scale=3; $sum / ${#sorted[@]}" | bc)
  echo "$avg"
}

for ((i = 0; i < ${#ranges[@]}; i += 2)); do
  n0=${ranges[i]}
  n1=${ranges[i+1]}
  if [[ $i == 0 ]]; then
      echo -n "| Range \\ Threads"
      for ((j = 0; j < ${#threads_c[@]}; j += 1)); do echo -n " | ${threads_c[j]}"; done
      echo " | Number of primes |"
      echo -n "|-----------------"
      for ((j = 0; j < ${#threads_c[@]}; j += 1)); do echo -n "|---"; done
      echo "|-------------------|"
  fi
  primes_c=0
  durations=()
  echo -n "[${n0}, ${n1})"
  for ((t = 0; t < ${#threads_c[@]}; t += 1)); do
    thrd=${threads_c[t]}
    durations=()
    for j in {1..6}; do
      s=$(./lfp -t $thrd $n0 $n1)
      tpc=$(echo $s | sed 's:.*is \([0-9]\+\)\..*:\1:g')
      if [[ $primes_c == 0 ]]; then
	primes_c=$tpc
      elif [[ $primes_c != $tpc ]]; then
	echo "Inconsistency in results, previous run gave ${primes_c} primes, last one gave ${tpc}."
	exit 1
      fi
      durations+=($(echo $s | sed 's:.* \([^s]\+\)s$:\1:g'))
    done
    avg=$(compute_avg "${durations[@]}")
    echo -n " | ${avg}"
  done
  echo " | $primes_c |"
done

echo "|----------------------------------------------------------------------|"

