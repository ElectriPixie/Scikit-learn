#!/bin/bash

# Set default values
test_size=0.2
random_state=42
n_neighbors=5
save=false

# Parse command-line arguments
for arg in "$@"; do
  case $arg in
    --test-size*)
      test_size="${arg#--test-size=*}"
      ;;
    --random-state*)
      random_state="${arg#--random-state=*}"
      ;;
    --n-neighbors*)
      n_neighbors="${arg#--n-neighbors=*}"
      ;;
    --save*)
      save=true
      ;;
  esac
done

# Run wineQuality.py with arguments
python wineQuality.py --test_size $test_size --random_state $random_state --n_neighbors $n_neighbors --save $save