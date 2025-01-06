#!/bin/bash

# Set default values
test_size=0.2
random_state=42
n_estimators=100
learning_rate=0.1
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
    --n-estimators*)
      n_estimators="${arg#--n-estimators=*}"
      ;;
    --learning-rate*)
      learning_rate="${arg#--learning-rate=*}"
      ;;
    --save*)
      save=true
      ;;
  esac
done

# Run wineQuality.py with arguments
python wineQuality.py --test_size $test_size --random_state $random_state --n_estimators $n_estimators --learning_rate $learning_rate --save $save