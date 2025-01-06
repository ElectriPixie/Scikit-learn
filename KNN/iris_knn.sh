#!/bin/bash

# Set default values
n_neighbors=5
average="macro"
save=false

# Parse command-line arguments
for arg in "$@"; do
  case $arg in
    --n_neighbors*)
      n_neighbors="${arg#--n_neighbors=}"
      ;;
    --average*)
      average="${arg#--average=}"
      ;;
    --save*)
      save=true
      ;;
  esac
done

# Run the Python script with the provided values
python3 iris_knn.py --n_neighbors "$n_neighbors" --average "$average" --save "$save"
