#!/bin/bash

layers=(10)
#layers=(15)
#layers=(0 5 10 15 20 25 31) #for mistral
#layers=(0 5 10 15 20 25 30 35 41) #for gemma-2 9B
#layers=(15 20 25 30 35 41)
#layers=(0 5 10 15 20 25 31) #for mistral 7B
#positions=("premise_first" "premise_last" "conclusion_first" "conclusion_last" "last")
positions=("conclusion_last")
portions=(1 2 3 4)
#portions=(1)

cuda_gpu=0
topdir="models-spose"
#-spose"

trainfilter="balanced"
testfilter="balanced"

# Nested loop to iterate over layers and positions
for layer in "${layers[@]}"; do
    for position in "${positions[@]}"; do
        for portion in "${portions[@]}"; do
            # Call your Python script with the current layer and position
            echo "Running: python eval_boundless_das_rev.py layer $layer position $position" portion "$portion"
            python eval_boundless_das_revt.py "$position" "$layer" "$trainfilter" "$testfilter" "$portion" "$cuda_gpu" "$topdir"
        done
    done
done
