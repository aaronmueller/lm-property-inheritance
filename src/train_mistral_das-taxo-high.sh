#!/bin/bash

#layers=(0 5 10 15 20 25 30 35 41) #gemma2 9B
layers=(0 5 10 15 20 25 31) #for mistral 7B
#layers=(15 20 25 30 35 41)
positions=("premise_first" "premise_last" "conclusion_first" "conclusion_last" "last")
filter="high-sim-pos"
model="mistral"
sim="spose"

cuda_gpu="0"


# Nested loop to iterate over layers and positions
for layer in "${layers[@]}"; do
    for position in "${positions[@]}"; do
        # Call your Python script with the current layer and position
        echo "Running: python boundless_das2.py layer $layer position $position"
        python boundless_das2.py "$position" "$layer" "$filter" $cuda_gpu" "$sim" "$model"
    done
done
