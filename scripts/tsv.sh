# make tsv stimuli
python src/tsv_generation.py
python src/tsv_generation.py --qa
python src/tsv_generation.py --qa --declarative

# run tsv exps for mistral

declare -a models=(mistralai/Mistral-7b-Instruct-v0.2)

for model in "${models[@]}"; do
    python src/tsv.py --model $model
    python src/tsv_qa.py --model $model --tsv_stimuli data/things/tsv/stimuli/things-tsv-qa-stimuli.csv
    python src/tsv_qa.py --model $model --tsv_stimuli data/things/tsv/stimuli/things-tsv-qa-declarative-stimuli.csv
done
