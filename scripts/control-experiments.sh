# declare -a models=(google/Gemma-2-9B-it google/Gemma-2-2B-it meta-llama/Meta-Llama-3-8B-Instruct)
declare -a triples=(sense_based_ns)

TRANSFORMERS_CACHE="/home/shared/km_cache"
device="cuda:0"

for triples in ${triples[@]}; do
    if [[ $triples == "sense_based_ns" ]]; then
        triples_path="data/things/stimuli-pairs/things-inheritance-sense_based_sim-pairs.csv"
        save_dir="data/things/results/things-sense_based-ns"
    elif [[ $triples == "spose_prototype" ]]; then
        triples_path="data/things/stimuli-pairs/things-inheritance-SPOSE_prototype_sim-pairs.csv"
        save_dir="data/things/results/things-SPOSE_prototype-ns"
    fi

    # python src/behavioral_eval.py \
    #     --batch_size 16 \
    #     --num_examples -1 \
    #     --device $device \
    #     --model google/Gemma-2-9B-it \
    #     --triples_path $triples_path \
    #     --save \
    #     --save_dir ${save_dir}_multi-property \
    #     --qa_format \
    #     --prompt_template variation-qa-2 \
    #     --chat_format \
    #     --multi_property

    # python src/behavioral_eval.py \
    #     --batch_size 16 \
    #     --num_examples -1 \
    #     --device $device \
    #     --model google/Gemma-2-9B-it \
    #     --triples_path $triples_path \
    #     --save \
    #     --save_dir ${save_dir}_multi-property_prop-contrast \
    #     --qa_format \
    #     --prompt_template variation-qa-2 \
    #     --chat_format \
    #     --multi_property \
    #     --prop_contrast

    # python src/behavioral_eval.py \
    #     --batch_size 16 \
    #     --num_examples -1 \
    #     --device $device \
    #     --model google/Gemma-2-2B-it \
    #     --triples_path $triples_path \
    #     --save \
    #     --save_dir ${save_dir}_multi-property \
    #     --qa_format \
    #     --prompt_template variation-qa-1 \
    #     --multi_property

    # python src/behavioral_eval.py \
    #     --batch_size 16 \
    #     --num_examples -1 \
    #     --device $device \
    #     --model google/Gemma-2-2B-it \
    #     --triples_path $triples_path \
    #     --save \
    #     --save_dir ${save_dir}_multi-property_prop-contrast \
    #     --qa_format \
    #     --prompt_template variation-qa-1 \
    #     --multi_property \
    #     --prop_contrast

    python src/behavioral_eval.py \
        --batch_size 16 \
        --num_examples -1 \
        --device $device \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --triples_path $triples_path \
        --save \
        --save_dir ${save_dir}_multi-property \
        --qa_format \
        --prompt_template variation-qa-2 \
        --multi_property

    python src/behavioral_eval.py \
        --batch_size 16 \
        --num_examples -1 \
        --device $device \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --triples_path $triples_path \
        --save \
        --save_dir ${save_dir}_multi-property_prop-contrast \
        --qa_format \
        --prompt_template variation-qa-2 \
        --multi_property \
        --prop_contrast

    python src/behavioral_eval.py \
        --batch_size 16 \
        --num_examples -1 \
        --device $device \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --triples_path $triples_path \
        --save \
        --save_dir ${save_dir}_multi-property \
        --qa_format \
        --prompt_template variation-qa-1-mistral-special \
        --multi_property

    python src/behavioral_eval.py \
        --batch_size 16 \
        --num_examples -1 \
        --device $device \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --triples_path $triples_path \
        --save \
        --save_dir ${save_dir}_multi-property_prop-contrast \
        --qa_format \
        --prompt_template variation-qa-1-mistral-special \
        --multi_property \
        --prop_contrast
done

