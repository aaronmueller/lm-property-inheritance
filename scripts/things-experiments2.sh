
# declare -a models=(google/Gemma-2-2B-it google/Gemma-2-9B-it)
declare -a models=(mistralai/Mistral-7B-Instruct-v0.2)
# declare -a triples=(taxonomic sense_based_ns spose_prototype)
# declare -a models=(meta-llama/Llama-3-8B-Instruct)
# declare -a triples=(sense_based_ns spose_prototype)
# declare -a triples=(spose_prototype)
declare -a triples=(spose_prototype)
declare -a templates=(initial-qa variation-qa-1 variation-qa-2)

TRANSFORMERS_CACHE="/home/shared/km_cache"

device="cuda:0"
# python src/_test.py

for triples in ${triples[@]}; do
    
    if [[ $triples == "taxonomic" ]]; then
        triples_path="data/things/things-triples-actual.csv"
        save_dir="data/things/results/taxonomic/"
    elif [[ $triples == "sense_based_ns" ]]; then
        triples_path="data/things/negative-samples/things-sense_based-ns_triples.csv"
        save_dir="data/things/results/things-sense_based-ns/"
    elif [[ $triples == "spose_prototype" ]]; then
        triples_path="data/things/negative-samples/things-SPOSE_prototype-ns_triples.csv"
        save_dir="data/things/results/things-SPOSE_prototype-ns/"
    fi

    for model in "${models[@]}"; do
        for template in "${templates[@]}"; do
            echo "Running experiment for model $model and template $template"

            # python src/eval_change_batched.py \
            #     --batch_size 16 \
            #     --num_examples 32 \
            #     --device $device \
            #     --model $model \
            #     --triples_path $triples_path \
            #     --qa_format \
            #     --prompt_template $template
            
            python src/eval_change_batched.py \
                --batch_size 16 \
                --num_examples -1 \
                --device $device \
                --model $model \
                --triples_path $triples_path \
                --save \
                --save_dir $save_dir \
                --qa_format \
                --prompt_template $template

            python src/eval_change_batched.py \
                --batch_size 16 \
                --num_examples -1 \
                --device $device \
                --model $model \
                --triples_path $triples_path \
                --save \
                --save_dir $save_dir \
                --qa_format \
                --prompt_template $template\
                --chat_format


            python src/eval_change_batched.py \
                --batch_size 16 \
                --num_examples -1 \
                --device $device \
                --model $model \
                --triples_path $triples_path \
                --save \
                --save_dir $save_dir \
                --qa_format \
                --prompt_template $template \
                --induction

            python src/eval_change_batched.py \
                --batch_size 16 \
                --num_examples -1 \
                --device $device \
                --model $model \
                --triples_path $triples_path \
                --save \
                --save_dir $save_dir \
                --qa_format \
                --prompt_template $template\
                --chat_format \
                --induction
        
        done
    done
done
