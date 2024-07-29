
declare -a models=(mistralai/Mistral-7B-Instruct-v0.2)
declare -a triples=(taxonomic sense_based_ns)
# declare -a templates=(initial-qa variation-qa-1 variation-qa-1-mistral-special variation-qa-2)
declare -a templates=(variation-qa-1-mistral-special)

TRANSFORMERS_CACHE="/home/shared/km_cache"

# python src/_test.py

for triples in ${triples[@]}; do
    
    if [[ $triples == "taxonomic" ]]; then
        triples_path="data/things/things-triples.csv"
        save_dir="data/things/results/taxonomic/"
    elif [[ $triples == "sense_based_ns" ]]; then
        triples_path="data/things/things-sense_based_ns-triples.csv"
        save_dir="data/things/results/things-sense_based_ns/"
    fi

    for model in "${models[@]}"; do
        for template in "${templates[@]}"; do
            echo "Running experiment for model $model and template $template"

            python src/eval_change_batched.py \
                --batch_size 16 \
                --num_examples 32 \
                --device cuda:1 \
                --model $model \
                --triples_path $triples_path \
                --qa_format \
                --prompt_template $template
            
            python src/eval_change_batched.py \
                --batch_size 16 \
                --num_examples -1 \
                --device cuda:1 \
                --model $model \
                --triples_path $triples_path \
                --save \
                --save_dir $save_dir \
                --qa_format \
                --prompt_template $template

            python src/eval_change_batched.py \
                --batch_size 16 \
                --num_examples -1 \
                --device cuda:1 \
                --model $model \
                --triples_path $triples_path \
                --save \
                --save_dir $save_dir \
                --qa_format \
                --prompt_template $template\
                --chat_format


            # python src/eval_change_batched.py \
            #     --batch_size 16 \
            #     --num_examples -1 \
            #     --device cuda:1 \
            #     --model $model \
            #     --triples_path data/things/things-sense_based_ns-triples.csv \
            #     --save \
            #     --save_dir data/things/results/things-sense_based_ns/ \
            #     --qa_format \
            #     --prompt_template $template \
            #     --induction

            # python src/eval_change_batched.py \
            #     --batch_size 16 \
            #     --num_examples -1 \
            #     --device cuda:1 \
            #     --model $model \
            #     --triples_path data/things/things-sense_based_ns-triples.csv \
            #     --save \
            #     --save_dir data/things/results/things-sense_based_ns/ \
            #     --qa_format \
            #     --prompt_template $template\
            #     --chat_format \
            #     --induction
        
        done
    done
done