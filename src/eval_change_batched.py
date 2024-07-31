import argparse
import csv
import config
import pathlib
import random
import torch

import numpy as np

from minicons import scorer
from pilot import generate_stimuli
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BitsAndBytesConfig


def main(args):

    model = args.model
    model_name = model.replace("/", "_")
    num_examples = args.num_examples
    triples_path = args.triples_path
    lemmas_path = args.lemmas_path
    quantize = args.quantize
    induction = args.induction
    qa_format = args.qa_format
    chat_format = args.chat_format
    prompt_template = args.prompt_template

    label_separator = config.PROMPTS[prompt_template]["label-separator"]

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = scorer.IncrementalLMScorer(
            model, quantization_config=bnb_config, device="auto"
        )
    else:
        model = scorer.IncrementalLMScorer(model, device=args.device)

    if chat_format:
        tokenizer = model.tokenizer
    else:
        tokenizer = None

    stimuli = generate_stimuli(
        triples_path,
        lemmas_path,
        prompt_cfg=config.PROMPTS[prompt_template],
        induction=induction,
        tokenizer=tokenizer,
    )

    # uniform subsample
    random.seed(12)

    if num_examples == -1:
        num_examples = len(stimuli)
    else:
        # shuffle only if not all samples have been specified
        if args.save:
            raise Exception(
                "cannot shuffle when saving examples (i.e., when --save has been passed)."
            )
        random.shuffle(stimuli)

    stimuli = stimuli[:num_examples]

    formatted_stimuli = []
    for instance in tqdm(stimuli, desc="Examples", total=num_examples):
        if qa_format:
            prefixes = instance
            queries = ["Yes"] * 3
        else:
            prefixes = ["", instance[1], instance[2]]
            queries = [instance[0]] * 3
        formatted_stimuli.append((prefixes, queries))

    print(formatted_stimuli[:4])

    control_minus_empty = []
    prompt_minus_control = []

    empty_scores_all = []
    control_scores_all = []
    prompt_scores_all = []

    triples_dl = DataLoader(formatted_stimuli, batch_size=args.batch_size)

    for batch in tqdm(triples_dl, desc="Batches", total=len(triples_dl)):
        prefixes, queries = batch
        empty_prefixes, control_prefixes, prompt_prefixes = prefixes
        empty_queries, control_queries, prompt_queries = queries
        prefixes = empty_prefixes + control_prefixes + prompt_prefixes
        queries = empty_queries + control_queries + prompt_queries

        if qa_format:
            empty_scores_yes = model.conditional_score(
                empty_prefixes, empty_queries, separator=label_separator
            )
            control_scores_yes = model.conditional_score(
                control_prefixes, control_queries, separator=label_separator
            )
            prompt_scores_yes = model.conditional_score(
                prompt_prefixes, prompt_queries, separator=label_separator
            )

            empty_scores_no = model.conditional_score(
                empty_prefixes, ["No"] * len(empty_queries), separator=label_separator
            )
            control_scores_no = model.conditional_score(
                control_prefixes,
                ["No"] * len(control_queries),
                separator=label_separator,
            )
            prompt_scores_no = model.conditional_score(
                prompt_prefixes, ["No"] * len(prompt_queries), separator=label_separator
            )

            empty_scores = np.array(empty_scores_yes) - np.array(empty_scores_no)
            control_scores = np.array(control_scores_yes) - np.array(control_scores_no)
            prompt_scores = np.array(prompt_scores_yes) - np.array(prompt_scores_no)
        else:
            empty_scores = model.sequence_score(empty_queries)
            control_scores = model.conditional_score(control_prefixes, control_queries)
            prompt_scores = model.conditional_score(prompt_prefixes, prompt_queries)

        empty_scores_all.extend(empty_scores)
        control_scores_all.extend(control_scores)
        prompt_scores_all.extend(prompt_scores)

    control_minus_empty = np.array(control_scores_all) - np.array(empty_scores_all)
    prompt_minus_control = np.array(prompt_scores_all) - np.array(control_scores_all)

    if args.save:
        # model_name = model.replace("/", "_")
        save_dir = args.save_dir
        if args.induction:
            save_dir += "/induction/"
        else:
            save_dir += "/deduction/"
        if args.qa_format:
            save_dir += "qa_format"
        else:
            save_dir += "logprobs"

        model_name += f"_{args.prompt_template}"
        if args.chat_format:
            model_name += "_chat-format"

        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{save_dir}/{model_name}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["empty", "control", "prompt"])
            for e, c, p in zip(empty_scores_all, control_scores_all, prompt_scores_all):
                writer.writerow([e, c, p])

    # print if flag is passed
    if not args.dont_debug:
        print(f"Empty: {np.mean(empty_scores_all)} ({np.std(empty_scores_all)})")
        print(f"Control: {np.mean(control_scores_all)} ({np.std(control_scores_all)})")
        print(f"Prompt: {np.mean(prompt_scores_all)} ({np.std(prompt_scores_all)})")
        print("\n\n")
        print(
            f"control - empty: {np.mean(control_minus_empty)} ({np.std(control_minus_empty)})"
        )
        print(
            f"prompt - control: {np.mean(prompt_minus_control)} ({np.std(prompt_minus_control)})"
        )

        diff_behavior = 0
        for i in range(len(control_minus_empty)):
            if prompt_minus_control[i] > 0:
                diff_behavior += 1

        print(f"Diff Behavior: {diff_behavior / len(control_minus_empty)}")

        # accuracy = 0
        # for i in range(len(control_minus_empty)):
        #     # Accuracy = num of times prompt > control
        #     if prompt_scores_all[i] > 0:
        #         accuracy += 1

        # print(f"Prompt Diff Accuracy: {accuracy / len(control_minus_empty)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7b-v0.1")
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument(
        "--triples_path",
        type=str,
        default="data/things/things-triples.csv",
        help="Path to file containing the triples in the form of hypernym/anchor/hyponym.",
    )
    parser.add_argument(
        "--lemmas_path",
        type=str,
        default="data/things/things-lemmas-annotated.csv",
        help="Path to file containing lexical information of concepts such as their singular and plural forms, the form used for expressing property knowledge, etc.",
    )
    parser.add_argument(
        "--induction",
        action="store_true",
        help="If false (default), evaluate deduction.",
    )
    parser.add_argument(
        "--qa_format",
        action="store_true",
        help="If false (default), get probability of 'X has property.' If true, use yes/no contrasts.",
    )
    parser.add_argument(
        "--chat_format", action="store_true", help="If true, use chat format."
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="initial-qa",
        help="Prompt template key based on prompts in config.py",
    )
    parser.add_argument(
        "--quantize", action="store_true", help="If true, use 4-bit quantization."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for running inference."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="If true, save examples to a file.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/things/results/taxonomic",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--dont_debug", action="store_true", help="If true, don't print diff results."
    )
    args = parser.parse_args()
    main(args)
