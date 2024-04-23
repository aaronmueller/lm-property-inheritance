import argparse
import csv
import pathlib
import random
import torch

import numpy as np

from minicons import scorer
from pilot import get_triples
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

    triples = get_triples(
        triples_path, lemmas_path, qa_format=qa_format, induction=induction
    )

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

    # uniform subsample
    random.seed(12)

    if num_examples == -1:
        num_examples = len(triples)
    else:
        # shuffle only if not all samples have been specified
        if args.save:
            raise Exception(
                "cannot shuffle when saving examples (i.e., when --save has been passed)."
            )
        random.shuffle(triples)

    triples = triples[:num_examples]

    formatted_triples = []
    for triple in tqdm(triples, desc="Examples", total=num_examples):
        if qa_format:
            prefixes = [
                f"Is it true that {triple[0]}? Answer with Yes/No:",
                triple[1],
                triple[2],
            ]
            queries = ["Yes"] * 3
        else:
            prefixes = ["", triple[1], triple[2]]
            queries = [triple[0]] * 3

        formatted_triples.append((prefixes, queries))

    print(formatted_triples[:4])

    control_minus_empty = []
    prompt_minus_control = []

    empty_scores_all = []
    control_scores_all = []
    prompt_scores_all = []
    triples_dl = DataLoader(formatted_triples, batch_size=args.batch_size)
    for batch in tqdm(triples_dl, desc="Batches", total=len(triples_dl)):
        prefixes, queries = batch
        empty_prefixes, control_prefixes, prompt_prefixes = prefixes
        empty_queries, control_queries, prompt_queries = queries
        prefixes = empty_prefixes + control_prefixes + prompt_prefixes
        queries = empty_queries + control_queries + prompt_queries

        if qa_format:
            empty_scores_yes = model.conditional_score(empty_prefixes, empty_queries)
            control_scores_yes = model.conditional_score(
                control_prefixes, control_queries
            )
            prompt_scores_yes = model.conditional_score(prompt_prefixes, prompt_queries)

            empty_scores_no = model.conditional_score(
                empty_prefixes, ["No"] * len(empty_queries)
            )
            control_scores_no = model.conditional_score(
                control_prefixes, ["No"] * len(control_queries)
            )
            prompt_scores_no = model.conditional_score(
                prompt_prefixes, ["No"] * len(prompt_queries)
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

        # if args.num_samples != -1:
        #     save_dir += f"-{args.num_samples}/"

        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{save_dir}/{model_name}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["empty", "control", "prompt"])
            for e, c, p in zip(empty_scores_all, control_scores_all, prompt_scores_all):
                writer.writerow([e, c, p])

    # print if flag is passed
    # avg scores
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

        # print(prompt_scores_all[:10])

        diff_accuracy = 0
        for i in range(len(control_minus_empty)):
            # Accuracy = num of times prompt > control
            if prompt_minus_control[i] > 0:
                diff_accuracy += 1

        print(f"Diff Accuracy: {diff_accuracy / len(control_minus_empty)}")

        accuracy = 0
        for i in range(len(control_minus_empty)):
            # Accuracy = num of times prompt > control
            if prompt_scores_all[i] > 0:
                accuracy += 1

        print(f"Prompt Diff Accuracy: {accuracy / len(control_minus_empty)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7b-v0.1")
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument(
        "--triples_path", type=str, default="data/things/things-triples.csv"
    )
    parser.add_argument(
        "--lemmas_path", type=str, default="data/things/things-lemmas-annotated.csv"
    )
    parser.add_argument(
        "--induction",
        action="store_true",
        help="If false (default), evaluate deduction.",
    )
    parser.add_argument(
        "--qa_format",
        action="store_true",
        help="If false (default), get probability of 'NN is daxable.' If true, use yes/no contrasts.",
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
    parser.add_argument("--dont_debug", action="store_true", help="If true, don't print diff results.")
    args = parser.parse_args()

    # triples_path = "data/things/things-triples.csv"
    # lemmas_path = "data/things/things-lemmas-annotated.csv"
    # eval_change(
    #     args.model,
    #     args.num_examples,
    #     triples_path,
    #     lemmas_path,
    #     quantize=args.quantize,
    #     induction=args.induction,
    #     qa_format=args.qa_format,
    #     batch_size=args.batch_size,
    #     device=args.device,
    # )
    main(args)
