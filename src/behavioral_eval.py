import argparse
import csv
import config
import pathlib
import random
import torch
import utils

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
    # if "gemma" in model_name.lower():
    #     label_separator = ""

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

    triples = utils.read_csv_dict(triples_path)
    bad_ids = []
    for i, t in enumerate(triples):
        if t["premise"] == t["conclusion"]:
            bad_ids.append(i)

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

    formatted_stimuli = [stimulus[-1] for stimulus in stimuli]
    # for instance in tqdm(stimuli, desc="Examples", total=num_examples):
    #     if qa_format:
    #         prefixes = instance[-1]
    #         queries = ["Yes"] * 3
    #     else:
    #         prefixes = ["", instance[1], instance[2]]
    #         queries = [instance[0]] * 3
    #     formatted_stimuli.append((prefixes, queries))

    print(formatted_stimuli[:4])

    stimuli_dl = DataLoader(formatted_stimuli, batch_size=args.batch_size)

    predictions = []
    truth = [s["hypernymy"].title() for i, s in enumerate(triples) if i not in bad_ids]

    logprob_results = []

    for batch in tqdm(stimuli_dl):
        dist = model.next_word_distribution(batch)
        queries = [("Yes", "No")] * len(batch)

        logprobs, ranks = model.query(dist, queries, prob=False)

        for y, n in logprobs:
            logprob_results.append((y, n, y - n))
            if y > n:
                predictions.append("Yes")
            else:
                predictions.append("No")

    filtered_predictions = [p for i, p in enumerate(predictions) if i not in bad_ids]
    sensitivity = np.mean(
        [truth[i] == filtered_predictions[i] for i in range(len(filtered_predictions))]
    )
    print(f"Taxonomic Sensitivity: {sensitivity}")

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
            writer.writerow(["yes", "no", "diff"])
            writer.writerows(logprob_results)


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
