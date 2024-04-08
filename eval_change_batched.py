import numpy as np
import random
import torch
import argparse

from src.pilot import get_triples
from tqdm import tqdm
from minicons import scorer
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def eval_change(
    model,
    num_examples,
    triples_path,
    lemmas_path,
    quantize=False,
    induction=False,
    qa_format=False,
):
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
        model = scorer.IncrementalLMScorer(model, device="cpu")

    # uniform subsample
    random.seed(12)
    random.shuffle(triples)
    
    if num_examples == -1:
        num_examples = len(triples)
    triples = triples[:num_examples]

    formatted_triples = []
    for triple in tqdm(triples, desc="Examples", total=num_examples):
        if qa_format:
            prefixes = [
                f"Is it true that {triple[0]}? Answer with yes/no: ",
                triple[1] + " ",
                triple[2] + " ",
            ]
            queries = ["Yes"] * 3
        else:
            prefixes = [
                "",
                triple[1].split("Conclusion:")[0] + "Conclusion: ",
                triple[2].split("Conclusion:")[0] + "Conclusion: ",
            ]
            queries = [triple[0]] * 3

        formatted_triples.append((prefixes, queries))

    control_minus_empty = []
    prompt_minus_control = []
    triples_dl = DataLoader(formatted_triples, batch_size=args.batch_size)
    for batch in tqdm(triples_dl, desc="Batches", total=len(triples_dl)):
        prefixes, queries = batch
        scores = model.conditional_score(prefixes, queries)
        for i in range(len(scores[0])):
            prompt_minus_control.append(scores[2][i] - scores[1][i])
            control_minus_empty.append(scores[1][i] - scores[0][i])

    print(f"control - empty: {np.mean(control_minus_empty)} ({np.std(control_minus_empty)})")
    print(f"prompt - control: {np.mean(prompt_minus_control)} ({np.std(prompt_minus_control)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7b-v0.1")
    parser.add_argument("--num_examples", type=int, default=100)
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
        "--batch_size", type=int, default=8, help="Batch size for evaluation."
    )
    args = parser.parse_args()

    triples_path = "data/things/things-triples.csv"
    lemmas_path = "data/things/things-lemmas-annotated.csv"
    eval_change(
        args.model,
        args.num_examples,
        triples_path,
        lemmas_path,
        quantize=args.quantize,
        induction=args.induction,
        qa_format=args.qa_format,
    )
