"""Code to get LM log probabilities for taxonomic sentence verification stimuli."""

import argparse
import pathlib
import utils

from minicons import scorer
from torch.utils.data import DataLoader
from tqdm import tqdm


def main(args):
    model = args.model
    model_name = model.replace("/", "_")

    lm = scorer.IncrementalLMScorer(model, device=args.device, torch_dtype="bfloat16")

    tsv_stimuli = utils.read_csv_dict(args.tsv_stimuli)

    dl = DataLoader(tsv_stimuli, batch_size=args.batch_size, shuffle=False)

    results = []
    for batch in tqdm(dl, desc="Batches"):
        scores = lm.conditional_score(batch['prefix'], batch['stimulus'])

        for i, s in enumerate(scores):
            results.append(
                {
                    "hyponym": batch["hyponym"][i],
                    "hypernym": batch["hypernym"][i],
                    "prefix": batch["prefix"][i],
                    "stimulus": batch["stimulus"][i],
                    "hypernymy": batch["hypernymy"][i],
                    "score": s,
                }
            )

    utils.write_csv_dict(f"data/things/things-{model_name}-tsv.csv", results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tsv_stimuli", type=pathlib.Path, default="data/things/things-tsv-stimuli.csv")
    args = parser.parse_args()

    main(args)
