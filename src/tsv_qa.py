import argparse
import pathlib
import utils

from minicons import scorer
from torch.utils.data import DataLoader
from tqdm import tqdm

def main(args):
    model = args.model
    model_name = model.replace("/", "_")

    qa_type = "qa-declarative" if "declarative" in args.tsv_stimuli else "qa"

    lm = scorer.IncrementalLMScorer(model, device=args.device, torch_dtype="bfloat16")

    tsv_stimuli = utils.read_csv_dict(args.tsv_stimuli)

    dl = DataLoader(tsv_stimuli, batch_size=args.batch_size, shuffle=False)

    results = []
    for batch in tqdm(dl, desc="Batches"):
        yes_scores = lm.conditional_score(batch['question'], ['Yes'] * len(batch['question']))
        no_scores = lm.conditional_score(batch['question'], ['No'] * len(batch['question']))

        for i, (y, n) in enumerate(zip(yes_scores, no_scores)):
            results.append(
                {
                    "hyponym": batch["hyponym"][i],
                    "hypernym": batch["hypernym"][i],
                    "question": batch["question"][i],
                    "hypernymy": batch["hypernymy"][i],
                    "score": y - n
                }
            )

    utils.write_csv_dict(f"data/things/things-{model_name}-tsv-{qa_type}.csv", results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tsv_stimuli", type=str, default="data/things/things-tsv-qa-declarative-stimuli.csv")
    args = parser.parse_args()

    main(args)
