import argparse
import csv
import json

import lexicon
import config
import utils

from collections import defaultdict
from pilot import get_triples


def main(args):
    # triples = get_triples("data/things/things-sense-based-pairs.csv", "data/things/things-lemmas-annotated.csv", induction=False, qa_format=True)
    # print(triples[0])

    PROMPT = "Given a premise, produce a conclusion that is true.\nPremise: {}\nConclusion: {}"
    QA_PROMPT = "Given that {}, is it true that {}? Answer with Yes/No:"
    CONTROL = "nothing is daxable"

    # read in concepts
    concepts = defaultdict(lexicon.Concept)
    with open(args.lemma_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # concepts.append(lemma2concept(row))
            if row["remove"] != "1":
                concepts[row["lemma"]] = utils.lemma2concept(row)

    # construct fake property (can also refer to config.py)
    fake_property = lexicon.Property("daxable", "is daxable", "are daxable")

    # read triples
    pairs = utils.read_csv_dict(args.triple_path)
    stimuli = []
    control_cases = set()
    for i, pair in enumerate(pairs):
        # print concept(hyponym) is a concept(anchor)
        hyponym = pair["hyponym"]
        anchor = pair["anchor"]
        if hyponym in concepts.keys() and anchor in concepts.keys():
            child = concepts[hyponym]
            parent = concepts[anchor]
            if args.qa_format:
                _, control, stimulus = utils.create_sample(
                    parent, child, fake_property, QA_PROMPT, control_sentence=CONTROL, qa=True, induction=args.induction
                )
            else:
                _, control, stimulus = utils.create_sample(
                    parent, child, fake_property, PROMPT, control_sentence=CONTROL,
                    induction=args.induction
                )
            
            pairs[i]["stimulus"] = stimulus
            if args.induction:
                control_cases.add(("nothing", anchor, control))
            else:
                control_cases.add(("nothing", hyponym, control))

    for nothing, conclusion, stimulus in control_cases:
        pairs.append({"anchor": nothing, "hyponym": conclusion, "stimulus": stimulus})

    utils.write_csv_dict(args.save_path, pairs)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triple_path", type=str, default="data/things/things-sense-based-pairs.csv")
    parser.add_argument("--lemma_path", type=str, default="data/things/things-lemmas-annotated.csv")
    parser.add_argument("--induction", action="store_true")
    parser.add_argument("--qa_format", action="store_true")
    parser.add_argument("--save_path", type=str, default="data/things/things-sense-based-pairs-stimuli.csv")
    args = parser.parse_args()

    main(args)

