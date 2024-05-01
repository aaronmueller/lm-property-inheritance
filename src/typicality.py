import argparse
import csv
import json

import lexicon
import utils

from collections import defaultdict


def lemma2concept(entry):
    return lexicon.Concept(
        lemma=entry["lemma"],
        singular=entry["singular"],
        plural=entry["plural"],
        article=entry["article"],
        generic=entry["generic"],
        taxonomic_phrase=entry["taxonomic_phrase"],
    )


def main(args):
    triple_path = args.triple_path
    lemma_path = args.lemma_path

    concepts = defaultdict(lexicon.Concept)
    with open(lemma_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # concepts.append(lemma2concept(row))
            if row["remove"] != "1":
                concepts[row["lemma"]] = lemma2concept(row)

    tsv = []
    triples = utils.read_csv_dict(triple_path)
    for triple in triples:
        anchor = triple["anchor"]
        hyponym = triple["hyponym"]

        if hyponym in concepts.keys() and anchor in concepts.keys():
            child = concepts[hyponym]
            parent = concepts[anchor]

            # taxonomic phrase generation
            sentence = child.is_a(parent)
            tsv.append(anchor, hyponym, sentence)

    
        

    '''
    TODO: function that returns the property, a control sentence (empty for now), property given some prompt
    Given a premise, produce a conclusion that is true.
    premise: {anchor} are daxable.
    conclusion: {hyponym} are daxable.

    Yes/No format
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--triple_path",
        type=str,
        default="data/things/things-triples.csv",
        help="path to the triples csv",
    )
    parser.add_argument(
        "--lemma_path",
        type=str,
        default="data/things/things-lemmas-annotated.csv",
        help="path to the lemma csv",
    )
    args = parser.parse_args()
    main(args)
