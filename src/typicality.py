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
            if row['remove'] != '1':
                concepts[row['lemma']] = lemma2concept(row)

    fake_property = lexicon.Property("daxable", "is daxable", "are daxable")

    # read triples
    triples = utils.read_csv_dict(triple_path)
    for triple in triples:
        # print concept(hyponym) is a concept(anchor)
        hyponym = triple["hyponym"]
        anchor = triple["anchor"]
        try:
            # print(concepts[hyponym].is_a(concepts[anchor]))
            # print(concepts[hyponym].property_sentence(fake_property))
            anchor_property = concepts[anchor].property_sentence(fake_property)
            hyponym_property = concepts[hyponym].property_sentence(fake_property)
            # print(f"Given that {anchor_property}{hyponym_property}")
            '''
            Given a premise, produce a conclusion that is true.
            premise: birds are daxable.
            conclusion: robins 
            [p(robins are daxable.) - p(robins are daxable.|left)]/p(robins are daxable.)
            '''
        except:
            pass
        

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
