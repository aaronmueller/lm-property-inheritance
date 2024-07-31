"""Code to generate sentences for taxonomic sentence verification"""

import argparse
import csv
import json

import lexicon
import config
import pathlib
import utils

from collections import defaultdict
from ordered_set import OrderedSet


def main(args):
    
    pathlib.Path("data/things/tsv").mkdir(parents=True, exist_ok=True)
    pathlib.Path("data/things/tsv/stimuli").mkdir(parents=True, exist_ok=True)

    qa = args.qa #boolean
    declarative = args.declarative #boolean

    things_concepts_raw = utils.read_things("data/things/things-lemmas-annotated.csv")
    things_concepts = {x["lemma"]: x for x in things_concepts_raw}

    things_triples = utils.read_csv_dict("data/things/things-triples-actual.csv")

    # concept-hypernym pairs
    concept_hypernyms = defaultdict(lambda: False)
    for triple in things_triples:
        concept_hypernyms[(triple["hyponym"], triple["anchor"])] = True

    # unique hypernym categories
    hypernyms = OrderedSet([x["anchor"] for x in things_triples])
    hyponyms = OrderedSet([x["hyponym"] for x in things_triples])

    # print(len(concept_hypernyms), len(hypernyms), len(things_concepts))

    tsv = []
    for hypernym in hypernyms:
        for hyponym in hyponyms:
            hyponym_concept = utils.lemma2concept(things_concepts[hyponym])
            hypernym_concept = utils.lemma2concept(things_concepts[hypernym])

            if not qa:
                prefix, stimulus = hyponym_concept.is_a(hypernym_concept, split=True)
                hypernymy = concept_hypernyms[(hyponym, hypernym)]
                tsv.append(
                    {
                        "hyponym": hyponym,
                        "hypernym": hypernym,
                        "prefix": prefix,
                        "stimulus": stimulus,
                        "hypernymy": hypernymy,
                    }
                )
            else:
                question = hyponym_concept.inquisitive_is_a(hypernym_concept, declarative)
                hypernymy = concept_hypernyms[(hyponym, hypernym)]
                tsv.append(
                    {
                        "hyponym": hyponym,
                        "hypernym": hypernym,
                        "question": question,
                        "hypernymy": hypernymy
                    }
                )

    print(f"Total stimuli: {len(tsv)}, Sneak peek:")
    for i in range(5):
        print(tsv[i])

    # outfile
    if qa:
        if declarative:
            utils.write_csv_dict("data/things/tsv/stimuli/things-tsv-qa-declarative-stimuli.csv", tsv)
        else:
            utils.write_csv_dict("data/things/tsv/stimuli/things-tsv-qa-stimuli.csv", tsv)
    else:
        utils.write_csv_dict("data/things/tsv/stimuli/things-tsv-stimuli.csv", tsv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", action="store_true")
    parser.add_argument("--declarative", action="store_true")
    args = parser.parse_args()

    main(args)
