import argparse
import csv
import json

# import src.config
# from src import lexicon, config, utils
# import src.utils
import lexicon
import config
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


def create_sample(
    parent: lexicon.Concept,
    child: lexicon.Concept,
    prop: lexicon.Property,
    prompt: str,
    control_sentence: str = "this is a sentence.",
    qa=False,
    induction: bool = False,
) -> tuple:
    """Returns a triple with the following format:
    (child property sentence,
    child property sentence given some control sentence,
    child property sentence given parent property sentence embedded in a prompt)

    if induction=True, child and parent positions are reversed
    """
    parent_property = parent.property_sentence(prop)
    child_property = child.property_sentence(prop)

    if induction:
        conclusion = parent_property
        premise = child_property
    else:
        premise = parent_property
        conclusion = child_property

    if not qa:
        control_prompt = prompt.format(control_sentence, "").strip()
        reasoning_prompt = prompt.format(premise, "").strip()
    else:
        control_prompt = prompt.format(control_sentence, conclusion)
        reasoning_prompt = prompt.format(premise, conclusion)

    return (conclusion, control_prompt, reasoning_prompt)

def save_triples(triple_path, lemma_path):
    concepts = defaultdict(lexicon.Concept)
    with open(lemma_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # concepts.append(lemma2concept(row))
            if row["remove"] != "1":
                concepts[row["lemma"]] = lemma2concept(row)

    final_triples = []
    triples = utils.read_csv_dict(triple_path)
    for triple in triples:
        # print concept(hyponym) is a concept(anchor)
        hyponym = triple["hyponym"]
        anchor = triple["anchor"]
        if hyponym in concepts.keys() and anchor in concepts.keys():
                final_triples.append((triple['hypernym'], anchor, hyponym))

    return final_triples



def get_triples(triple_path, lemma_path, induction=False, qa_format=False):
    PROMPT = "Given a premise, produce a conclusion that is true.\nPremise: {}\nConclusion: {}"
    QA_PROMPT = "Given that {}, is it true that {}? Answer with Yes/No:"
    CONTROL = "nothing is daxable"
    # PROMPT = "Given a premise, produce a conclusion that is true.\nPremise: a man was dancing\nConclusion: someone was moving\n\nPremise: {}\nConclusion: {}"
    # QA_PROMPT = "Given that a man was dancing, is it true that someone was moving? Answer with yes/no: Yes\nGiven that {}, is it true that {}? Answer with Yes/No:"

    triples_prompts = []

    # read in concepts
    concepts = defaultdict(lexicon.Concept)
    with open(lemma_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # concepts.append(lemma2concept(row))
            if row["remove"] != "1":
                concepts[row["lemma"]] = lemma2concept(row)

    # construct fake property (can also refer to config.py)
    fake_property = lexicon.Property("daxable", "is daxable", "are daxable")

    # read triples
    triples = utils.read_csv_dict(triple_path)
    for triple in triples:
        # print concept(hyponym) is a concept(anchor)
        hyponym = triple["hyponym"]
        anchor = triple["anchor"]
        if hyponym in concepts.keys() and anchor in concepts.keys():
            child = concepts[hyponym]
            parent = concepts[anchor]
            if qa_format:
                triple = create_sample(
                    parent, child, fake_property, QA_PROMPT, control_sentence=CONTROL, qa=True, induction=induction
                )
            else:
                triple = create_sample(
                    parent, child, fake_property, PROMPT, control_sentence=CONTROL,
                    induction=induction
                )

            triples_prompts.append(triple)
        else:
            print(triple, hyponym in concepts.keys(), anchor in concepts.keys())
    # print(triples_prompts[0])
    return triples_prompts


#    '''
#    TODO: function that returns the property, a control sentence (empty for now), property given some prompt
#    Given a premise, produce a conclusion that is true.
#    premise: {anchor} are daxable.
#    conclusion: {hyponym} are daxable.

#    Yes/No format
#    '''


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
    parser.add_argument(
        "--save",
        action="store_true",
        help="If true, save the triples to a json file",
    )
    args = parser.parse_args()
    triple_prompts = get_triples(args.triple_path, args.lemma_path)
    # print(triple_prompts[:10])
    print(f"{len(triple_prompts)} total instances")
    
    if args.save:
        triples = save_triples(args.triple_path, args.lemma_path)
        with open("data/things/things-triples-actual.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["hypernym", "anchor", "hyponym"])
            writer.writerows(triples)