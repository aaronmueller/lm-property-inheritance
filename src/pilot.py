import argparse
import config
import csv
import json
import lexicon
import utils

from collections import defaultdict
from prompt import Prompt


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
            final_triples.append((triple["hypernym"], anchor, hyponym))

    return final_triples


def get_triples(triple_path, lemma_path, induction=False, qa_format=False):
    """NOW DEPRECATED."""
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
                    parent,
                    child,
                    fake_property,
                    QA_PROMPT,
                    control_sentence=CONTROL,
                    qa=True,
                    induction=induction,
                )
            else:
                triple = create_sample(
                    parent,
                    child,
                    fake_property,
                    PROMPT,
                    control_sentence=CONTROL,
                    induction=induction,
                )

            triples_prompts.append(triple)
        else:
            print(triple, hyponym in concepts.keys(), anchor in concepts.keys())
    print(triples_prompts[0])
    return triples_prompts


def generate_stimuli(
    triple_path,
    lemma_path,
    prompt_cfg,
    induction=False,
    tokenizer=None,
):

    stimuli = []

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
    prompt = Prompt(prompt_cfg["template"], prompt_cfg["zero_shot"])

    # read triples
    triples = utils.read_csv_dict(triple_path)
    for triple in triples:
        try:
            hyponym = triple["hyponym"]
            anchor = triple["anchor"]
        except:
            hyponym = triple["premise"]
            anchor = triple["conclusion"]
        if hyponym in concepts.keys() and anchor in concepts.keys():
            child = concepts[hyponym]
            parent = concepts[anchor]

            if induction:
                premise = child
                conclusion = parent
            else:
                premise = parent
                conclusion = child

            if tokenizer is not None:
                stimulus_instance = prompt.create_sample_tokenized(
                    premise, conclusion, fake_property, tokenizer=tokenizer
                )
            else:
                stimulus_instance = prompt.create_sample(
                    premise, conclusion, fake_property
                )

            stimuli.append(stimulus_instance)
        else:
            pass
    return stimuli


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
        help="If true, save the triples to a csv file",
    )
    args = parser.parse_args()
    triple_prompts = generate_stimuli(args.triple_path, args.lemma_path, config.PROMPTS['initial-phrasal'])
    # print(triple_prompts[:10])
    print(f"{len(triple_prompts)} total instances")

    if args.save:
        triples = save_triples(args.triple_path, args.lemma_path)
        with open("data/things/things-triples-actual.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["hypernym", "anchor", "hyponym"])
            writer.writerows(triples)
