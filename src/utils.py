import csv
import json
import lexicon


def read_tsv_dict(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def read_csv_dict(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def write_csv_dict(path, data):
    with open(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

def read_things(path):
    things = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for line in reader:
            if line['remove'] != '1':
                things.append(line)
    return things


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
