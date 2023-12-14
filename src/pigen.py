"""Generate property inheritance stimuli"""

import pathlib
import random
import json
import inflect

import config

from collections import defaultdict
from semantic_memory import memory, list_utils
from ordered_set import OrderedSet

engine = inflect.engine()

MEM_PATH = "data/semantic-memory"

pathlib.Path("data/stimuli/").mkdir(exist_ok=True, parents=True)

# load semantic memory
mem = memory.Memory(
    concept_path=f"{MEM_PATH}/concept_senses.csv",
    feature_path=f"{MEM_PATH}/xcslb_compressed.csv",
    matrix_path=f"{MEM_PATH}/concept_matrix.txt",
    feature_metadata=f"{MEM_PATH}/feature_lexicon.csv",
)

mem.create()

# sample 8 properties that belong to animals
animals = mem.taxonomy["animal.n.01"].descendants()

properties = []
for animal in animals:
    properties.extend(sorted(list(mem.concept_features[animal])))


animal_property_space = defaultdict(dict)
for animal in animals:
    features = mem.concept_features[animal]
    animal_property_space[animal] = {
        "yes": sorted(list(features)),
        "no": sorted(list(OrderedSet(properties).difference(features))),
    }

# sample 8 animal concepts for prompt
random.seed(1234)

icl_concepts = random.sample(animals, 8)
icl_labels = ["yes"] * 4 + ["no"] * 4
icl_data = []
icl_properties = []

# shuffle icl labels
random.shuffle(icl_labels)

for i, (concept, label) in enumerate(zip(icl_concepts, icl_labels)):
    # sample_space = animal_property_space[concept][label]
    sample_space = [
        p
        for p in animal_property_space[concept][label]
        if "color" not in p and "good" not in p and "tall" not in p and "short" not in p
    ]
    sampled_property = random.sample(sample_space, 1)[0]

    generic_concept = engine.plural_noun(concept.replace("_", " "))
    generic_property = mem.feature_lexicon[sampled_property].pluralized
    generic = f"{generic_concept} {generic_property}"
    if label == "yes":
        implicit_knowledge = f"{mem.lexicon[concept].article} {sampled_property}"
    else:
        implicit_knowledge = f"{mem.lexicon[concept].article} {mem.feature_lexicon[sampled_property].negation}"

    icl_properties.append(sampled_property)
    icl_data.append(
        {
            "idx": i,
            "concept": concept,
            "property": sampled_property,
            "implicit_knowledge": implicit_knowledge,
            # "implicit_knowledge": generic, TODO: add after figuring out negated cases for generics
            "label": label,
        }
    )

random.shuffle(icl_data)
# test data

test_concepts = random.sample(animals, 128)
test_labels = ["yes"] * 64 + ["no"] * 64
test_data = []

for i, (concept, label) in enumerate(zip(test_concepts, test_labels)):
    sample_space = [
        p
        for p in animal_property_space[concept][label]
        if p not in icl_properties
        and "color" not in p
        and "good" not in p
        and "tall" not in p
        and "short" not in p
        and "heavy" not in p
        and "eat" not in p
        and "found" not in p
        and "attracted" not in p
        and "like" not in p
        and "lives" not in p
        and "large" not in p
        and "friend" not in p
        and "kill" not in p
        and "appetite" not in p
        and "loyal" not in p
        and "playful" not in p
    ]
    sampled_property = random.sample(sample_space, 1)[0]

    generic_concept = engine.plural_noun(concept.replace("_", " "))
    generic_property = mem.feature_lexicon[sampled_property].pluralized
    generic = f"{generic_concept} {generic_property}"
    if label == "yes":
        implicit_knowledge = f"{mem.lexicon[concept].article} {sampled_property}"
    else:
        implicit_knowledge = f"{mem.lexicon[concept].article} {mem.feature_lexicon[sampled_property].negation}"

    test_data.append(
        {
            "idx": i,
            "concept": concept,
            "property": sampled_property,
            "implicit_knowledge": implicit_knowledge,
            # "implicit_knowledge": generic, TODO: add after figuring out negated cases for generics
            "label": label,
        }
    )

# save to data/stimuli/prompt.jsonl and data/stimuli/test.jsonl
with open("data/stimuli/prompt_metadata.jsonl", "w") as f:
    for entry in icl_data:
        json.dump(entry, f)
        f.write("\n")

with open("data/stimuli/test_metadata.jsonl", "w") as f:
    for entry in test_data:
        json.dump(entry, f)
        f.write("\n")

nonce_words = config.NONCE_WORDS
# icl nonce words
icl_nonce_words = random.sample(nonce_words, 8)
test_nonce_word = [w for w in nonce_words if w not in icl_nonce_words][0]


def _create_question(entry, nw, is_real=True, is_test=False):
    answer = entry["label"]
    if is_test:
        answer = ""
    if is_real:
        return f"Question: Is it true that {mem.lexicon[entry['concept']].article} {entry['property']}? Answer: {answer}".strip()
    else:
        return f"A {nw} is a type of {entry['concept'].replace('_', ' ')}. Question: Is it true that a {nw} {entry['property']}? Answer: {answer}".strip()


icl_strings_real = []
icl_strings_nw = []
for nw, entry in zip(icl_nonce_words, icl_data):
    real_question = _create_question(entry, nw, is_real=True)
    nw_question = _create_question(entry, nw, is_real=False)

    icl_strings_real.append(real_question)
    icl_strings_nw.append(nw_question)

# write real_prompt.txt and nw_prompt.txt
with open("data/stimuli/real_prompt.txt", "w") as f:
    for line in icl_strings_real:
        f.write(line + "\n")

with open("data/stimuli/pi_prompt.txt", "w") as f:
    for line in icl_strings_nw:
        f.write(line + "\n")

# generate test data for real and nw
test_nw = []
test_real = []
for entry in test_data:
    real_question = _create_question(entry, test_nonce_word, is_real=True, is_test=True)
    nw_question = _create_question(entry, test_nonce_word, is_real=False, is_test=True)

    real_entry = {k: v for k, v in entry.items()}
    nw_entry = {k: v for k, v in entry.items()}
    real_entry.update({"question": real_question})
    nw_entry.update({"question": nw_question})

    test_nw.append(nw_entry)
    test_real.append(real_entry)

# write test_real.jsonl and test_nw.jsonl

with open("data/stimuli/test_real.jsonl", "w") as f:
    for entry in test_real:
        json.dump(entry, f)
        f.write("\n")

with open("data/stimuli/test_pi.jsonl", "w") as f:
    for entry in test_nw:
        json.dump(entry, f)
        f.write("\n")
