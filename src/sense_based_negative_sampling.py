import argparse
import csv
import json
import random

import lexicon
import config
import utils
import torch
import math

from collections import defaultdict
from semantic_memory import taxonomy, vsm, vsm_utils
from nltk.corpus import wordnet as wn
from ordered_set import OrderedSet

VECTORS="../lmms-sp-wsd.albert-xxlarge-v2.synsets.vectors.txt"

sense_embeddings = vsm.VectorSpaceModel("LMMS-ALBERT")
# takes about a min to load...
sense_embeddings.load_vectors(VECTORS)

def lemma2concept(entry):
    return lexicon.Concept(
        lemma=entry["lemma"],
        singular=entry["singular"],
        plural=entry["plural"],
        article=entry["article"],
        generic=entry["generic"],
        taxonomic_phrase=entry["taxonomic_phrase"],
    )

lemma_path = "data/things/things-lemmas-annotated.csv"

# read in concepts
CONCEPTS = defaultdict(lexicon.Concept)
with open(lemma_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["remove"] != "1":
            CONCEPTS[row["lemma"]] = lemma2concept(row)
CONCEPTS = dict(CONCEPTS)

things_senses = set()
concepts_annotated_senses = defaultdict(set)
with open("data/things/things-senses-annotated.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["sense"] != "-":
            senses = row['sense'].split("&")
            for sense in senses:
                things_senses.add(sense)
                concepts_annotated_senses[row["concept"]].add(sense)

concepts_annotated_senses = dict(concepts_annotated_senses)

# for concept, senses in concepts_annotated_senses.items():
#     for s in senses:
#         try:
#             sense_embeddings(s)
#         except:
#             print(concept, s)

triple_path = "data/things/things-triples-actual.csv"
triples = utils.read_csv_dict(triple_path)

concept_universe = set()
anchor_synsets = defaultdict(str)
anchor_children = defaultdict(set)
synset_anchors = defaultdict(set)
nonsense = set()
hypernyms = defaultdict(str)
for triple in triples:
    hypernym = triple["hypernym"]
    hyponym = triple["hyponym"]
    anchor = triple["anchor"]
    if triple["anchor-sense"] == "-":
        nonsense.add(anchor)
        if anchor in [
            "breakfast",
            "office supply",
            "school supply",
            "women's clothing",
        ]:
            if anchor == "breakfast":
                anchor_sense = "breakfast.n.00"
            elif anchor == "office supply":
                anchor_sense = "office_supply.n.00"
            elif anchor == "school supply":
                anchor_sense = "school_supply.n.00"
            elif anchor == "women's clothing":
                anchor_sense = "womens_clothing.n.00"
        anchor_synsets[anchor] = anchor_sense
        synset_anchors[anchor_sense].add(anchor)
        hypernyms[hyponym] = anchor
    else:
        anchor_sense = triple["anchor-sense"]
        if anchor == "headwear":
            # print(triple)
            anchor_sense = "headdress.n.01"
        elif anchor == "toy":
            anchor_sense = "plaything.n.01"
        elif anchor == "protective clothing":
            anchor_sense = "protective_covering.n.01"
        elif anchor == "breakfast":
            anchor_sense = "breakfast.n.00"
        elif anchor == "office supply":
            anchor_sense = "office_supply.n.00"
        elif anchor == "school supply":
            anchor_sense = "school_supply.n.00"
        elif anchor == "women's clothing":
            anchor_sense = "womens_clothing.n.00"
        if anchor_sense == "none":
            print(triple)
        anchor_synsets[anchor] = anchor_sense
        synset_anchors[anchor_sense].add(anchor)
        hypernyms[hyponym] = anchor
    # concept_universe.add(hypernym)
    concept_universe.add(hyponym)
    # concept_universe.add(anchor)
    anchor_children[anchor].add(hyponym)

anchor_children = dict(anchor_children)
anchor_synsets = dict(anchor_synsets)
synset_anchors = dict(synset_anchors)
hypernyms = dict(hypernyms)

NONSENSES = {
    "breakfast.n.00": [
        "bacon.n.01",
        "bagel.n.01",
        "bread.n.01",
        "breakfast.n.01",
        "breakfast_food.n.01",
        "cereal.n.03",
        "cream_cheese.n.01",
        "crepe.n.01",
        "crescent_roll.n.01",
        "doughnut.n.02",
        "egg.n.02",
        "french_fries.n.01",
        "granola.n.01",
        "grits.n.01",
        "ham.n.01",
        "hamburger.n.01",
        "hash.n.01",
        "honey.n.01",
        "jam.n.01",
        "maple_syrup.n.01",
        "marmalade.n.01",
        "muffin.n.01",
        "oatmeal.n.01",
        "omelet.n.01",
        "pancake.n.01",
        "pastry.n.02",
        "peanut_butter.n.01",
        "sandwich.n.01",
        "sausage.n.01",
        "scone.n.01",
        "scrambled_eggs.n.01",
        "syrup.n.01",
        "toast.n.01",
        "waffle.n.01",
        "yogurt.n.01",
    ],
    "office_supply.n.00": [
        "binder.n.03",
        "clipboard.n.01",
        "envelope.n.01",
        "eraser.n.01",
        "file.n.01",
        "booklet.n.01",
        "fountain_pen.n.01",
        "glue.n.01",
        "highlighter.n.01",
        "ink.n.01",
        "notebook.n.01",
        "notepad.n.01",
        "paper.n.01",
        "paper_clip.n.01",
        "pen.n.01",
        "pencil.n.01",
        "pencil_sharpener.n.01",
        "punch.n.03",
        "rubber_band.n.01",
        "scissors.n.01",
        "staple.n.05",
        "stapler.n.01",
        "tack.n.02",
        "tape.n.01",
        "thumbtack.n.01",
    ],
    "school_supply.n.00": [
        "backpack.n.01",
        "binder.n.03",
        "book.n.01",
        "calculator.n.02",
        "chalk.n.04",
        "blackboard.n.01",
        "crayon.n.01",
        "eraser.n.01",
        "booklet.n.01",
        "fountain_pen.n.01",
        "glue.n.01",
        "highlighter.n.01",
        "ink.n.01",
        "notebook.n.01",
        "inkwell.n.01",
        "marker.n.03",
        "notepad.n.01",
        "paper.n.01",
        "paper_clip.n.01",
        "pen.n.01",
        "pencil.n.01",
        "pencil_sharpener.n.01",
        "scissors.n.01",
        "staple.n.05",
        "stapler.n.01",
    ],
    "womens_clothing.n.00": [
        "bikini.n.02",
        "blouse.n.01",
        "feather_boa.n.01",
        "brassiere.n.01",
        "corset.n.01",
        "dress.n.01",
        "nylons.n.01",
        "garter.n.01",
        "headscarf.n.01",
        "kimono.n.01",
        "legging.n.01",
        "lingerie.n.01",
        "pants_suit.n.01",
        "pantyhose.n.01",
        "skirt.n.02",
        "stocking.n.01",
        "tiara.n.01",
        "head_covering.n.01",
    ],
}

def synset_names(concept, return_map=False, both=False):
    # synset_list = [synset.name() for synset in wn.synsets(concept, "n")]
    synset_list = [s for s in concepts_annotated_senses[concept]]
    if return_map:
        synset_map = {s: concept for s in synset_list}
        # synset_map = 
        concept_map = {concept: s for s in synset_list}
        return synset_map, concept_map
    elif both:
        return (
            synset_list,
            {s: concept for s in synset_list},
            {concept: s for s in synset_list},
        )
    else:
        return synset_list

synset_map = {}
concept_map = {}
concept_synsets = defaultdict(set)
for concept in concept_universe:
    lst, smap, cmap = synset_names(concept, both=True)
    synset_map.update(smap)
    concept_map.update(cmap)
    concept_synsets[concept].update(set(lst))
    # synset_map.update(synset_names(concept, return_map=True))

concept_synsets = dict(concept_synsets)

def is_hypernym(synset, target):
    try:
        target = wn.synset(target)
        for path in wn.synset(synset).hypernym_paths():
            if target in path:
                return True
    except:
        return False
    return False

all_synsets = set()
for k,v in concept_synsets.items():
    for vv in v:
        if vv in things_senses:
            all_synsets.add(vv)

synset_universe = set(list(all_synsets) + list(anchor_synsets.values()))
synset_universe = [s for s in synset_universe if s not in NONSENSES.keys()]

print(f"Total Synsets: {len(synset_universe)}")

universe_vectors = sense_embeddings(synset_universe)

excess = []
excess_vocab = []
for k, v in NONSENSES.items():
    excess.append(sense_embeddings(v).mean(0))
    excess_vocab.append(k)

reduced_sense_embeddings = vsm.VectorSpaceModel("Reduced Sense Embs")
reduced_sense_embeddings.load_vectors_from_tensor(
    torch.cat((universe_vectors, torch.stack(excess))), synset_universe + excess_vocab
)

reduced_sense_embeddings

anchor_neighbors = defaultdict(list)
for anchor, anchor_synset in anchor_synsets.items():
    # leftover = set()
    # anchor = synset_anchors[anchor_synset]
    space = concept_universe - anchor_children[anchor]
    space_synsets = set()
    for c in space:
        if c in CONCEPTS.keys():
            for cs in concept_synsets[c]:
                if cs not in anchor_synsets.values() and cs in synset_universe:
                    space_synsets.add(cs)

    space_synsets = list(space_synsets)
    # filter out hypernymy cases
    space_synsets = [s for s in space_synsets if is_hypernym(s, anchor_synset) == False]

    neighbors = reduced_sense_embeddings.neighbor(
        anchor_synset,
        space=space_synsets,
        k=len(anchor_children[anchor]),
        names_only=True,
        ignore_first=False,
    )[0]
    neighbor_concepts = [synset_map[n] for n in neighbors]
    neighbor_concepts = list(OrderedSet(neighbor_concepts))[
        : math.ceil(len(anchor_children[anchor]) / 2)
    ]
    anchor_neighbors[anchor].extend(neighbor_concepts)

    # non-neighbors
    non_neighbors = reduced_sense_embeddings.neighbor(
        anchor_synset,
        space=space_synsets,
        k=len(anchor_children[anchor]),
        names_only=True,
        ignore_first=False,
        nearest=False,
    )[0]
    non_neighbor_concepts = [synset_map[n] for n in non_neighbors]
    non_neighbor_concepts = list(OrderedSet(non_neighbor_concepts))[
        : math.ceil(len(anchor_children[anchor]) / 2)
    ]
    anchor_neighbors[anchor].extend(non_neighbor_concepts)

# print({k: len(v) for k, v in anchor_children.items()})

anchor_neighbors = dict(anchor_neighbors)
random.seed(42)
anchor_neighbors = {k: random.sample(v, len(v)) for k, v in anchor_neighbors.items()}

# print({k: len(v) for k, v in anchor_neighbors.items()})

negative_sample_triples = []
for anchor, negative_samples in anchor_neighbors.items():
    for ns in negative_samples:
        negative_sample_triples.append((anchor, anchor_synsets[anchor], ns))

real_synset_map = defaultdict(list)
for c, senses in concepts_annotated_senses.items():
    for s in senses:
        real_synset_map[s].append(c)

real_synset_map = dict(real_synset_map)

# save anchor, every concept similarities
anchor_concept_sims = defaultdict(list)
for anchor, anchor_synset in anchor_synsets.items():
    neighbor_sims = reduced_sense_embeddings.neighbor(anchor_synset, k = len(reduced_sense_embeddings.vocab), ignore_first=False)
    for concept_sense, sim in neighbor_sims[0]:
        try:
            concepts = real_synset_map[concept_sense]
        except:
            concepts = synset_anchors[concept_sense]
        for c in concepts:
            anchor_concept_sims[(anchor, c)].append(sim)

anchor_concept_sims_csv = []
for (anchor, concept), sims in anchor_concept_sims.items():
    anchor_concept_sims_csv.append((anchor, concept, max(sims)))

space = concept_universe
space_synsets = set()
for c in space:
    # if c in concepts.keys():
    for cs in concept_synsets[c]:
            # if cs not in anchor_synsets.values() and cs in synset_universe:
        space_synsets.add(cs)

with open("data/things/similarity/things-sense_based.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["concept1", "concept2", "similarity"])
    for row in anchor_concept_sims_csv:
        writer.writerow(row)

premise_sims = defaultdict(list)

with open("data/things/negative-samples/things-sense_based-ns_triples.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["premise", "premise-sense", "conclusion", "similarity"])
    for triple in negative_sample_triples:
        similarity = max(anchor_concept_sims[(triple[0], triple[2])])
        premise_sims[triple[0]].append(similarity)
        writer.writerow(triple + (similarity,))

## save along with taxonomic triples (from triples)


triples = utils.read_csv_dict("data/things/things-triples-actual.csv")

taxonomic_pairs = set()
for items in triples:
    premise = items["anchor"]
    conclusion = items["hyponym"]
    similarity = max(anchor_concept_sims[(premise, conclusion)])
    premise_sims[premise].append(similarity)
    taxonomic_pairs.add((premise, conclusion, similarity))


final_pairs = set()

for item in negative_sample_triples:
    premise, conclusion = item[0], item[2]
    similarity = max(anchor_concept_sims[(premise, conclusion)])

    # similarity-bucket: high if above median of premise_sims[premise], else low
    similarity_bucket = "high" if similarity >= torch.tensor(premise_sims[premise]).median() else "low"
    hypernymy = "no"
    premise_form = CONCEPTS[premise].generic_surface_form()
    conclusion_form = CONCEPTS[conclusion].generic_surface_form()

    final_pairs.add((premise, conclusion, hypernymy, similarity, similarity_bucket, premise_form, conclusion_form))

for item in taxonomic_pairs:
    premise, conclusion, similarity = item
    similarity_bucket = "high" if similarity >= torch.tensor(premise_sims[premise]).median() else "low"
    hypernymy = "yes"
    premise_form = CONCEPTS[premise].generic_surface_form()
    conclusion_form = CONCEPTS[conclusion].generic_surface_form()

    final_pairs.add((premise, conclusion, hypernymy, similarity, similarity_bucket, premise_form, conclusion_form))

with open("data/things/stimuli-pairs/things-inheritance-sense_based_sim-pairs.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["premise", "conclusion", "hypernymy", "similarity_raw", "similarity_binary", "premise_form", "conclusion_form"])
    for row in final_pairs:
        writer.writerow(row)