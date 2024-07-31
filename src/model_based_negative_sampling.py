import argparse
import csv
import json
import math
import random
import re

import lexicon
import config
import utils
import torch

from collections import defaultdict
from semantic_memory import vsm, vsm_utils
from nltk.corpus import wordnet as wn
from minicons import scorer, cwe
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

def main(args):
    model = args.model
    model_name = model.replace("/", "_")

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
    with open("data/things/things-senses-annotated.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sense"] != "-":
                senses = row['sense'].split("&")
                for sense in senses:
                    things_senses.add(sense)

    triple_path = "data/things/things-triples-actual.csv"
    triples = utils.read_csv_dict(triple_path)

    anchors = set()
    hyponyms = set()
    anchor_children = defaultdict(set)
    concept_universe = set()

    for triple in triples:
        hypernym = triple["hypernym"]
        hyponym = triple["hyponym"]
        anchor = triple["anchor"]

        anchors.add(anchor)
        hyponyms.add(hyponym)
        anchor_children[anchor].add(hyponym)

    prop = lexicon.Property("daxable", "is daxable", "are daxable")

    concept_space = defaultdict(str)
    for c, concept in CONCEPTS.items():
        if concept.generic == "s":
            concept_space[c] = re.split(r'^(a|an)', concept.article)[-1]
        else:
            concept_space[c] = concept.plural

    concept_space = dict(concept_space)

    # prepare batches
    query_pairs = [(v,v) for v in concept_space.values()]

    lm = cwe.CWE(model, "cuda:0")

    layerwise = defaultdict(list)

    query_dl = DataLoader(query_pairs, batch_size = 16)

    for batch in tqdm(query_dl):
        query = list(zip(*batch))

        embs = lm.extract_representation(query, layer = "all")
        for i, emb in enumerate(embs):
            layerwise[i].extend(emb)

    layerwise = {k: torch.stack(v) for k,v in layerwise.items()}

    layerwise_vsms = {k: vsm.VectorSpaceModel(f"LM-layer-{k}") for k,v in layerwise.items()}

    for i, vs_model in layerwise_vsms.items():
        vs_model.load_vectors_from_tensor(layerwise[i], list(concept_space.keys()))

    anchor_neighbors = defaultdict(list)
    for anchor in anchors:
        space = hyponyms - anchor_children[anchor]
        space = [c for c in space if c in CONCEPTS.keys()]

        # i = final layer
        neighbors = layerwise_vsms[i].neighbor(anchor, k=len(space), space=space, names_only=True, ignore_first=False)
        anchor_neighbors[anchor].extend(neighbors[0][: math.ceil(len(anchor_children[anchor])/2)])

        non_neighbors = list(reversed(neighbors[0]))
        anchor_neighbors[anchor].extend(non_neighbors[: math.ceil(len(anchor_children[anchor])/2)])

    anchor_neighbors = dict(anchor_neighbors)
    random.seed(42)
    anchor_neighbors = {k: random.sample(v, len(v)) for k, v in anchor_neighbors.items()}

    negative_sample_triples = []
    for anchor, negative_samples in anchor_neighbors.items():
        for ns in negative_samples:
            negative_sample_triples.append((anchor, ns))

    anchor_concept_sims = defaultdict(list)
    for anchor in anchors:
        space = hyponyms
        space = [c for c in space if c in CONCEPTS.keys()]

        neighbor_sims = layerwise_vsms[i].neighbor(anchor, k = len(space), space=space, ignore_first=False)
        for concept, sim in neighbor_sims[0]:
            anchor_concept_sims[(anchor, concept)].append(sim)

    anchor_concept_sims_csv = []
    for (anchor, concept), sims in anchor_concept_sims.items():
        anchor_concept_sims_csv.append((anchor, concept, max(sims)))

    with open(f"data/things/similarity/things-{model_name}_final_layer.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["concept1", "concept2", "similarity"])
        for row in anchor_concept_sims_csv:
            writer.writerow(row)

    premise_sims = defaultdict(list)

    with open(f"data/things/negative-samples/things-{model_name}_final_layer-ns_triples.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["premise", "conclusion", "similarity"])
        for triple in negative_sample_triples:
            similarity = max(anchor_concept_sims[(triple[0], triple[1])])
            premise_sims[triple[0]].append(similarity)
            writer.writerow(triple + (similarity,))


    taxonomic_pairs = []
    for items in triples:
        premise = items["anchor"]
        conclusion = items["hyponym"]
        similarity = max(anchor_concept_sims[(premise, conclusion)])
        premise_sims[premise].append(similarity)
        taxonomic_pairs.append((premise, conclusion, similarity))


    final_pairs = []

    for item in negative_sample_triples:
        premise, conclusion = item[0], item[1]
        similarity = max(anchor_concept_sims[(premise, conclusion)])

        # similarity-bucket: high if above median of premise_sims[premise], else low
        similarity_bucket = "high" if similarity >= torch.tensor(premise_sims[premise]).median() else "low"
        hypernymy = "no"
        premise_form = CONCEPTS[premise].generic_surface_form()
        conclusion_form = CONCEPTS[conclusion].generic_surface_form()

        final_pairs.append((premise, conclusion, hypernymy, similarity, similarity_bucket, premise_form, conclusion_form))

    for item in taxonomic_pairs:
        premise, conclusion, similarity = item
        similarity_bucket = "high" if similarity >= torch.tensor(premise_sims[premise]).median() else "low"
        hypernymy = "yes"
        premise_form = CONCEPTS[premise].generic_surface_form()
        conclusion_form = CONCEPTS[conclusion].generic_surface_form()

        final_pairs.append((premise, conclusion, hypernymy, similarity, similarity_bucket, premise_form, conclusion_form))

    with open(f"data/things/stimuli-pairs/things-inheritance-{model_name}_final_layer_sim-pairs.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["premise", "conclusion", "hypernymy", "similarity_raw", "similarity_binary", "premise_form", "conclusion_form"])
        for row in final_pairs:
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args)