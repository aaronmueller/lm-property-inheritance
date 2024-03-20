"""
saves:

hypernym, hypernym-sense, anchor, anchor-sense, hyponym
"""

import csv
import json

from collections import defaultdict


things_hypernyms = {}
with open("data/things/THINGS hypernyms - Sheet1.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['bad'] != "1":
            things_hypernyms[row["anchor"]] = row


things = []
with open("data/things/category53_longFormat.tsv", "r") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        things.append(row)


triples = []

for entry in things:
    try:
        anchor = entry['category']
        anchor_lemma = things_hypernyms[anchor]['lemma']
        hyponym = entry['Word']
        hypernym_synset = things_hypernyms[anchor]['hypernym']
        hypernym_lemma = things_hypernyms[anchor]['hypernym_lemma']
        anchor_synset = things_hypernyms[anchor]['synset']
    except:
        pass

    triples.append((hypernym_lemma, hypernym_synset, anchor_lemma, anchor_synset, hyponym))

    # print(f"{hypernym_lemma} > {anchor_lemma} > {hyponym}")

# write to csv
with open("data/things/things-triples.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["hypernym", "hypernym-sense", "anchor", "anchor-sense", "hyponym"])
    for triple in triples:
        writer.writerow(triple)