import config
import math
import re
import torch
import utils
import random
import csv
import lexicon

from semantic_memory import vsm
from tqdm import trange
from collections import defaultdict

random.seed(42)

embeddings = []
with open("data/things/spose_embedding_66d_sorted.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        vector = [float(x) for x in line.split()]
        embeddings.append(vector)

embeddings = torch.tensor(embeddings)

triples = utils.read_csv_dict("data/things/things-triples-actual.csv")
things_categories = set([t['anchor'] for t in triples])
things_hyponyms = set([t['hyponym'] for t in triples])

raw_categories = utils.read_tsv_dict("data/things/category53_longFormat.tsv")
vocab_words = defaultdict(str)
for entry in raw_categories:
    if entry["Word"] in things_hyponyms or entry["Word"] in things_categories:
        vocab_words[entry["uniqueID"]] = entry["Word"]
    else:
        continue
vocab_words = dict(vocab_words)

vocab_raw = open("data/things/unique_id.txt", "r").readlines()
vocab_raw = [x.strip() for x in vocab_raw]

vocab = []
for word in vocab_raw:
    try:
        vocab.append(vocab_words[word.strip()])
    except KeyError:
        continue

things = vsm.VectorSpaceModel("THINGS 66d")
things.load_vectors_from_tensor(embeddings, vocab_raw)

things_concepts = [t['concept'] for t in utils.read_csv_dict("data/things/things-senses-annotated.csv")]

# if concept contains a number, store it
numbered_concepts = set([re.split(r'\d', v)[0] for v in vocab_raw if re.search(r'\d', v)])

anchor_children = defaultdict(set)
for t in triples:
    anchor_children[t['anchor']].add(t['hyponym'])

anchor_children = dict(anchor_children)

categories = defaultdict(list)
for entry in raw_categories:
    cat = entry["category"]
    if cat in things_categories:
        categories[cat].append(entry["uniqueID"])
    elif cat in config.CATEGORY_REPLACEMENTS.keys():
        categories[config.CATEGORY_REPLACEMENTS[cat]].append(entry["uniqueID"])
    else:
        continue

categories = dict(categories)


things_prototype_tensor = []
new_vocab = []

for uid, word in vocab_words.items():
    things_prototype_tensor.append(things(uid).squeeze(0))
    new_vocab.append(uid)

for cat, objects in categories.items():
    if cat not in vocab_raw:
        prototype = things(objects).mean(0)
        things_prototype_tensor.append(prototype)
        new_vocab.append(cat)
        vocab_words[cat] = cat

things_prototypes = vsm.VectorSpaceModel("THINGS 66d Prototypes")
things_prototypes.load_vectors_from_tensor(torch.stack(things_prototype_tensor), new_vocab)

def embedding2sim(embedding, chunk_size=1000):
    n_objects = embedding.shape[0]
    
    # Compute similarity matrix
    sim = torch.matmul(embedding, embedding.t())
    esim = torch.exp(sim)
    
    cp = torch.zeros(n_objects, n_objects)
    
    for i in range(0, n_objects, chunk_size):
        i_end = min(i + chunk_size, n_objects)
        i_indices = torch.arange(i, i_end)
        
        for j in range(i + 1, n_objects, chunk_size):
            j_end = min(j + chunk_size, n_objects)
            j_indices = torch.arange(j, j_end)
            
            esim_ij = esim[i_indices][:, j_indices].unsqueeze(2)
            esim_ik = esim[i_indices].unsqueeze(1)
            esim_jk = esim[j_indices].unsqueeze(0)
            
            # Compute ctmp for all valid k
            ctmp = esim_ij / (esim_ij + esim_ik + esim_jk)
            
            # Create mask to exclude k == i and k == j
            mask = torch.ones(i_end - i, j_end - j, n_objects, dtype=torch.bool)
            for idx, ii in enumerate(range(i, i_end)):
                mask[idx, :, ii] = False
            for idx, jj in enumerate(range(j, j_end)):
                mask[:, idx, jj] = False
            
            ctmp = ctmp.masked_fill(~mask, 0)
            
            # Sum and normalize
            cp_chunk = ctmp.sum(dim=2) / (n_objects - 2)
            cp[i_indices[:, None], j_indices] = cp_chunk
    
    # Enforce symmetry without adding the matrix to itself
    cp = torch.triu(cp)  # Keep the upper triangle
    cp = cp + cp.t() - torch.diag(cp.diagonal())  # Mirror the upper triangle to the lower

    # Debugging: Check cp after making symmetric
    if torch.any(cp > 1):
        print(f"Warning: cp contains values greater than 1 after symmetry! There are {(cp > 1).sum().item()} such values.")

    cp.diagonal().fill_(1)
    
    # Symmetry check: Compare upper and lower triangular parts
    upper_triangle = torch.triu(cp, diagonal=1)
    lower_triangle = torch.tril(cp, diagonal=-1).t()  # Transpose lower triangle to compare with upper
    
    if not torch.allclose(upper_triangle, lower_triangle):
        print("Error: The matrix is not symmetric!")
    
    return cp

spose_sim_prototypes = embedding2sim(things_prototypes.embeddings, chunk_size=100)

class PairwiseSim:
    def __init__(self, sim_matrix, vocab2idx):
        self.sim_matrix = sim_matrix
        self.vocab2idx = vocab2idx
        self.vocab = list(self.vocab2idx.keys())

    def __call__(self, word1, word2):
        idx1 = self.vocab2idx[word1]
        idx2 = self.vocab2idx[word2]
        return self.sim_matrix[idx1, idx2]
    
    def neighbors(self, word, k=10, space=None):
        # space = list of values to consider before returning neighbors, specificed by the user.
        idx = self.vocab2idx[word]
        neighbors = self.sim_matrix[idx].argsort(descending=True)
        if space:
            return [(self.vocab[n], self.sim_matrix[idx, n].item()) for n in neighbors[:k] if self.vocab[n] in space]
        else:
            return [(self.vocab[n], self.sim_matrix[idx, n].item()) for n in neighbors[:k]]
        
spose_prototypes = PairwiseSim(spose_sim_prototypes, things_prototypes.vocab2idx)

print({k: len(set(v)) for k, v in anchor_children.items()})

space = set(vocab_words.keys()) - set(categories.keys())
space = set([c for c in space if vocab_words[c] in things_concepts])

anchor_neighbors = defaultdict(list)
for anchor, children in anchor_children.items():
    # space = hyponyms - anchor_children[anchor]
    # space = [c for c in space if c in CONCEPTS.keys()]
    sample_space = list(space - children)

    samples = len(anchor_children[anchor])

    # if not even, allocate samples/2 to each half
    if samples % 2 != 0:
        neighbor_half = math.ceil(samples / 2)
        non_neighbor_half = math.floor(samples / 2)
    else:
        neighbor_half = samples // 2
        non_neighbor_half = samples // 2

    assert neighbor_half + non_neighbor_half == samples

    neighbors = spose_prototypes.neighbors(anchor, k=len(sample_space), space=sample_space)
    anchor_neighbors[anchor].extend([c for c, sim in neighbors[:neighbor_half]])

    non_neighbors = list(reversed(neighbors))
    anchor_neighbors[anchor].extend([c for c, sim in non_neighbors[:non_neighbor_half]])

anchor_neighbors = dict(anchor_neighbors)
anchor_neighbors = {k: random.sample(v, len(v)) for k, v in anchor_neighbors.items()}

negative_sample_triples = []
for anchor, negative_samples in anchor_neighbors.items():
    for ns in negative_samples:
        negative_sample_triples.append((anchor, vocab_words[ns]))

anchor_concept_sims = defaultdict(list)
for anchor, hyponyms in anchor_children.items():
    space = set(vocab_words.keys())
    neighbor_sims = spose_prototypes.neighbors(anchor, k = len(space), space=space)
    for concept, sim in neighbor_sims:
        anchor_concept_sims[(anchor, vocab_words[concept])].append(sim)
anchor_concept_sims = dict(anchor_concept_sims)

anchor_concept_sims_csv = []
for (anchor, concept), sims in anchor_concept_sims.items():
    anchor_concept_sims_csv.append((anchor, concept, max(sims)))

with open(f"data/things/similarity/things-SPOSE_prototype.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["concept1", "concept2", "similarity"])
    for row in anchor_concept_sims_csv:
        writer.writerow(row)

premise_sims = defaultdict(list)

with open(f"data/things/negative-samples/things-SPOSE_prototype-ns_triples.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["premise", "conclusion", "similarity"])
    for triple in negative_sample_triples:
        similarity = max(anchor_concept_sims[(triple[0], triple[1])])
        premise_sims[triple[0]].append(similarity)
        writer.writerow(triple + (similarity,))

taxonomic_pairs = set()
for anchor, hyponyms in anchor_children.items():
    for hyponym in hyponyms:
        taxonomic_pairs.add((anchor, hyponym, max(anchor_concept_sims[(anchor, hyponym)])))

# read in concepts
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

CONCEPTS = defaultdict(lexicon.Concept)
with open(lemma_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["remove"] != "1":
            CONCEPTS[row["lemma"]] = lemma2concept(row)

CONCEPTS = dict(CONCEPTS)

final_pairs = set()

for item in negative_sample_triples:
    premise, conclusion = item[0], item[1]
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

with open(f"data/things/stimuli-pairs/things-inheritance-SPOSE_prototype_sim-pairs.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["premise", "conclusion", "hypernymy", "similarity_raw", "similarity_binary", "premise_form", "conclusion_form"])
    for row in final_pairs:
        writer.writerow(row)
