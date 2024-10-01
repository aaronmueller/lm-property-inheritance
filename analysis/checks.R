library(tidyverse)


things <- read_csv("data/things/things-triples-actual.csv")

things %>%
  count(anchor, hyponym, sort=TRUE)

tax <- things %>%
  distinct(anchor, hyponym) %>%
  select(premise=anchor, conclusion=hyponym) %>%
  arrange(premise, conclusion)

sb_triples <- read_csv("data/things/negative-samples/things-sense_based-ns_triples.csv")

sb <- sb_triples %>%
  distinct(premise, conclusion) %>%
  arrange(premise, conclusion)

sb %>%
  anti_join(tax)

tax_sb <- read_csv("data/things/stimuli-pairs/things-inheritance-sense_based_sim-pairs.csv")

tax_sb %>%
  distinct(premise, conclusion, hypernymy) %>%
  count(premise, hypernymy) %>%
  pivot_wider(names_from = hypernymy, values_from = n) %>%
  filter(yes > no)

tax_mb <- read_csv("data/things/stimuli-pairs/things-inheritance-mistralai_Mistral-7B-Instruct-v0.2_final_layer_sim-pairs.csv")
tax_mb %>%
  distinct(premise, conclusion, hypernymy) %>%
  count(premise, hypernymy) %>%
  pivot_wider(names_from = hypernymy, values_from = n) %>%
  filter(yes > no)



things_lemmas <- read_csv("data/things/things-lemmas-annotated.csv")

things_lemmas %>%
  select(lemma, singular = article, plural, taxonomic_phrase, generic) %>%
  mutate(singular = str_remove(singular, "^(a|an)\\s")) %>%
  write_csv("~/projects/hypernymy-signals/data/hypernymy/things-lemmas-annotated.csv")
