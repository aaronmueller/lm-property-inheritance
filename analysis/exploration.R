library(tidyverse)

things_og <- read_tsv("data/things/category53_longFormat.tsv")

og_lemmas <- things_og %>%
  mutate(uniqueID = str_replace_all(uniqueID, "\\d", "")) %>%
  distinct(uniqueID) %>%
  rename(lemma = uniqueID)


og_pairs <- things_og %>%
  mutate(uniqueID = str_replace_all(uniqueID, "\\d", "")) %>%
  distinct(category, uniqueID) %>%
  filter(category != uniqueID) %>%
  select(premise = category, conclusion = uniqueID) %>%
  mutate(
    premise = str_replace_all(premise, " ", "_"),
    conclusion = str_replace_all(conclusion, " ", "_")
  )

new_pairs <- stimuli %>%
  filter(premise != conclusion, hypernymy == "yes") %>%
  mutate(
    premise = str_replace_all(premise, " ", "_"),
    conclusion = str_replace_all(conclusion, " ", "_")
  ) %>%
  select(premise, conclusion)

og_pairs %>%
  anti_join(new_pairs) %>%
  View()

things <- read_csv("data/things/things-lemmas-annotated.csv")

new_lemmas <- things %>%
  mutate(lemma = str_replace_all(lemma, " ", "_")) %>%
  distinct(lemma)

og_lemmas %>%
  anti_join(new_lemmas)

new_lemmas %>%
  anti_join(og_lemmas)

stimuli <- read_csv("data/things/stimuli-pairs/things-inheritance-sense_based_sim-pairs.csv")

stimuli %>%
  filter(hypernymy == "yes", premise != conclusion) %>%
  count(premise, conclusion)
