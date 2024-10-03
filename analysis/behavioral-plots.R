library(tidyverse)
library(fs)
library(ggtext)

# model_names <- tribble(
#   ~model, ~model_id,
#   "Gemma-2-2B-it", "Gemma-2-2B-it",
#   "Gemma-2-9B-it", "Gemma-2-9B-it",
#   "Mistral-7B-Instruct-v0.2", "Mistral-7B-Instruct-v0.2"
# )

sense_based_sim <- read_csv("data/things/similarity/things-sense_based.csv")
spose_sim <- read_csv("data/things/similarity/things-SPOSE_prototype.csv")

taxonomic_triples <- read_csv("data/things/things-triples-actual.csv") %>%
  mutate(
    hyponym_type = "taxonomic",
    id = row_number()
  )

sense_based_triples <- read_csv("data/things/negative-samples/things-sense_based-ns_triples.csv") %>%
  mutate(
    hyponym_type = "sense_based-ns",
    id = row_number()
  ) %>%
  select(anchor = premise, anchor_sense = `premise-sense`, hyponym = conclusion, similarity, hyponym_type, id)

spose_triples <- read_csv("data/things/negative-samples/things-SPOSE_prototype-ns_triples.csv") %>%
  mutate(
    hyponym_type = "SPOSE_prototype-ns",
    id = row_number()
  ) %>%
  select(anchor = premise, hyponym = conclusion, similarity, hyponym_type, id)

sense_based_stimuli <- read_csv("data/things/stimuli-pairs/things-inheritance-sense_based_sim-pairs.csv")
spose_stimuli <- read_csv("data/things/stimuli-pairs/things-inheritance-SPOSE_prototype_sim-pairs.csv")

raw_results <- dir_ls("data/things/results/", regexp = "*.csv", recurse=TRUE) %>%
  map_df(read_csv, .id = "file") 

results <- raw_results %>%
  mutate(
    hyponym_type = str_extract(file, "(?<=results/)(.*)(?=/(deduction|induction))") %>% str_replace("things-", ""),
    negative_sample_type = str_extract(hyponym_type, "(?<=things-)(.*)(?=_ns)"),
    reasoning = case_when(
      str_detect(file, "deduction") ~ "deduction",
      TRUE ~ "induction"
    ),
    setting = case_when(
      str_detect(file, "logprobs") ~ "Phrasal",
      TRUE ~ "QA (Yes-No ratio)"
    ),
    chat_format = case_when(
      str_detect(file, "chat-format") ~ TRUE,
      TRUE ~ FALSE
    ),
    model = str_extract(file, "(?<=(logprobs|qa_format)/)(.*)(?=.csv)") %>%
      str_remove("_chat-format"),
    model = str_replace(model, "(google_|mistralai_)", ""),
    template = str_extract(model, "_(.*)") %>% str_replace("_", ""),
    model = str_extract(model, "(.*)_") %>% str_replace("_", "")
    # template = str_extract(file, glue::glue("(?<={model}_)(.*)(?=(_chat-format.csv|.csv))"))
  ) %>%
  filter(setting != "Phrasal") %>%
  select(-file) %>%
  group_by(model, hyponym_type, setting, reasoning, chat_format, template) %>%
  mutate(id = row_number()) %>%
  ungroup() %>%
  inner_join(bind_rows(
    taxonomic_triples %>% select(id, anchor, hyponym, hyponym_type),
    # model_based_triples %>% select(id, anchor, hyponym, hyponym_type),
    sense_based_triples %>% select(id, anchor, hyponym, hyponym_type),
    spose_triples %>% select(id, anchor, hyponym, hyponym_type)
  )) %>%
  mutate(
    premise = case_when(reasoning == "deduction" ~ anchor, TRUE ~ hyponym),
    conclusion = case_when(reasoning == "deduction" ~ hyponym, TRUE ~ anchor),
  )

# deduction

deduction <- results %>%
  filter(reasoning == "deduction") %>%
  select(model, chat_format, template, anchor, hyponym, hyponym_type, setting, logprob = prompt, control) %>%
  distinct() %>%
  group_by(model, setting, chat_format, template, anchor, hyponym, hyponym_type) %>% 
  slice(1) %>%
  ungroup()

sim_deduction <- bind_rows(
  deduction %>%
    filter(hyponym_type %in% c("taxonomic", "sense_based-ns")) %>%
    inner_join(sense_based_stimuli %>% rename(anchor = premise, hyponym = conclusion)),
  deduction %>%
    filter(hyponym_type %in% c("taxonomic", "SPOSE_prototype-ns")) %>%
    inner_join(spose_stimuli %>% rename(anchor = premise, hyponym = conclusion))
)



