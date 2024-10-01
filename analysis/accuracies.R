library(tidyverse)
library(fs)
library(ggtext)

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

model_based_triples <- read_csv("data/things/negative-samples/things-mistralai_Mistral-7B-Instruct-v0.2_final_layer-ns_triples.csv") %>%
  mutate(
    hyponym_type = "model_specific-ns", 
    id = row_number()
  ) %>%
  select(anchor = premise, hyponym = conclusion, similarity, hyponym_type, id)

raw_results <- dir_ls("data/things/results/", regexp = "*.csv", recurse=TRUE) %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    hyponym_type = str_extract(file, "(?<=results/)(.*)(?=/(deduction|induction))") %>% str_replace("things-", ""),
    negative_sample_type = str_extract(hyponym_type, "(?<=things-)(.*)(?=_ns)"),
    # negative_sample_type = case_when(
    #   is.na(negative_sample_type) ~ "none",
    #   TRUE ~ negative_sample_type
    # ),
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
    # template = str_extract(file, glue::glue("(?<={model}_)(.*)(?=(_chat-format.csv|.csv))"))
  ) %>%
  select(-file) %>%
  group_by(model, hyponym_type, setting, reasoning, chat_format) %>%
  mutate(id = row_number()) %>%
  ungroup() %>%
  inner_join(bind_rows(
    taxonomic_triples %>% select(id, anchor, hyponym, hyponym_type),
    model_based_triples %>% select(id, anchor, hyponym, hyponym_type),
    sense_based_triples %>% select(id, anchor, hyponym, hyponym_type),
    spose_triples %>% select(id, anchor, hyponym, hyponym_type)
  )) %>%
  mutate(
    premise = case_when(reasoning == "deduction" ~ anchor, TRUE ~ hyponym),
    conclusion = case_when(reasoning == "deduction" ~ hyponym, TRUE ~ anchor),
  )

raw_results %>% count(model)
  filter(hyponym_type != "model_specific_ns") %>%
  mutate(
    
  )

raw_results %>%
  filter(setting != "Phrasal") %>%
  group_by(model, hyponym_type, setting, reasoning, chat_format) %>%
  mutate(
    vs_control = prompt > control,
    vs_empty = prompt > empty,
    correctness = case_when(
      hyponym_type == "taxonomic" ~ prompt > 0,
      TRUE ~ prompt < 0
    )
  ) |>
  ungroup() %>%
  # filter(hyponym_type != "model_specific_ns") %>%
  # filter(!hyponym_type %in% c("model_specific-ns", "SPOSE_prototype-ns")) %>%
  filter(hyponym_type %in% c("taxonomic", "SPOSE_prototype-ns")) %>%
  # filter(hyponym_type == "taxonomic") %>%
  group_by(model, setting, reasoning, chat_format) %>%
  summarize(
    n = n(),
    control_accuracy = mean(vs_control),
    empty_accuracy = mean(vs_empty),
    taxonomic_accuracy = mean(correctness)
  ) %>% View()

