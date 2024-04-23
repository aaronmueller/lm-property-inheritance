library(tidyverse)
library(fs)
library(glue)

sense_based_sim <- read_csv("data/things/things-sense_based_anchor_sims.csv")
model_based_sim <- read_csv("data/things/things-mistralai_Mistral-7B-Instruct-v0.2_layer32_anchor_sims.csv")

taxonomic_triples <- read_csv("data/things/things-triples-actual.csv") %>%
  mutate(
    hyponym_type = "taxonomic", 
    id = row_number()
  )

sense_based_triples <- read_csv("data/things/things-sense_based_ns-triples.csv") %>%
  mutate(
    hyponym_type = "sense_based_ns", 
    id = row_number()
  )

model_based_triples <- read_csv("data/things/things-mistralai_Mistral-7B-Instruct-v0.2_layer32_ns-triples.csv") %>%
  mutate(
    hyponym_type = "model_specific_ns", 
    id = row_number()
  )

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
    model = str_extract(file, "(?<=(logprobs|qa_format)/)(.*)(?=.csv)")
  ) %>%
  select(-file)


raw_results %>%
  group_by(model, hyponym_type, setting, reasoning) %>%
  mutate(id = row_number()) %>%
  ungroup() %>%
  inner_join(bind_rows(
    taxonomic_triples %>% select(id, anchor, hyponym),
    model_based_triples %>% select(id, anchor, hyponym)
  ))

raw_results %>%
  mutate(
    hyponym_type = factor(
      hyponym_type, 
      levels = c("taxonomic", "sense_based_ns", "model_specific_ns"),
      labels = c("Taxonomic", "Sense-based\nNS", "Model-specific\nNS")
    )
  )
  group_by(model, hyponym_type, setting, reasoning) %>%
  summarize(
    n = n(),
    ste = 1.96 * plotrix::std.error(prompt),
    logprob = mean(prompt)
  ) %>%
  ggplot(aes(hyponym_type, logprob, color = reasoning, shape = reasoning, group = reasoning)) +
  geom_point(size = 2)+
  geom_linerange(aes(ymin = logprob-ste, ymax = logprob+ste)) +
  geom_line() +
  scale_y_continuous(breaks = scales::pretty_breaks()) +
  scale_color_brewer(palette = "Dark2") +
  facet_wrap(~setting, scales = "free_y") +
  theme_bw(base_size = 16, base_family = "Times") + 
  theme(
    legend.position = "top",
    panel.grid = element_blank()
  ) +
  labs(
    x = "Hyponym Type",
    y = "LogProb",
    color = "Reasoning",
    shape = "Reasoning"
  )
