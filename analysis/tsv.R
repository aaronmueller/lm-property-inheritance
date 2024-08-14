library(tidyverse)
library(fs)
library(ggtext)

read_tsv_responses <- function(path) {
  tsv <- read_csv(path)
  
  tsv %>%
    group_by(hypernym) %>%
    mutate(normalized_score = (score - min(score))/(max(score) - min(score))) %>%
    ungroup()
}


mistral_tsv <- read_tsv_responses("data/things/things-mistralai_Mistral-7B-Instruct-v0.2-tsv.csv")
mistral_tsv_qa <- read_tsv_responses("data/things/things-mistralai_Mistral-7B-Instruct-v0.2-tsv-qa.csv")
mistral_tsv_qa_declarative <- read_tsv_responses("data/things/things-mistralai_Mistral-7B-Instruct-v0.2-tsv-qa-declarative.csv")
ensemble <- mistral_tsv %>%
  select(hyponym, hypernym, declarative_score = normalized_score) %>%
  inner_join(
    mistral_tsv_qa %>% select(hyponym, hypernym, qa_score = normalized_score)
  ) %>%
  inner_join(
    mistral_tsv_qa_declarative %>% select(hyponym, hypernym, qa_declarative_score = normalized_score)
  ) %>%
  mutate(
    mean_score = (declarative_score + qa_score + qa_declarative_score)/3
  ) %>%
  group_by(hypernym) %>%
  mutate(
    normalized_score = (mean_score - min(mean_score))/(max(mean_score) - min(mean_score))
  ) 

mistral_tsv_qa %>%
  filter(hypernym == "food", hypernymy==FALSE) %>%
  arrange(-score)


## read inheritance scores...

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
    sense_based_triples %>% select(id, anchor, hyponym, hyponym_type)
  ))


# deduction
deduction <- raw_results %>%
  filter(reasoning == "deduction") %>%
  select(model, chat_format, anchor, hyponym, hyponym_type, setting, logprob = prompt, control)

bind_rows(
  deduction %>% inner_join(mistral_tsv %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-declarative")),
  deduction %>% inner_join(mistral_tsv_qa %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-qa")),
  deduction %>% inner_join(mistral_tsv_qa_declarative %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-qa-declarative")),
  deduction %>% inner_join(ensemble %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-ensemble")),
  deduction %>% filter(hyponym_type != "sense_based_ns") %>% inner_join(model_based_sim %>% select(anchor, hyponym = concept, sim = similarity) %>% mutate(sim_type = "model-based")),
  deduction %>% filter(hyponym_type != "model_specific_ns") %>% inner_join(sense_based_sim %>% select(anchor, hyponym = concept, sim = similarity) %>% mutate(sim_type = "sense-based"))
) %>%
  mutate(controlled_logprob = logprob - control) %>%
  group_by(model, chat_format, setting, sim_type, hyponym_type) %>%
  nest() %>%
  mutate(
    n = map_dbl(data, function(x) {nrow(x)}),
    logprob_cor = map_dbl(data, function(x) {
      cor(x$logprob, x$sim, method = "spearman")
    }),
    controlled_logprob_cor = map_dbl(data, function(x) {
      cor(x$controlled_logprob, x$sim, method = "spearman")
    })
  ) %>% select(-data) %>% View()

deduction %>% 
  inner_join(
    ensemble %>% 
      select(anchor=hypernym, hyponym, sim = normalized_score) %>% 
      mutate(sim_type = "tsv-ensemble")
  ) %>%
  filter(setting != "Phrasal") %>%
  mutate(controlled_logprob = logprob - control) %>%
  group_by(model, anchor, chat_format, setting, sim_type, hyponym_type) %>%
  nest() %>%
  mutate(
    n = map_dbl(data, function(x) {nrow(x)}),
    logprob_cor = map_dbl(data, function(x) {
      cor(x$logprob, x$sim, method = "spearman")
    }),
    controlled_logprob_cor = map_dbl(data, function(x) {
      cor(x$controlled_logprob, x$sim, method = "spearman")
    })
  ) %>% select(-data) %>% 
  View()


deduction %>% 
  filter(model == "mistralai_Mistral-7B-Instruct-v0.2") %>%
  # filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  mutate(hypernymy = case_when(
    hyponym_type == "taxonomic" ~ "yes", TRUE ~ "no"
  )) %>%
  filter(hyponym_type != "model_specific_ns") %>% 
  inner_join(sense_based_sim %>% 
               select(anchor, hyponym = concept, sim = similarity) %>% 
               mutate(sim_type = "sense-based")) %>%
  filter(setting!="Phrasal") %>%
  ggplot(aes(sim, logprob, color = hypernymy)) +
  geom_point(size = 2, alpha = 0.5) +
  scale_color_brewer(palette = "Dark2") +
  theme_bw(base_size = 18, base_family = "Times") +
  theme(
    legend.position = "top",
    plot.title = element_markdown(size = 18),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Similarity (Sense-based)",
    y = "log P(Yes) - log P(No)",
    title = "<b>Prompt:</b>Given that X is daxable. Is it true that Y is daxable?<br>Answer with Yes/No:"
  )

deduction %>% 
  filter(model == "mistralai_Mistral-7B-Instruct-v0.2") %>%
  # filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  mutate(hypernymy = case_when(
    hyponym_type == "taxonomic" ~ "yes", TRUE ~ "no"
  )) %>%
  filter(hyponym_type != "model_specific_ns") %>% 
  inner_join(sense_based_sim %>% 
               select(anchor, hyponym = concept, sim = similarity) %>% 
               mutate(sim_type = "sense-based")) %>%
  filter(setting!="Phrasal") %>%
  ggplot(aes(sim, logprob, color = hypernymy)) +
  geom_point(size = 2, alpha = 0.5) +
  geom_hline(yintercept = 0.0, linetype="dashed") +
  facet_wrap(~anchor, nrow = 5) +
  scale_color_brewer(palette = "Dark2") +
  theme_bw(base_size = 18, base_family = "Times") +
  theme(
    legend.position = "top",
    plot.title = element_markdown(size = 18),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Similarity (Sense-based)",
    y = "log P(Yes) - log P(No)",
    title = "<b>Prompt:</b>Given that X is daxable. Is it true that Y is daxable?<br>Answer with Yes/No:"
  )


deduction %>% 
  # filter(model == "mistralai_Mistral-7B-Instruct-v0.2") %>%
  filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  mutate(hypernymy = case_when(
    hyponym_type == "taxonomic" ~ "yes", TRUE ~ "no"
  )) %>%
  # filter(hyponym_type == "taxonomic") %>%
  filter(hyponym_type != "model_specific_ns") %>%
  inner_join(ensemble %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-ensemble")) %>%
  # inner_join(sense_based_sim %>% select(anchor, hyponym = concept, sim = similarity) %>% mutate(sim_type = "sense-based")) %>%
  filter(setting!="Phrasal") %>%
  ggplot(aes(sim, logprob, color = hypernymy)) +
  geom_point(size = 2, alpha = 0.5) +
  geom_hline(yintercept = 0.0, linetype="dashed") +
  facet_wrap(~anchor, nrow = 5) +
  scale_color_brewer(palette = "Dark2") +
  theme_bw(base_size = 18, base_family = "Times") +
  theme(
    legend.position = "top",
    plot.title = element_markdown(size = 18),
    axis.text = element_text(color = "black")
  ) +
  labs(
    # x = "Similarity (Sense-based)",
    # x = "Typicality (within taxonomically related arguments)",
    x = "Typicality (for all arguments)",
    y = "log P(Yes) - log P(No)",
    title = "<b>Prompt:</b>Answer the question. Given that X is daxable. Is it true that Y is daxable?<0x0A>Answer with Yes/No.<0x0A>"
  )

deduction %>% 
  # filter(model == "mistralai_Mistral-7B-Instruct-v0.2") %>%
  filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  filter(hyponym_type != "sense_based_ns") %>%
  inner_join(model_based_sim %>% 
               select(anchor, hyponym = concept, sim = similarity) %>% 
               mutate(sim_type = "model-based")) %>%
  count(hyponym_type)
