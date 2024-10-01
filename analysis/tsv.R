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


mistral_tsv <- read_tsv_responses("data/things/tsv/results/things-mistralai_Mistral-7B-Instruct-v0.2-tsv.csv")
mistral_tsv_qa <- read_tsv_responses("data/things/tsv/results/things-mistralai_Mistral-7B-Instruct-v0.2-tsv-qa.csv")
mistral_tsv_qa_declarative <- read_tsv_responses("data/things/tsv/results/things-mistralai_Mistral-7B-Instruct-v0.2-tsv-qa-declarative.csv")
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

sense_based_sim <- read_csv("data/things/similarity/things-sense_based.csv")
model_based_sim <- read_csv("data/things/similarity/things-mistralai_Mistral-7B-Instruct-v0.2_layer32_anchor_sims.csv")
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
    # model_based_triples %>% select(id, anchor, hyponym, hyponym_type),
    sense_based_triples %>% select(id, anchor, hyponym, hyponym_type),
    spose_triples %>% select(id, anchor, hyponym, hyponym_type)
  )) %>%
  mutate(
    premise = case_when(reasoning == "deduction" ~ anchor, TRUE ~ hyponym),
    conclusion = case_when(reasoning == "deduction" ~ hyponym, TRUE ~ anchor),
  )


# deduction
deduction <- raw_results %>%
  filter(reasoning == "induction") %>%
  select(model, chat_format, anchor, hyponym, hyponym_type, setting, logprob = prompt, control)

sim_results <- bind_rows(
  deduction %>% inner_join(mistral_tsv %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-declarative")),
  deduction %>% inner_join(mistral_tsv_qa %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-qa")),
  deduction %>% inner_join(mistral_tsv_qa_declarative %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-qa-declarative")),
  deduction %>% inner_join(ensemble %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-ensemble")),
  # deduction %>% filter(hyponym_type == "m") %>% inner_join(model_based_sim %>% select(anchor, hyponym = concept, sim = similarity) %>% mutate(sim_type = "model-based")),
  deduction %>% filter(hyponym_type %in% c("sense_based-ns", "taxonomic")) %>% inner_join(sense_based_sim %>% select(anchor=concept1, hyponym = concept2, sim = similarity) %>% mutate(sim_type = "sense-based")),
  deduction %>% filter(hyponym_type %in% c("SPOSE_prototype-ns", "taxonomic")) %>% inner_join(spose_sim %>% select(anchor=concept1, hyponym = concept2, sim = similarity) %>% mutate(sim_type = "SPOSE"))
) %>%
  filter(setting != "Phrasal") %>%
  mutate(
    controlled_logprob = logprob - control,
    hypernymy = case_when(
      hyponym_type == "taxonomic" ~ "yes", TRUE ~ "no",
    ),
  ) %>%
  group_by(sim_type) %>%
  mutate(
    sim_class = case_when(
      sim > median(sim) ~ "high",
      TRUE ~ "low"
    )
  ) %>%
  ungroup() %>%
  mutate(
    condition = case_when(
      hypernymy == "yes" & sim_class == "high" ~ "Taxonomic\nHigh-Sim",
      hypernymy == "yes" & sim_class == "low" ~ "Taxonomic\nLow-Sim",
      hypernymy == "no" & sim_class == "low" ~ "Non-Taxonomic\nLow-Sim",
      TRUE ~ "Non-Taxonomic\nHigh-Sim",
    ),
    condition = factor(condition)
  )
  
  
sim_results %>% 
  group_by(model, chat_format, setting, hyponym_type, sim_type) %>%
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

sim_results%>%
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


sim_results %>% 
  # filter(str_detect(model, "initial-qa"), chat_format == FALSE) %>% 
  filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  filter(sim_type == "sense-based") %>%
  filter(hyponym_type %in% c("sense_based-ns", "taxonomic")) %>%
  mutate(hypernymy = case_when(
    hyponym_type == "taxonomic" ~ "yes", TRUE ~ "no"
  )) %>%
  ggplot(aes(sim, logprob, color = hypernymy, shape = hypernymy)) +
  geom_point(size = 2, alpha = 0.5) +
  geom_hline(yintercept = 0.0, linetype = "dashed") +
  scale_color_brewer(palette = "Dark2") +
  theme_bw(base_size = 18, base_family = "Times") +
  theme(
    legend.position = "top",
    plot.title = element_markdown(size = 18),
    axis.text = element_text(color = "black"),
    panel.grid = element_blank()
  ) +
  labs(
    x = "Similarity (Sense-based)",
    y = "log P(Yes) - log P(No)",
    title = "<b>Prompt:</b>Given that X is daxable. Is it true that Y is daxable?<br>Answer with Yes/No:"
  )


sim_results %>% 
  # filter(str_detect(model, "initial-qa"), chat_format == FALSE) %>%
  filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  filter(sim_type == "SPOSE") %>%
  filter(hyponym_type %in% c("SPOSE_prototype-ns", "taxonomic")) %>%
  # filter(sim_type == "sense-based") %>%
  # filter(hyponym_type %in% c("sense_based-ns", "taxonomic")) %>%
  group_by(hypernymy, sim_class, condition) %>%
  summarize(
    ste = 1.96 * plotrix::std.error(logprob),
    logprob = mean(logprob),
    n = n()
  ) %>%
  ungroup() %>%
  ggplot(aes(condition, logprob, color = hypernymy, fill = hypernymy, shape = sim_class, group = hypernymy)) +
  geom_point(size = 2) + 
  geom_linerange(aes(ymin = logprob-ste, ymax = logprob + ste)) +
  geom_line() +
  scale_color_brewer(palette = "Dark2", aesthetics = c("color", "fill")) +
  scale_shape_manual(values = c(24, 25)) +
  scale_y_continuous(breaks = scales::pretty_breaks(6)) +
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    legend.position = "none",
    panel.grid = element_blank(),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Condition",
    y = "log P(Yes) - log P(No)"
  )

sim_results %>% 
  # filter(str_detect(model, "initial-qa"), chat_format == FALSE) %>%
  filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  filter(sim_type == "sense-based") %>%
  filter(hyponym_type %in% c("sense_based-ns", "taxonomic")) %>%
  group_by(hypernymy, sim_class, condition) %>%
  summarize(
    ste = 1.96 * plotrix::std.error(logprob),
    logprob = mean(logprob),
    n = n()
  ) %>%
  ungroup() %>%
  ggplot(aes(condition, logprob, color = hypernymy, fill = hypernymy, shape = sim_class, group = hypernymy)) +
  geom_point(size = 2) + 
  geom_linerange(aes(ymin = logprob-ste, ymax = logprob + ste)) +
  geom_line() +
  scale_color_brewer(palette = "Dark2", aesthetics = c("color", "fill")) +
  scale_shape_manual(values = c(24, 25)) +
  scale_y_continuous(breaks = scales::pretty_breaks(6)) +
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    legend.position = "none",
    panel.grid = element_blank(),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Condition",
    y = "log P(Yes) - log P(No)"
  )

sim_results %>% 
  # filter(str_detect(model, "initial-qa"), chat_format == FALSE) %>% 
  filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  filter(sim_type == "SPOSE") %>%
  filter(hyponym_type %in% c("SPOSE_prototype-ns", "taxonomic")) %>%
  mutate(hypernymy = case_when(
    hyponym_type == "taxonomic" ~ "yes", TRUE ~ "no"
  )) %>%
  ggplot(aes(sim, logprob, color = hypernymy, shape = hypernymy)) +
  geom_point(size = 2, alpha = 0.5) +
  geom_hline(yintercept = 0.0, linetype = "dashed") +
  scale_color_brewer(palette = "Dark2") +
  theme_bw(base_size = 18, base_family = "Times") +
  theme(
    legend.position = "top",
    plot.title = element_markdown(size = 18),
    axis.text = element_text(color = "black"),
    panel.grid = element_blank()
  ) +
  labs(
    x = "Similarity (Sense-based)",
    y = "log P(Yes) - log P(No)",
    title = "<b>Prompt:</b>Given that X is daxable. Is it true that Y is daxable?<br>Answer with Yes/No:"
  )


deduction %>% 
  # filter(str_detect(model, "initial-qa"), chat_format == FALSE) %>%
  filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  mutate(hypernymy = case_when(
    hyponym_type == "taxonomic" ~ "yes", TRUE ~ "no"
  )) %>%
  filter(hyponym_type %in% c("SPOSE_prototype-ns", "taxonomic")) %>%
  inner_join(spose_sim %>% select(anchor=concept1, hyponym = concept2, sim = similarity) %>% mutate(sim_type = "SPOSE")) %>%
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

# deduction %>% 
#   filter(model == "mistralai_Mistral-7B-Instruct-v0.2") %>%
#   # filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
#   mutate(hypernymy = case_when(
#     hyponym_type == "taxonomic" ~ "yes", TRUE ~ "no"
#   )) %>%
#   filter(hyponym_type != "model_specific_ns") %>% 
#   inner_join(sense_based_sim %>% 
#                select(anchor, hyponym = concept, sim = similarity) %>% 
#                mutate(sim_type = "sense-based")) %>%
#   filter(setting!="Phrasal") %>%
deduction %>% 
  # filter(str_detect(model, "initial-qa"), chat_format == FALSE) %>%
  filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  mutate(hypernymy = case_when(
    hyponym_type == "taxonomic" ~ "yes", TRUE ~ "no"
  )) %>%
  filter(hyponym_type %in% c("SPOSE_prototype-ns", "taxonomic")) %>%
  inner_join(spose_sim %>% select(anchor=concept1, hyponym = concept2, sim = similarity) %>% mutate(sim_type = "SPOSE")) %>%
  # filter(hyponym_type == "taxonomic") %>%
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


### Fine grained plots?

deduction <- raw_results %>%
  filter(reasoning != "induction") %>%
  select(model, chat_format, anchor, hyponym, hyponym_type, setting, logprob = prompt, control)

sim_results <- bind_rows(
  deduction %>% inner_join(mistral_tsv %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-declarative")),
  deduction %>% inner_join(mistral_tsv_qa %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-qa")),
  deduction %>% inner_join(mistral_tsv_qa_declarative %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-qa-declarative")),
  deduction %>% inner_join(ensemble %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-ensemble")),
  # deduction %>% filter(hyponym_type == "m") %>% inner_join(model_based_sim %>% select(anchor, hyponym = concept, sim = similarity) %>% mutate(sim_type = "model-based")),
  deduction %>% filter(hyponym_type %in% c("sense_based-ns", "taxonomic")) %>% inner_join(sense_based_sim %>% select(anchor=concept1, hyponym = concept2, sim = similarity) %>% mutate(sim_type = "sense-based")),
  deduction %>% filter(hyponym_type %in% c("SPOSE_prototype-ns", "taxonomic")) %>% inner_join(spose_sim %>% select(anchor=concept1, hyponym = concept2, sim = similarity) %>% mutate(sim_type = "SPOSE"))
) %>%
  filter(setting != "Phrasal") %>%
  mutate(
    controlled_logprob = logprob - control,
    hypernymy = case_when(
      hyponym_type == "taxonomic" ~ "yes", TRUE ~ "no",
    ),
  ) %>%
  group_by(sim_type) %>%
  mutate(
    sim_class = case_when(
      sim > median(sim) ~ "high",
      TRUE ~ "low"
    )
  ) %>%
  ungroup() %>%
  mutate(
    condition = case_when(
      hypernymy == "yes" & sim_class == "high" ~ "Taxonomic\nHigh-Sim",
      hypernymy == "yes" & sim_class == "low" ~ "Taxonomic\nLow-Sim",
      hypernymy == "no" & sim_class == "low" ~ "Non-Taxonomic\nLow-Sim",
      TRUE ~ "Non-Taxonomic\nHigh-Sim",
    ),
    condition = factor(condition)
  )

# "accuracy"

sim_results %>% 
  # filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  filter(sim_type %in% c("sense-based", "SPOSE")) %>% 
  mutate(
    correctness = case_when(
      hypernymy == "yes" ~ logprob > 0,
      TRUE ~ logprob < 0
    )
  ) %>%
  group_by(model, chat_format, sim_type) %>% 
  summarize(
    accuracy = mean(correctness)
  )

# aggregated plot:

sim_results %>% 
  filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  filter(sim_type %in% c("sense-based", "SPOSE")) %>%
  group_by(sim_type, hypernymy, sim_class, condition) %>%
  summarize(
    ste = 1.96 * plotrix::std.error(logprob),
    logprob = mean(logprob),
    n = n()
  ) %>%
  ungroup() %>%
  ggplot(aes(condition, logprob, color = hypernymy, fill = hypernymy, shape = sim_class, group = hypernymy)) +
  geom_point(size = 2) + 
  geom_linerange(aes(ymin = logprob-ste, ymax = logprob + ste)) +
  geom_line() +
  facet_wrap(~sim_type) +
  scale_color_brewer(palette = "Dark2", aesthetics = c("color", "fill")) +
  scale_shape_manual(values = c(24, 25)) +
  scale_y_continuous(breaks = scales::pretty_breaks(6)) +
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    legend.position = "none",
    panel.grid = element_blank(),
    axis.text = element_text(color = "black"),
    plot.title = element_markdown(size = 18),
  ) +
  labs(
    title = "<b>Reasoning Type:</b> Inheritance",
    x = "Condition",
    y = "log P(Yes) - log P(No)"
  )

sim_results %>%
  filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  filter(sim_type == "sense-based") %>%
  ggplot(aes(sim, logprob, color = hypernymy, shape = hypernymy)) +
  geom_point(size = 2, alpha = 0.5) +
  geom_hline(yintercept = 0.0, linetype="dashed") +
  facet_wrap(~anchor, nrow = 5) +
  scale_color_brewer(palette = "Dark2") +
  theme_bw(base_size = 18, base_family = "Times") +
  theme(
    legend.position = "top",
    plot.title = element_markdown(size = 18),
    axis.text = element_text(color = "black"),
    panel.grid = element_blank()
  ) +
  labs(
    color = "Hypernymy",
    shape = "Hypernymy",
    x = "Similarity",
    y = "log P(Yes) - log P(No)",
    title = "<b>Reasoning Type:</b> Inheritance; <b>Similarity</b>: Sense-based"
  )

ggsave("paper/fine-grained-inheritance-sense-based-plot.pdf", height = 10.81, width = 16.73, dpi = 300, device=cairo_pdf)

sim_results %>%
  filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  filter(sim_type == "SPOSE") %>%
  ggplot(aes(sim, logprob, color = hypernymy, shape = hypernymy)) +
  geom_point(size = 2, alpha = 0.5) +
  geom_hline(yintercept = 0.0, linetype="dashed") +
  facet_wrap(~anchor, nrow = 5) +
  scale_color_brewer(palette = "Dark2") +
  theme_bw(base_size = 18, base_family = "Times") +
  theme(
    legend.position = "top",
    plot.title = element_markdown(size = 18),
    axis.text = element_text(color = "black"),
    panel.grid = element_blank()
  ) +
  labs(
    color = "Hypernymy",
    shape = "Hypernymy",
    x = "Similarity",
    y = "log P(Yes) - log P(No)",
    title = "<b>Reasoning Type:</b> Inheritance; <b>Similarity</b>: SPOSE"
  )

ggsave("paper/fine-grained-inheritance-SPOSE-plot.pdf", height = 10.81, width = 16.73, dpi = 300, device=cairo_pdf)



deduction <- raw_results %>%
  filter(reasoning == "induction") %>%
  select(model, chat_format, anchor, hyponym, hyponym_type, setting, logprob = prompt, control)

sim_results <- bind_rows(
  deduction %>% inner_join(mistral_tsv %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-declarative")),
  deduction %>% inner_join(mistral_tsv_qa %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-qa")),
  deduction %>% inner_join(mistral_tsv_qa_declarative %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-qa-declarative")),
  deduction %>% inner_join(ensemble %>% select(anchor=hypernym, hyponym, sim = normalized_score) %>% mutate(sim_type = "tsv-ensemble")),
  # deduction %>% filter(hyponym_type == "m") %>% inner_join(model_based_sim %>% select(anchor, hyponym = concept, sim = similarity) %>% mutate(sim_type = "model-based")),
  deduction %>% filter(hyponym_type %in% c("sense_based-ns", "taxonomic")) %>% inner_join(sense_based_sim %>% select(anchor=concept1, hyponym = concept2, sim = similarity) %>% mutate(sim_type = "sense-based")),
  deduction %>% filter(hyponym_type %in% c("SPOSE_prototype-ns", "taxonomic")) %>% inner_join(spose_sim %>% select(anchor=concept1, hyponym = concept2, sim = similarity) %>% mutate(sim_type = "SPOSE"))
) %>%
  filter(setting != "Phrasal") %>%
  mutate(
    controlled_logprob = logprob - control,
    hypernymy = case_when(
      hyponym_type == "taxonomic" ~ "yes", TRUE ~ "no",
    ),
  ) %>%
  group_by(sim_type) %>%
  mutate(
    sim_class = case_when(
      sim > median(sim) ~ "high",
      TRUE ~ "low"
    )
  ) %>%
  ungroup() %>%
  mutate(
    condition = case_when(
      hypernymy == "yes" & sim_class == "high" ~ "Taxonomic\nHigh-Sim",
      hypernymy == "yes" & sim_class == "low" ~ "Taxonomic\nLow-Sim",
      hypernymy == "no" & sim_class == "low" ~ "Non-Taxonomic\nLow-Sim",
      TRUE ~ "Non-Taxonomic\nHigh-Sim",
    ),
    condition = factor(condition)
  )


sim_results %>% 
  filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  filter(sim_type %in% c("sense-based", "SPOSE")) %>%
  group_by(sim_type, hypernymy, sim_class, condition) %>%
  summarize(
    ste = 1.96 * plotrix::std.error(logprob),
    logprob = mean(logprob),
    n = n()
  ) %>%
  ungroup() %>%
  ggplot(aes(condition, logprob, color = hypernymy, fill = hypernymy, shape = sim_class, group = hypernymy)) +
  geom_point(size = 2) + 
  geom_linerange(aes(ymin = logprob-ste, ymax = logprob + ste)) +
  geom_line() +
  facet_wrap(~sim_type) +
  scale_color_brewer(palette = "Dark2", aesthetics = c("color", "fill")) +
  scale_shape_manual(values = c(24, 25)) +
  scale_y_continuous(breaks = scales::pretty_breaks(6)) +
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    legend.position = "none",
    panel.grid = element_blank(),
    axis.text = element_text(color = "black"),
    plot.title = element_markdown(size = 18),
  ) +
  labs(
    title = "<b>Reasoning Type:</b> Reversed-Inheritance",
    x = "Condition",
    y = "log P(Yes) - log P(No)"
  )

sim_results %>%
  filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  filter(sim_type == "sense-based") %>%
  ggplot(aes(sim, logprob, color = hypernymy, shape = hypernymy)) +
  geom_point(size = 2, alpha = 0.5) +
  geom_hline(yintercept = 0.0, linetype="dashed") +
  facet_wrap(~anchor, nrow = 5) +
  scale_color_brewer(palette = "Dark2") +
  theme_bw(base_size = 18, base_family = "Times") +
  theme(
    legend.position = "top",
    plot.title = element_markdown(size = 18),
    axis.text = element_text(color = "black"),
    panel.grid = element_blank()
  ) +
  labs(
    color = "Hypernymy",
    shape = "Hypernymy",
    x = "Similarity",
    y = "log P(Yes) - log P(No)",
    title = "<b>Reasoning Type:</b> Reversed-Inheritance; <b>Similarity</b>: Sense-based"
  )

ggsave("paper/fine-grained-reversed-inheritance-sense-based-plot.pdf", height = 10.81, width = 16.73, dpi = 300, device=cairo_pdf)

sim_results %>%
  filter(str_detect(model, "mistral-special"), chat_format == FALSE) %>%
  filter(sim_type == "SPOSE") %>%
  ggplot(aes(sim, logprob, color = hypernymy, shape = hypernymy)) +
  geom_point(size = 2, alpha = 0.5) +
  geom_hline(yintercept = 0.0, linetype="dashed") +
  facet_wrap(~anchor, nrow = 5) +
  scale_color_brewer(palette = "Dark2") +
  theme_bw(base_size = 18, base_family = "Times") +
  theme(
    legend.position = "top",
    plot.title = element_markdown(size = 18),
    axis.text = element_text(color = "black"),
    panel.grid = element_blank()
  ) +
  labs(
    color = "Hypernymy",
    shape = "Hypernymy",
    x = "Similarity",
    y = "log P(Yes) - log P(No)",
    title = "<b>Reasoning Type:</b> Reversed-Inheritance; <b>Similarity</b>: SPOSE"
  )

ggsave("paper/fine-grained-reversed-inheritance-SPOSE-plot.pdf", height = 10.81, width = 16.73, dpi = 300, device=cairo_pdf)


