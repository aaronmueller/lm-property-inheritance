library(tidyverse)
library(fs)
library(ggtext)
library(lme4)
library(lmerTest)

# model_names <- tribble(
#   ~model, ~model_id,
#   "Gemma-2-2B-it", "Gemma-2-2B-it",
#   "Gemma-2-9B-it", "Gemma-2-9B-it",
#   "Mistral-7B-Instruct-v0.2", "Mistral-7B-Instruct-v0.2"
# )

sense_based_sim <- read_csv("data/things/similarity/things-sense_based.csv")
spose_sim <- read_csv("data/things/similarity/things-SPOSE_prototype.csv")

# taxonomic_triples <- read_csv("data/things/things-triples-actual.csv") %>%
#   mutate(
#     hyponym_type = "taxonomic",
#     id = row_number()
#   )
# 
# sense_based_triples <- read_csv("data/things/negative-samples/things-sense_based-ns_triples.csv") %>%
#   mutate(
#     hyponym_type = "sense_based-ns",
#     id = row_number()
#   ) %>%
#   select(anchor = premise, anchor_sense = `premise-sense`, hyponym = conclusion, similarity, hyponym_type, id)
# 
# spose_triples <- read_csv("data/things/negative-samples/things-SPOSE_prototype-ns_triples.csv") %>%
#   mutate(
#     hyponym_type = "SPOSE_prototype-ns",
#     id = row_number()
#   ) %>%
#   select(anchor = premise, hyponym = conclusion, similarity, hyponym_type, id)

sense_based_stimuli <- read_csv("data/things/stimuli-pairs/things-inheritance-sense_based_sim-pairs.csv")
spose_stimuli <- read_csv("data/things/stimuli-pairs/things-inheritance-SPOSE_prototype_sim-pairs.csv")

raw_results <- dir_ls("data/things/results/", regexp = "*.csv", recurse=TRUE) %>%
  map_df(read_csv, .id = "file") 

results <- raw_results %>%
  mutate(
    hyponym_type = str_extract(file, "(?<=results/)(.*)(?=/(deduction|induction))") %>% str_replace("things-", ""),
    # negative_sample_type = str_extract(file, "(?<=things-)(.*)(?=_ns)"),
    multi_property = str_detect(hyponym_type, "multi-property"),
    property_contrast = str_detect(hyponym_type, "prop-contrast"),
    hyponym_type = str_remove(hyponym_type, "_multi-property"),
    hyponym_type = str_remove(hyponym_type, "_prop-contrast"),
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
    model = str_replace(model, "(google_|mistralai_|meta-llama_Meta-)", ""),
    template = str_extract(model, "_(.*)") %>% str_replace("_", ""),
    model = str_extract(model, "(.*)_") %>% str_replace("_", "")
    # template = str_extract(file, glue::glue("(?<={model}_)(.*)(?=(_chat-format.csv|.csv))"))
  ) %>%
  filter(setting != "Phrasal") %>%
  select(-file) %>%
  group_by(model, hyponym_type, setting, reasoning, chat_format, template, multi_property, property_contrast) %>%
  mutate(id = row_number()) %>%
  ungroup() %>%
  inner_join(bind_rows(
    sense_based_stimuli %>% mutate(id = row_number(), hyponym_type = "sense_based-ns") %>% rename(anchor = premise, hyponym = conclusion),
    spose_stimuli %>% mutate(id = row_number(), hyponym_type = "SPOSE_prototype-ns") %>% rename(anchor = premise, hyponym = conclusion)
    # taxonomic_triples %>% select(id, anchor, hyponym, hyponym_type),
    # # model_based_triples %>% select(id, anchor, hyponym, hyponym_type),
    # sense_based_triples %>% select(id, anchor, hyponym, hyponym_type),
    # spose_triples %>% select(id, anchor, hyponym, hyponym_type)
  )) %>%
  mutate(
    premise = case_when(reasoning == "deduction" ~ anchor, TRUE ~ hyponym),
    conclusion = case_when(reasoning == "deduction" ~ hyponym, TRUE ~ anchor),
  )

results %>% count(model)

# deduction

deduction <- results %>%
  filter(reasoning == "deduction", multi_property == FALSE, property_contrast == FALSE) %>%
  mutate(
    condition = case_when(
      hypernymy == "yes" & similarity_binary == "high" ~ "+Tax\n+Sim",
      hypernymy == "yes" & similarity_binary == "low" ~ "+Tax\n-Sim",
      hypernymy == "no" & similarity_binary == "low" ~ "-Tax\n-Sim",
      TRUE ~ "-Tax\n+Sim",
    ),
    condition = factor(condition, levels = c("-Tax\n-Sim", "-Tax\n+Sim", "+Tax\n-Sim", "+Tax\n+Sim")),
    correctness = case_when(
      hypernymy == "yes" ~ diff > 0,
      TRUE ~ diff < 0
    ),
    sim_type = case_when(
      str_detect(hyponym_type, "sense_based") ~ "Sense",
      TRUE ~ "SPOSE"
    ),
    model = factor(
      model, 
      levels = c("Gemma-2-2B-it", "Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.2", "Gemma-2-9B-it"),
      labels = c("Gemma-2-2B-IT", "Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.2", "Gemma-2-9B-IT")
    )
  ) 
# %>%
  # filter(anchor != hyponym) 

taxonomic_sensitivities <- deduction %>%
  group_by(model, chat_format, template, sim_type) %>% 
  summarize(
    n = n(),
    taxonomic_sensitivity = mean(correctness)
  ) %>%
  ungroup()

# best_configs <- taxonomic_sensitivities %>%
#   group_by(model, sim_type) %>%
#   filter(taxonomic_sensitivity == max(taxonomic_sensitivity)) %>%
#   ungroup() %>%
#   select(model, chat_format, sim_type, template)

best_configs <- tribble(
  ~model, ~chat_format, ~template,
  "Gemma-2-2B-IT", FALSE, "variation-qa-1",
  "Gemma-2-9B-IT", TRUE, "variation-qa-2",
  "Llama-3-8B-Instruct", FALSE, "variation-qa-2",
  "Mistral-7B-Instruct-v0.2", FALSE, "variation-qa-1-mistral-special"
)

# similarity plots

logprobs <- deduction %>%
  inner_join(best_configs) %>%
  select(-correctness) %>%
  mutate(
    logprob = diff, 
    model = factor(
      model, 
      levels = c("Gemma-2-2B-IT", "Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.2", "Gemma-2-9B-IT")
    )
  )

logprobs %>% count(model)
# %>%
  # pivot_longer(c(logprob, controlled_logprob), names_to = "metric", values_to = "score")
  # group_by(model, sim_type, metric) %>%
  # mutate(
  #   score = (score - min(score))/(max(score) - min(score))
  # ) 

logprobs %>%
  ungroup() %>%
  group_by(model, sim_type) %>%
  mutate(
    yes = exp(yes),
    no = exp(no),
    rel_prob = yes/(yes + no)
  ) %>%
  ungroup() %>% View("detailed")
  group_by(model, sim_type, condition, hypernymy) %>%
  summarize(
    # ste = 1.96 * plotrix::std.error(diff),
    # logprob = mean(diff),
    ste = 1.96 * plotrix::std.error(rel_prob),
    logprob = mean(rel_prob)
  ) %>%
  ungroup() %>%
  # filter(metric == "logprob") %>%
  ggplot(aes(condition, logprob, color = sim_type, shape = sim_type, group = interaction(hypernymy, sim_type))) +
  geom_point(size = 2.5) + 
  geom_line() +
  geom_linerange(aes(ymin = logprob-ste, ymax = logprob+ste)) +
  # geom_hline(yintercept = 0, linetype="dashed") +
  geom_hline(yintercept = 0.5017, linetype = "dashed") +
  # facet_wrap(~model, scales="free_y", nrow = 1) +
  facet_wrap(~model, nrow = 1) +
  # scale_y_continuous(breaks = scales::pretty_breaks(5), labels = scales::label_number(accuracy = 0.1)) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(values = c("#7570b3", "#d95f02")) +
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    legend.position = "top",
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Conclusion Condition",
    y = "Relative Probability of 'Yes'",
    color = "Similarity Type",
    shape = "Similarity Type"
  )

ggsave("paper/tax-similarity-aggregate-rel-prob.pdf", width = 14.49, height = 4.35, dpi = 300, device=cairo_pdf)

logprobs %>%
  ungroup() %>%
  group_by(model, sim_type, condition, hypernymy) %>%
  summarize(
    ste = 1.96 * plotrix::std.error(logprob),
    logprob = mean(logprob)
  ) %>%
  ungroup() %>%
  # ggplot(aes(condition, logprob, color = sim_type, fill = sim_type, group = interaction(hypernymy, sim_type))) +
  ggplot(aes(condition, logprob, color = sim_type, fill = sim_type)) +
  geom_col(position = position_dodge(0.9), alpha = 0.7) +
  geom_linerange(aes(ymin = logprob-ste, ymax = logprob+ste), color = "black", position = position_dodge(0.9)) +
  geom_hline(yintercept = 0, linetype="dashed", linewidth = 1) +
  facet_wrap(~model, scales="free_y", nrow = 1) +
  scale_y_continuous(breaks = scales::pretty_breaks(7), labels = scales::label_number(accuracy = 0.5)) +
  scale_color_manual(values = c("#7570b3", "#d95f02"), aesthetics = c("color", "fill")) +
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    legend.position = "top",
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Conclusion Condition",
    y = "log P(Yes) - log P(No)",
    color = "Similarity",
    shape = "Similarity",
    fill = "Similarity"
  )

logprobs %>%
  ungroup() %>%
  group_by(model, sim_type) %>%
  mutate(
    yes = exp(yes),
    no = exp(no),
    rel_prob = yes/(yes + no)
  ) %>%
  ungroup() %>%
  group_by(model, sim_type, condition, hypernymy) %>%
  summarize(
    # ste = 1.96 * plotrix::std.error(diff),
    # logprob = mean(diff),
    ste = 1.96 * plotrix::std.error(rel_prob),
    logprob = mean(rel_prob)
  ) %>%
  ungroup() %>%
  # filter(metric == "logprob") %>%
  ggplot(aes(condition, logprob, color = sim_type, fill=sim_type, shape = sim_type, group = interaction(hypernymy, sim_type))) +
  geom_col(position = position_dodge(0.9), alpha = 0.7) +
  geom_linerange(aes(ymin = logprob-ste, ymax = logprob+ste), color = "black", position = position_dodge(0.9)) +
  # geom_hline(yintercept = 0, linetype="dashed") +
  geom_hline(yintercept = 0.5017, linetype = "dashed") +
  # facet_wrap(~model, scales="free_y", nrow = 1) +
  facet_wrap(~model, nrow = 1) +
  # scale_y_continuous(breaks = scales::pretty_breaks(5), labels = scales::label_number(accuracy = 0.1)) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(values = c("#7570b3", "#d95f02"), aesthetics = c("color", "fill")) +
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    legend.position = "top",
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Conclusion Condition",
    y = "Relative Probability of 'Yes'",
    color = "Similarity Type",
    shape = "Similarity Type",
    fill = "Similarity Type"
  )

ggsave("paper/tax-similarity-aggregate-rel-prob-col.pdf", width = 14.49, height = 4.35, dpi = 300, device=cairo_pdf)

logprobs %>%
  ungroup() %>%
  group_by(model, sim_type) %>%
  mutate(
    yes = exp(yes),
    no = exp(no),
    rel_prob = yes/(yes + no)
  ) %>%
  ungroup() %>%
  group_by(model, sim_type, condition, hypernymy) %>%
  summarize(
    # ste = 1.96 * plotrix::std.error(diff),
    # logprob = mean(diff),
    ste = 1.96 * plotrix::std.error(rel_prob),
    logprob = mean(rel_prob)
  ) %>%
  ungroup() %>%
  mutate(
    sim_type = factor(sim_type, levels = c("Sense", "SPOSE"), labels = c("Word-Sense", "SPoSE"))
  ) %>%
  # filter(metric == "logprob") %>%
  ggplot(aes(sim_type, logprob, color = condition, fill=condition)) +
  geom_col(position = position_dodge(0.5), alpha = 0.7, width = 0.5) +
  geom_linerange(aes(ymin = logprob-ste, ymax = logprob+ste), color = "black", position = position_dodge(0.5)) +
  # geom_hline(yintercept = 0, linetype="dashed") +
  geom_hline(yintercept = 0.5017, linetype = "dashed") +
  # facet_wrap(~model, scales="free_y", nrow = 1) +
  facet_wrap(~model, nrow = 1) +
  # scale_y_continuous(breaks = scales::pretty_breaks(5), labels = scales::label_number(accuracy = 0.1)) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(values = c("#dfc27d", "#a6611a", "#80cdc1", "#018571"), aesthetics = c("color", "fill")) +
  # scale_color_manual(values = c("#7570b3", "#d95f02"), aesthetics = c("color", "fill")) +
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    legend.position = "top",
    axis.text = element_text(color = "black"),
    axis.title.y = element_markdown(family="CMU Serif")
  ) +
  labs(
    x = "Similarity Type",
    # y = "P<sub>rel</sub>(Yes)",
    y = "<span style='font-family: Times'>Average </span><i>P</i><sub><p style='font-family: Inconsolata'>rel</p></sub>(<span style='font-family: Times'>Yes</span>)",
    color = "Conclusion Condition",
    shape = "Conclusion Condition",
    fill = "Conclusion Condition"
  )

ggsave("paper/tax-similarity-aggregate-rel-prob-col.pdf", width = 14.49, height = 4.35, dpi = 300, device=cairo_pdf)


# 1590 x 458


## scatterplots

logprobs %>%
  ggplot(aes(similarity_raw, logprob, shape = hypernymy, color = hypernymy)) +
  geom_point(alpha = 0.1, size = 3) +
  ggh4x::facet_grid2(sim_type ~ model, scales = "free", independent = "y") +
  geom_hline(yintercept = 0, linetype="dashed") +
  scale_color_manual(values = c("#bf812d", "#35978f")) +
  # scale_x_continuous(limits = c(0, 1)) +
  guides(colour = guide_legend(override.aes = list(alpha = 1, size = 3))) +
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    legend.position = "top",
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Similarity",
    y = "log P(Yes) - log P(No)",
    color = "Hypernymy",
    shape = "Hypernymy",
    fill = "Hypernymy"
  )

logprobs %>%
  group_by(model, sim_type, premise, hypernymy) %>%
  summarize(
    logprob = mean(logprob),
    similarity_raw = mean(similarity_raw)
  ) %>%
  ggplot(aes(similarity_raw, logprob, shape = hypernymy, color = hypernymy)) +
  geom_point(alpha = 0.5, size = 3) +
  ggh4x::facet_grid2(sim_type ~ model, scales = "free", independent = "all") +
  geom_hline(yintercept = 0, linetype="dashed") +
  scale_color_manual(values = c("#bf812d", "#35978f")) +
  # scale_x_continuous(limits = c(0, 1)) +
  guides(colour = guide_legend(override.aes = list(alpha = 1, size = 3))) +
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    legend.position = "top",
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Similarity",
    y = "log P(Yes) - log P(No)",
    color = "Hypernymy",
    shape = "Hypernymy",
    fill = "Hypernymy"
  )

plot_finegrained <- function(model_name, similarity) {
  logprobs %>%
    filter(model == model_name, sim_type == similarity) %>%
    ggplot(aes(similarity_raw, logprob, shape = hypernymy, color = hypernymy)) +
    geom_point(alpha = 0.4) +
    facet_wrap(~anchor) +
    # ggh4x::facet_grid2(sim_type ~ model, scales = "free", independent = "y") +
    geom_hline(yintercept = 0, linetype="dashed") +
    scale_color_manual(values = c("#bf812d", "#35978f")) +
    # scale_x_continuous(limits = c(0, 1)) +
    guides(colour = guide_legend(override.aes = list(alpha = 1, size = 3))) +
    theme_bw(base_size = 17, base_family = "Times") +
    theme(
      panel.grid = element_blank(),
      legend.position = "top",
      axis.text = element_text(color = "black")
    ) +
    labs(
      x = "Similarity",
      y = "log P(Yes) - log P(No)",
      color = "Hypernymy",
      shape = "Hypernymy",
      fill = "Hypernymy"
    )
}

# 1354 1310
plot_finegrained("Gemma-2-2B-it", "SPOSE")
plot_finegrained("Gemma-2-2B-it", "sense-based")
plot_finegrained("Gemma-2-9B-it", "SPOSE")
plot_finegrained("Gemma-2-9B-it", "sense-based")
plot_finegrained("Llama-3-8B-Instruct", "SPOSE")
plot_finegrained("Llama-3-8B-Instruct", "sense-based")
plot_finegrained("Mistral-7B-Instruct-v0.2", "SPOSE")
plot_finegrained("Mistral-7B-Instruct-v0.2", "sense-based")

# correlations

logprobs %>% 
  group_by(model, sim_type) %>%
  nest() %>%
  mutate(
    n = map_dbl(data, function(x) {nrow(x)}),
    logprob_cor = map_dbl(data, function(x) {
      cor(x$logprob, x$similarity_raw, method = "spearman")
    })
  ) %>% select(-data) %>% View()

logprobs %>% 
  group_by(model, hypernymy, sim_type) %>%
  nest() %>%
  mutate(
    n = map_dbl(data, function(x) {nrow(x)}),
    logprob_cor = map_dbl(data, function(x) {
      cor(x$logprob, x$similarity_raw, method = "spearman")
    }),
    # controlled_logprob_cor = map_dbl(data, function(x) {
    #   cor(x$controlled_logprob, x$similarity_raw, method = "spearman")
    # })
  ) %>% select(-data) %>% View()

logprobs %>% 
  group_by(model, anchor, sim_type) %>%
  nest() %>%
  mutate(
    n = map_dbl(data, function(x) {nrow(x)}),
    logprob_cor = map_dbl(data, function(x) {
      cor(x$logprob, x$similarity_raw, method = "spearman")
    })
  ) %>% select(-data) %>% View()

# distribution

logprobs %>% 
  group_by(model, anchor, sim_type) %>%
  nest() %>%
  mutate(
    n = map_dbl(data, function(x) {nrow(x)}),
    logprob_cor = map_dbl(data, function(x) {
      cor(x$logprob, x$similarity_raw, method = "spearman")
    })
  ) %>% select(-data) %>%
  ggplot(aes(logprob_cor, color = sim_type, fill = sim_type)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~model, nrow = 1) +
  scale_color_manual(values = c("#7570b3", "#d95f02"), aesthetics = c("color", "fill")) +
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    legend.position = "top",
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Correlation with Similarity across Premise Categories",
    y = "Density",
    color = "Similarity",
    fill = "Similarity"
  )

# 1403 473

# lmers

model_results <- logprobs %>%
  ungroup() %>%
  group_by(model, sim_type) %>%
  mutate(
    yes = exp(yes),
    no = exp(no),
    rel_prob = yes/(yes + no)
  ) %>%
  ungroup() %>%
  filter(model == "Gemma-2-9B-it", sim_type == "SPOSE") %>%
  mutate(
    hypernymy = case_when(
      hypernymy == "yes" ~ 1,
      TRUE ~ 0
    ),
    anchor = factor(anchor),
    similarity = similarity_raw
  )

fit1 <- lmer(rel_prob ~ hypernymy * similarity + (hypernymy * similarity | anchor), data = model_results)
summary(fit1)
fit2 <- lmer(rel_prob ~ hypernymy + similarity + (hypernymy + similarity | anchor), data = model_results)

fit3 <- lmer(logprob ~ hypernymy + (hypernymy | anchor), data = model_results)
fit4 <- lmer(logprob ~ similarity + (similarity | anchor), data = model_results)

# summary(fit1)
anova(fit1, fit2)
anova(fit1, fit3)
anova(fit1, fit4)



### Controls

results %>%
  filter(reasoning == "deduction", multi_property == TRUE) %>%
  mutate(
    condition = case_when(
      hypernymy == "yes" & similarity_binary == "high" ~ "Tax.\nHigh-Sim",
      hypernymy == "yes" & similarity_binary == "low" ~ "Tax.\nLow-Sim",
      hypernymy == "no" & similarity_binary == "low" ~ "Non-Tax.\nLow-Sim",
      TRUE ~ "Non-Tax.\nHigh-Sim",
    ),
    condition = factor(condition, levels = c("Non-Tax.\nLow-Sim", "Non-Tax.\nHigh-Sim", "Tax.\nLow-Sim", "Tax.\nHigh-Sim"))
  ) %>%
  mutate(
    # correctness = case_when(
    #   hypernymy == "yes" ~ diff > 0,
    #   TRUE ~ diff < 0
    # ),
    correctness = case_when(
      property_contrast == TRUE & diff < 0 ~ 1,
      property_contrast == TRUE & diff > 0 ~ 0,
      property_contrast == FALSE & hypernymy == "yes" ~ diff > 0,
      property_contrast == FALSE & hypernymy == "no" ~ diff < 0
    ),
    sim_type = case_when(
      str_detect(hyponym_type, "sense_based") ~ "sense-based",
      TRUE ~ "SPOSE"
    )
  ) %>%
  filter(anchor != hyponym) %>%
  group_by(model, sim_type, property_contrast) %>%
  summarize(
    sensitivity = mean(correctness)
  )

logprobs %>%
  filter(hypernymy == "yes") %>%
  distinct(anchor, hyponym) %>%
  count(anchor, sort=TRUE) %>%
  rename(premise = anchor, N = n) %>%
  write_csv("data/premise.csv")
