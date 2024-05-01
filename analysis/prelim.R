library(tidyverse)
library(fs)
library(glue)
library(broom)
library(lmerTest)

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
  select(-file) %>%
  group_by(model, hyponym_type, setting, reasoning) %>%
  mutate(id = row_number()) %>%
  ungroup() %>%
  inner_join(bind_rows(
    taxonomic_triples %>% select(id, anchor, hyponym, hyponym_type),
    model_based_triples %>% select(id, anchor, hyponym, hyponym_type),
    sense_based_triples %>% select(id, anchor, hyponym, hyponym_type)
  ))

raw_results %>%
  mutate(
    hyponym_type = factor(
      hyponym_type, 
      levels = c("taxonomic", "sense_based_ns", "model_specific_ns"),
      labels = c("Taxonomic", "Sense NS", "Model NS")
    )
  ) %>%
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
    panel.grid = element_blank(),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Hyponym Type",
    y = "Measure",
    color = "Reasoning",
    shape = "Reasoning"
  )



# sense_based 
raw_results %>% 
  filter(hyponym_type != "model_specific_ns") %>%
  inner_join(sense_based_sim %>% rename(hyponym = concept)) %>%
  group_by(model, setting, reasoning) %>%
  # group_by(model, setting, reasoning, anchor) %>%
  nest() %>%
  mutate(
    cors = map(data, function(x) {
      cor.test(x$prompt, x$similarity, method = "spearman") %>%
        tidy()
    })
  ) %>%
  select(-data) %>%
  unnest(cors) %>%
  ungroup() %>%
  select(reasoning, setting, anchor, estimate, p.value)

raw_results %>% 
  filter(hyponym_type != "sense_based_ns") %>%
  inner_join(model_based_sim %>% rename(hyponym = concept)) %>%
  group_by(model, setting, reasoning, anchor) %>%
  nest() %>%
  mutate(
    cors = map(data, function(x) {
      cor.test(x$prompt, x$similarity, method = "pearson") %>%
        tidy()
    })
  ) %>%
  select(-data) %>%
  unnest(cors) %>%
  ungroup() %>%
  select(reasoning, setting, anchor, estimate, p.value) %>% View()



sense_based_results <- raw_results %>% 
  filter(hyponym_type != "model_specific_ns") %>%
  inner_join(sense_based_sim %>% rename(hyponym = concept)) %>%
  mutate(
    taxonomic = case_when(
      hyponym_type == "taxonomic" ~ 1,
      TRUE ~ -1
    ),
    anchor = factor(anchor)
  )

model_based_results <- raw_results %>% 
  filter(hyponym_type != "sense_based_ns") %>%
  inner_join(model_based_sim %>% rename(hyponym = concept)) %>%
  mutate(
    taxonomic = case_when(
      hyponym_type == "taxonomic" ~ 1,
      TRUE ~ -1
    ),
    anchor = factor(anchor)
  )


# deduction lmer

fit_deduction_phrasal <- lmer(prompt ~ taxonomic * similarity + (taxonomic * similarity|anchor),
                              data = sense_based_results %>% 
                                filter(setting=="Phrasal", reasoning=="deduction"), REML=FALSE)

summary(fit_deduction_phrasal)


## high-low sim comparison


sense_based_results %>%
  filter(reasoning == "induction", str_detect(setting, "QA")) %>%
  group_by(model, hyponym_type, anchor) %>%
  mutate(
    sim_class = case_when(
      similarity >= median(similarity) ~ "high_sim",
      TRUE ~ "low_sim"
    ),
    class = case_when(
      taxonomic == 1 ~ glue::glue("taxonomic_{sim_class}"),
      TRUE ~ glue::glue("non_taxonomic_{sim_class}")
    )
  ) %>%
  ungroup() %>%
  select(anchor, hyponym, class) %>%
  write_csv("data/things/things-sense-based-pairs.csv")

bind_rows(
  sense_based_results %>% mutate(sim_type = "Sense-based Sim"),
  model_based_results %>% mutate(sim_type = "Model-based Sim")
) %>% View()
  group_by(model, reasoning, setting, hyponym_type, sim_type) %>%
  summarize(
    sim = mean(similarity)
  ) %>% View()


bind_rows(
  sense_based_results %>% mutate(sim_type = "Sense-based Sim"),
  model_based_results %>% mutate(sim_type = "Model-based Sim")
) %>% View()
  group_by(model, reasoning, setting, hyponym_type, anchor, sim_type) %>%
  mutate(
    sim_class = case_when(
      similarity >= median(similarity) ~ "high_sim",
      TRUE ~ "low_sim"
    ),
    class = case_when(
      taxonomic == 1 ~ glue::glue("taxonomic\n{sim_class}"),
      TRUE ~ glue::glue("NS\n{sim_class}")
    ),
    class = factor(
      class, 
      levels = c("taxonomic\nhigh_sim", "taxonomic\nlow_sim", "NS\nhigh_sim", "NS\nlow_sim"),
      labels = c("Taxonomic\nHigh Sim", "Taxonomic\nLow Sim", "NS\nHigh Sim", "NS\nLow Sim")
    )
  ) %>%
  ungroup() %>%
  group_by(model, hyponym_type, setting, reasoning, sim_type, sim_class, class) %>%
  summarize(
    n = n(),
    ste = 1.96 * plotrix::std.error(prompt),
    logprob = mean(prompt)
  ) %>%
  ggplot(aes(class, logprob, color = reasoning, shape = sim_type, group = reasoning)) +
  geom_point(size = 2.5) + 
  geom_line() +
  geom_linerange(aes(ymin = logprob-ste, ymax = logprob+ste)) +
  scale_color_brewer(palette = "Dark2") +
  # facet_wrap(~ setting + hyponym_type, scales = "free_y")
  ggh4x::facet_grid2(sim_type ~ setting, scales = "free_y", independent = "y") +
  theme_bw(base_size=16, base_family="Times") +
  theme(
    legend.position = "top",
    panel.grid = element_blank(),
    axis.text = element_text(color = "black")
  ) +
  guides(shape="none") +
  labs(
    x = "Conclusion Concept Class",
    y = "Measure",
    color = "Reasoning"
  )
  

