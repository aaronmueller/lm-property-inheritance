library(tidyverse)
library(fs)
library(ggtext)
library(ggfittext)
library(colorspace)
library(patchwork)

das_result_files <- dir_ls("src/results/", regexp = "*.csv", recurse = TRUE) %>%
  discard(str_detect(., "(trash|argmax-\\d|yn-\\d|portion|flip)"))

das_results_raw <- das_result_files %>%
  map_df(read_tsv, .id = "file", col_names = c("layer", "position", "score"))

das_results_raw %>%
  filter(str_detect(file, "iia-yn")) %>% View()

das_results <- das_results_raw %>%
  mutate(
    model = case_when(
      str_detect(file, "gemma-2-2b-it-variation-qa-2") ~ "ignore",
      str_detect(file, "gemma-2-2b") ~ "Gemma-2-2B-IT",
      str_detect(file, "gemma-2-9b") ~ "Gemma-2-9B-IT",
      str_detect(file, "Llama") ~ "Llama-3-8B-Instruct",
      str_detect(file, "mistral") ~ "Mistral-7B-Instruct-v0.2"
    ),
    experiment = case_when(
      str_detect(file, "balanced-balanced") & !str_detect(file, "control") ~ "Balanced",
      str_detect(file, "balanced-low-sim-pos") & !str_detect(file, "control") ~ "Unambiguous",
      str_detect(file, "control") ~ "Control",
      str_detect(file, "high-sim-pos-high-sim-pos") ~ "Ambiguous-Test",
      str_detect(file, "high-sim-pos-low-sim-pos") ~ "Ambiguous-Gen",
    ),
    similarity_type = case_when(
      str_detect(file, "spose") ~ "SPoSE",
      TRUE ~ "Word-Sense"
    ),
    measurement = case_when(
      str_detect(file, "argmax") ~ "argmax",
      str_detect(file, "yn_fix") ~ "yn_fix",
      TRUE ~ "yn"
    ),
    reversed = str_detect(file, "rev/"),
    layer = factor(layer),
    experiment = factor(experiment, levels = c("Balanced", "Control", "Ambiguous-Test", "Ambiguous-Gen", "Unambiguous")),
    position = str_replace(position, "_", "-"),
    position = factor(position, levels = rev(c("premise-first", "premise-last", "conclusion-first", "conclusion-last", "last")))
  ) %>%
  select(-file) %>%
  filter(str_detect(measurement, "yn"), model != "ignore") 

das_results %>% 
  count(model, experiment, measurement, similarity_type, reversed) %>% View()

gemma9_all <- das_results %>%
  filter(experiment != "Unambiguous") %>%
  filter(model == "Gemma-2-9B-IT", reversed == FALSE, measurement == "yn_fix") %>%
  mutate(
    text = case_when(
      score > 0.6 ~ format(signif(score, 2), nsmall = 2),
      TRUE ~ NA
    )
  ) %>%
  ggplot(aes(layer, position, color = score, fill = score, label = text)) +
  geom_tile() +
  geom_text(color = "white", family = "Times", size = 2.5) +
  # geom_fit_text(reflow = TRUE, grow = FALSE, contrast = FALSE, color = "white", family = "Times") +
  facet_grid(similarity_type ~ experiment) +
  # scale_fill_gradient(low = "white", high = "#c51b8a", aesthetics = c("color", "fill"), limits = c(0.46, 0.948), breaks = c(0.5, 0.6, 0.7, 0.8, 0.9)) +
  # scale_fill_gradient(low = "white", high = "#54278f", aesthetics = c("color", "fill"), limits = c(0.46, 0.948), breaks = c(0.5, 0.6, 0.7, 0.8, 0.9)) +
  scale_fill_distiller(palette = "Purples", aesthetics = c("color", "fill"), direction = 1) +
  # scale_fill_viridis_c(aesthetics = c("color", "fill")) +
  scale_x_discrete(expand = c(0,0)) +
  scale_y_discrete(expand = c(0,0)) +
  theme_bw(base_size = 15, base_family = "Times") +
  theme(
    axis.text = element_text(color = "black"),
    strip.text = element_text(color = "black")
  ) +
  labs(
    x = "Layer",
    y = "Position",
    fill = "IIA",
    color = "IIA",
    title = "Gemma-2-9B-IT"
  )

gemma9_all
ggsave("paper/gemma-2-9b-it-das-iia.pdf", width = 11.89, height = 4.41, dpi= 300, device=cairo_pdf)


## gemma 2b

gemma2_all <- das_results %>%
  filter(experiment != "Unambiguous") %>%
  filter(str_detect(model, "2B"), reversed == FALSE, measurement == "yn_fix") %>%
  mutate(
    text = case_when(
      score >= 0.55 ~ format(signif(score, 2), nsmall = 2),
      TRUE ~ NA
    )
  ) %>%
  ggplot(aes(layer, position, color = score, fill = score, label = text)) +
  geom_tile() +
  geom_text(color = "white", family = "Times", size = 2.5) +
  # geom_fit_text(reflow = TRUE, grow = FALSE, contrast = FALSE, color = "white", family = "Times") +
  facet_grid(similarity_type ~ experiment) +
  scale_color_continuous_sequential(palette = "Magenta", aesthetics = c("color", "fill"), limits = c(0.2, 1.0), breaks = c(0.2, 0.4, 0.6, 0.8, 1.0)) +
  # scale_fill_gradient(low = "white", high = "#c51b8a", aesthetics = c("color", "fill"), limits = c(0.46, 0.948), breaks = c(0.5, 0.6, 0.7, 0.8, 0.9)) +
  # scale_fill_gradient(low = "white", high = "#54278f", aesthetics = c("color", "fill"), limits = c(0.46, 0.948), breaks = c(0.5, 0.6, 0.7, 0.8, 0.9)) +
  # scale_fill_distiller(palette = "Purples", aesthetics = c("color", "fill"), direction = 1) +
  # scale_fill_viridis_c(aesthetics = c("color", "fill")) +
  scale_x_discrete(expand = c(0,0)) +
  scale_y_discrete(expand = c(0,0)) +
  theme_bw(base_size = 15, base_family = "Times") +
  theme(
    axis.text = element_text(color = "black"),
    strip.text = element_text(color = "black")
  ) +
  labs(
    x = "Layer",
    y = "Position",
    fill = "IIA",
    color = "IIA",
    title = "Gemma-2-2B-IT"
  )

gemma2_all

ggsave("paper/gemma-2-2b-it-das-iia.pdf", gemma2_all, width = 11.89, height = 4.41, dpi= 300, device=cairo_pdf)


## llama

llama_all <- das_results %>%
  filter(experiment != "Unambiguous") %>%
  filter(str_detect(model, "Llama"), reversed == FALSE, measurement == "yn_fix") %>%
  mutate(
    text = case_when(
      score > 0.53 ~ format(signif(score, 2), nsmall = 2),
      TRUE ~ NA
    )
  ) %>% 
  ggplot(aes(layer, position, color = score, fill = score, label = text)) +
  geom_tile() +
  geom_text(color = "white", family = "Times", size = 2.5) +
  # geom_fit_text(reflow = TRUE, grow = FALSE, contrast = FALSE, color = "white", family = "Times") +
  facet_grid(similarity_type ~ experiment) +
  scale_color_continuous_sequential(palette = "Mint", aesthetics = c("color", "fill"), limits = c(0.3, 1.0)) +
  # scale_fill_gradient(low = "white", high = "#c51b8a", aesthetics = c("color", "fill"), limits = c(0.46, 0.948), breaks = c(0.5, 0.6, 0.7, 0.8, 0.9)) +
  # scale_fill_gradient(low = "white", high = "#54278f", aesthetics = c("color", "fill"), limits = c(0.46, 0.948), breaks = c(0.5, 0.6, 0.7, 0.8, 0.9)) +
  # scale_fill_distiller(palette = "Purples", aesthetics = c("color", "fill"), direction = 1) +
  # scale_fill_viridis_c(aesthetics = c("color", "fill")) +
  scale_x_discrete(expand = c(0,0)) +
  scale_y_discrete(expand = c(0,0)) +
  theme_bw(base_size = 15, base_family = "Times") +
  theme(
    axis.text = element_text(color = "black"),
    strip.text = element_text(color = "black")
  ) +
  labs(
    x = "Layer",
    y = "Position",
    fill = "IIA",
    color = "IIA",
    title = "Llama-3-8B-Instruct"
  )

llama_all
ggsave("paper/llama-3-8b-instruct-das-iia.pdf", llama_all, width = 11.89, height = 4.41, dpi= 300, device=cairo_pdf)


## mistral

mistrall <- das_results %>%
  filter(experiment != "Unambiguous") %>%
  filter(str_detect(model, "Mistral"), reversed == FALSE) %>%
  mutate(
    text = case_when(
      score > 0.55 ~ format(signif(score, 2), nsmall = 2),
      TRUE ~ NA
    )
  ) %>% 
  ggplot(aes(layer, position, color = score, fill = score, label = text)) +
  geom_tile() +
  geom_text(color = "white", family = "Times", size = 2.5) +
  # geom_fit_text(reflow = TRUE, grow = FALSE, contrast = FALSE, color = "white", family = "Times") +
  facet_grid(similarity_type ~ experiment) +
  scale_color_continuous_sequential(palette = "Peach", aesthetics = c("color", "fill"), limits = c(0.4, 0.93)) +
  # scale_fill_gradient(low = "white", high = "#c51b8a", aesthetics = c("color", "fill"), limits = c(0.46, 0.948), breaks = c(0.5, 0.6, 0.7, 0.8, 0.9)) +
  # scale_fill_gradient(low = "white", high = "#54278f", aesthetics = c("color", "fill"), limits = c(0.46, 0.948), breaks = c(0.5, 0.6, 0.7, 0.8, 0.9)) +
  # scale_fill_distiller(palette = "Purples", aesthetics = c("color", "fill"), direction = 1) +
  # scale_fill_viridis_c(aesthetics = c("color", "fill")) +
  scale_x_discrete(expand = c(0,0)) +
  scale_y_discrete(expand = c(0,0)) +
  theme_bw(base_size = 15, base_family = "Times") +
  theme(
    axis.text = element_text(color = "black"),
    strip.text = element_text(color = "black")
  ) +
  labs(
    x = "Layer",
    y = "Position",
    fill = "IIA",
    color = "IIA",
    title = "Mistral-7B-Instruct-v0.2"
  )

mistrall
ggsave("paper/mistral-das-iia.pdf", mistrall, width = 11.89, height = 4.41, dpi= 300, device=cairo_pdf)


gemma2_all / mistrall / llama_all / gemma9_all

ggsave("paper/alllll-lms-das-iia.pdf", width = 12.26, height = 18.32, dpi=300, device=cairo_pdf())


# smol plots

gemma_smol <- das_results %>%
  filter(experiment != "Unambiguous") %>%
  filter(model == "Gemma-2-9B-IT", reversed == FALSE, measurement == "yn_fix") %>%
  filter(similarity_type == "SPoSE") %>%
  mutate(
    text = case_when(
      score > 0.6 ~ format(signif(score, 2), nsmall = 2),
      TRUE ~ NA
    )
  ) %>%
  ggplot(aes(layer, position, color = score, fill = score, label = text)) +
  geom_tile() +
  geom_text(color = "white", family = "Times", size = 3) +
  # geom_fit_text(reflow = TRUE, grow = FALSE, contrast = FALSE, color = "white", family = "Times") +
  # facet_grid(similarity_type ~ experiment) +
  facet_wrap(~ experiment, nrow = 1) +
  # scale_fill_gradient(low = "white", high = "#c51b8a", aesthetics = c("color", "fill"), limits = c(0.46, 0.948), breaks = c(0.5, 0.6, 0.7, 0.8, 0.9)) +
  # scale_fill_gradient(low = "white", high = "#54278f", aesthetics = c("color", "fill"), limits = c(0.46, 0.948), breaks = c(0.5, 0.6, 0.7, 0.8, 0.9)) +
  scale_fill_distiller(palette = "Purples", aesthetics = c("color", "fill"), direction = 1) +
  # scale_fill_viridis_c(aesthetics = c("color", "fill")) +
  scale_x_discrete(expand = c(0,0)) +
  scale_y_discrete(expand = c(0,0)) +
  theme_bw(base_size = 15, base_family = "Times") +
  theme(
    axis.text = element_text(color = "black"),
    strip.text = element_text(color = "black")
  ) +
  labs(
    x = "Layer",
    y = "Position",
    fill = "IIA",
    color = "IIA",
    title = "Gemma-2-9B-IT w/ SPoSE Similarity"
  )

gemma_smol

mistral_smol <- das_results %>%
  filter(experiment != "Unambiguous") %>%
  filter(str_detect(model, "Mistral"), reversed == FALSE, measurement == "yn") %>%
  filter(similarity_type == "SPoSE") %>%
  mutate(
    text = case_when(
      score > 0.55 ~ format(signif(score, 2), nsmall = 2),
      TRUE ~ NA
    )
  ) %>%
  ggplot(aes(layer, position, color = score, fill = score, label = text)) +
  geom_tile() +
  geom_text(color = "white", family = "Times", size = 3) +
  # geom_fit_text(reflow = TRUE, grow = FALSE, contrast = FALSE, color = "white", family = "Times") +
  # facet_grid(similarity_type ~ experiment) +
  facet_wrap(~ experiment, nrow = 1) +
  # scale_fill_gradient(low = "white", high = "#c51b8a", aesthetics = c("color", "fill"), limits = c(0.46, 0.948), breaks = c(0.5, 0.6, 0.7, 0.8, 0.9)) +
  # scale_fill_gradient(low = "white", high = "#54278f", aesthetics = c("color", "fill"), limits = c(0.46, 0.948), breaks = c(0.5, 0.6, 0.7, 0.8, 0.9)) +
  scale_color_continuous_sequential(palette = "Peach", aesthetics = c("color", "fill"), limits = c(0.4, 0.93)) +
  # scale_fill_viridis_c(aesthetics = c("color", "fill")) +
  scale_x_discrete(expand = c(0,0)) +
  scale_y_discrete(expand = c(0,0)) +
  theme_bw(base_size = 15, base_family = "Times") +
  theme(
    axis.text = element_text(color = "black"),
    strip.text = element_text(color = "black")
  ) +
  labs(
    x = "Layer",
    y = "Position",
    fill = "IIA",
    color = "IIA",
    title = "Mistral-7B-Instruct-v0.2 w/ SPoSE Similarity"
  )

gemma_smol / mistral_smol

ggsave("paper/gemma-mistral-spose-das-iia.pdf", height=6.13, width=12.26, dpi=300, device=cairo_pdf())



# balanced-unambiguous

gemma9_unambig <- das_results %>%
  filter(
    model == "Gemma-2-9B-IT", 
    reversed == FALSE, 
    measurement == "yn_fix", 
    experiment %in% c("Balanced", "Unambiguous")
  ) %>%
  mutate(
    text = case_when(
      score > 0.6 ~ format(signif(score, 2), nsmall = 2),
      TRUE ~ NA
    ),
    experiment2 = case_when(experiment == "Balanced" ~ "Bal", experiment == "Unambiguous" ~ "Gen")
  ) %>%
  ggplot(aes(layer, position, color = score, fill = score, label = text)) +
  geom_tile() +
  geom_text(color = "white", family = "Times", size = 2.5) +
  facet_grid(similarity_type ~ experiment2) +
  scale_fill_distiller(palette = "Purples", aesthetics = c("color", "fill"), direction = 1) +
  scale_x_discrete(expand = c(0,0)) +
  scale_y_discrete(expand = c(0,0)) +
  theme_bw(base_size = 15, base_family = "Times") +
  theme(
    axis.text = element_text(color = "black"),
    strip.text = element_text(color = "black")
  ) +
  labs(
    x = "Layer",
    y = "Position",
    fill = "IIA",
    color = "IIA",
    title = "Gemma-2-9B-IT"
  )


gemma2_unambig <- das_results %>%
  filter(str_detect(model, "2B"), reversed == FALSE, measurement == "yn_fix", 
         experiment %in% c("Balanced", "Unambiguous")) %>%
  mutate(
    text = case_when(
      score > 0.55 ~ format(signif(score, 2), nsmall = 2),
      TRUE ~ NA
    ),
    experiment2 = case_when(experiment == "Balanced" ~ "Bal", experiment == "Unambiguous" ~ "Gen")
  ) %>%
  ggplot(aes(layer, position, color = score, fill = score, label = text)) +
  geom_tile() +
  geom_text(color = "white", family = "Times", size = 2.5) +
  facet_grid(similarity_type ~ experiment2) +
  scale_color_continuous_sequential(palette = "Magenta", aesthetics = c("color", "fill"), limits = c(0.3, 1.0)) +
  scale_x_discrete(expand = c(0,0)) +
  scale_y_discrete(expand = c(0,0)) +
  theme_bw(base_size = 15, base_family = "Times") +
  theme(
    axis.text = element_text(color = "black"),
    strip.text = element_text(color = "black")
  ) +
  labs(
    x = "Layer",
    y = "Position",
    fill = "IIA",
    color = "IIA",
    title = "Gemma-2-2B-IT"
  )

llama_unambig <- das_results %>%
  filter(str_detect(model, "Llama"), reversed == FALSE, measurement == "yn_fix", 
         experiment %in% c("Balanced", "Unambiguous")) %>%
  mutate(
    text = case_when(
      score > 0.53 ~ format(signif(score, 2), nsmall = 2),
      TRUE ~ NA
    ),
    experiment2 = case_when(experiment == "Balanced" ~ "Bal", experiment == "Unambiguous" ~ "Gen")
  ) %>% 
  ggplot(aes(layer, position, color = score, fill = score, label = text)) +
  geom_tile() +
  geom_text(color = "white", family = "Times", size = 2.5) +
  facet_grid(similarity_type ~ experiment2) +
  scale_color_continuous_sequential(palette = "Mint", aesthetics = c("color", "fill"), limits = c(0.38, 1.0)) +
  scale_x_discrete(expand = c(0,0)) +
  scale_y_discrete(expand = c(0,0)) +
  theme_bw(base_size = 15, base_family = "Times") +
  theme(
    axis.text = element_text(color = "black"),
    strip.text = element_text(color = "black")
  ) +
  labs(
    x = "Layer",
    y = "Position",
    fill = "IIA",
    color = "IIA",
    title = "Llama-3-8B-Instruct"
  )


mistral_unambig <- das_results %>%
  filter(str_detect(model, "Mistral"), reversed == FALSE, 
         experiment %in% c("Balanced", "Unambiguous")) %>%
  mutate(
    text = case_when(
      score > 0.60 ~ format(signif(score, 2), nsmall = 2),
      TRUE ~ NA
    ),
    experiment2 = case_when(experiment == "Balanced" ~ "Bal", experiment == "Unambiguous" ~ "Gen")
  ) %>%
  ggplot(aes(layer, position, color = score, fill = score, label = text)) +
  geom_tile() +
  geom_text(color = "white", family = "Times", size = 2.5) +
  # geom_fit_text(reflow = TRUE, grow = FALSE, contrast = FALSE, color = "white", family = "Times") +
  facet_grid(similarity_type ~ experiment2) +
  scale_color_continuous_sequential(palette = "Peach", aesthetics = c("color", "fill"), limits = c(0.45, 0.9)) +
  scale_x_discrete(expand = c(0,0)) +
  scale_y_discrete(expand = c(0,0)) +
  theme_bw(base_size = 15, base_family = "Times") +
  theme(
    axis.text = element_text(color = "black"),
    strip.text = element_text(color = "black")
  ) +
  labs(
    x = "Layer",
    y = "Position",
    fill = "IIA",
    color = "IIA",
    title = "Mistral-7B-Instruct-v0.2"
  )


(gemma2_unambig + mistral_unambig) / (llama_unambig + gemma9_unambig)

ggsave("paper/unambiguous_das_results.pdf", width = 12.15, height = 7.80, dpi = 300, device=cairo_pdf())
