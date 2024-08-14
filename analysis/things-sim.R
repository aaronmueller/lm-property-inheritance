library(R.matlab)
library(tidyverse)

spose_similarity <- readMat("data/things/spose_similarity.mat")
spose_rows <- read_file("data/things/unique_id.txt") %>% str_trim() %>% str_split("\n") %>% .[[1]]

spose_similarity$spose.sim[1,]

dim(spose_similarity$spose.sim)

concepts <- enframe(spose_rows) %>%
  rename(concept_id = name, concept = value)


things_sim <- data.frame(spose_similarity$spose.sim, row.names = spose_rows) %>%
  rownames_to_column() %>%
  tibble() %>%
  rename(concept1 = rowname) %>%
  pivot_longer(X1:X1854, names_to = "concept_id", values_to = "similarity") %>%
  mutate(concept_id = as.integer(str_remove(concept_id, "X"))) %>%
  inner_join(concepts %>% select(concept_id, concept2 = concept))

concepts %>%
  filter(str_detect(concept, "\\d"))
