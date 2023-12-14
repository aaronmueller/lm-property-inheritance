# Concept Hierarchies
This repository contains code for observing and manipulating conceptual hierarchies in language models.

## TODOs
- [ ] Add COMPS data
	- [ ] Control data
	- [ ] Concept hierarchy data
	- [ ] Novel word acquisition data
- [ ] ICL baselines
- [x] Attribution patching code
- [ ] Circuit discovery code
- [ ] Circuit editing


## Packages

Pip installable ones:

```bash
transformers
semantic-memory
minicons # for tests?
```

## Stimuli generation

To generate stimuli, first download the data used to generate COMPS:

```bash
bash data/download_semantic_memory.sh
```

then run the generation script (`pigen.py` -- "property inheritance generator"):

```bash
python src/pigen.py
```

this will create a directory in data called `stimuli` with the following files:
```txt
pi_prompt.txt: incontext prompt with 8 examples containing property inheritance queries
real_prompt.txt: incontext prompt with 8 examples containing queries to test general property knowledge
prompt_metadata.jsonl: metadata (concepts and properties) used to generate icl prompts
test_metadata.jsonl: metadata (concepts and properties) used to generate test data
test_pi.jsonl: test data for property inheritance
test_real.jsonl: test data for general property knowledge
```