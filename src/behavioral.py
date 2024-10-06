import pandas as pd
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from torch.utils.data import DataLoader
import csv
import torch
from tqdm import tqdm
import sklearn.metrics

import utils
import prompt
import config
import lexicon

LEMMA_PATH = "../data/things/things-lemmas-annotated.csv"

def load_model(model_name, device):
    #_config, tokenizer, model = create_mistral()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    _ = model.to(device)  # single gpu
    _ = model.eval()  # always no grad on the model
    return model, tokenizer

def find_sublist_indices(L, subL):
    len_L = len(L)
    len_subL = len(subL)

    # Iterate through L with a sliding window of size len_subL
    for i in range(len_L - len_subL + 1):
        if L[i:i + len_subL] == subL:
            return list(range(i, i + len_subL))
    return []


def load_concepts():
    # load all unique concepts
    concepts = defaultdict(lexicon.Concept)
    with open(LEMMA_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["remove"] != "1":
                concepts[row["lemma"]] = utils.lemma2concept(row)
    concepts = dict(concepts)

    prop = lexicon.Property(
        property_name="daxable",
        singular="is daxable",
        plural="are daxable",
    )
    return prop, concepts

def make_inputs_labels_dataset(input_ids, labels, attention_masks, batch_size):
    prealign_dataset = Dataset.from_dict(
        {"input_ids": input_ids,
         "labels": labels,
         "attention_mask": attention_masks
         } )
    prealign_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    prealign_dataloader = DataLoader(prealign_dataset, batch_size=batch_size)
    return prealign_dataloader

def flatten(L): return [u for i in L for u in i]


def icl_accuracy(dataloader, model, tokenizer, yes_label, no_label):
    total_count = 0
    correct_count = 0
    preds = []
    gold = []
    yes_over_no = []
    Yes_id = tokenizer.convert_tokens_to_ids(yes_label)
    No_id = tokenizer.convert_tokens_to_ids(no_label)

    with torch.no_grad():
        for step, inputs in enumerate(tqdm(dataloader)):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(model.device)
                    #attention_mask = inputs['attention_mask'].to(model.device)
            # aligning forward!
            outputs = model(
                input_ids=inputs["input_ids"],
                labels=inputs["labels"],
                attention_mask=inputs['attention_mask']
            )

            #logprobs  = torch.nn.functional.log_softmax(outputs.logits[:, -1, :], dim=-1)

            actual_test_labels = inputs["labels"][:, -1]
            pred_test_labels = torch.argmax(outputs.logits[:, -1], dim=-1)

            yes_greater_than_no = outputs.logits[:, -1,  Yes_id] >=  outputs.logits[:, -1,  No_id]
            yes_greater_than_no = yes_greater_than_no.tolist()
            yes_over_no.extend(yes_greater_than_no)

            #preds.append([tokenizer.decode(i) for i in pred_test_labels])
            #gold.append([tokenizer.decode(i) for i in actual_test_labels])
            preds.append([i.tolist() for i in pred_test_labels])
            gold.append([i.tolist() for i in actual_test_labels])

            correct_labels = actual_test_labels == pred_test_labels

            total_count += len(correct_labels)
            correct_count += correct_labels.sum().tolist()
    current_acc = round(correct_count / total_count, 2)
    print(f"[WARNING: THIS NEEDS TO BE GOOD!] prealign task accuracy: (argmax style) {current_acc}")

    gold = flatten(gold)
    preds = flatten(preds)

    gold = [tokenizer.decode(i) for i in gold]
    preds = [tokenizer.decode(i) for i in preds]
    print("SET OF GOLD LABELS: ",  Counter(gold))
    print("SET OF PREDICTED LABELS (next token, argmax): ",  Counter(preds))

    a2 = sklearn.metrics.accuracy_score([i.strip() for i in gold], preds)
    print("Accuracy (argmax) if we remove space from gold labels: ", a2)

    yes_or_no_according_to_logitdiff = ['Yes' if i else 'No' for i in yes_over_no]
    a3 = sklearn.metrics.accuracy_score([i.strip() for i in gold], yes_or_no_according_to_logitdiff)
    print("Accuracy according to logit diff between 'yes' and 'no' token: ", a3)

    return current_acc, gold, preds, yes_over_no


def load_data(csv_filepath,
               tokenizer,
               shuffle_seed=42,
               data_fraction=.05,
               prompt_config='initial-qa',
               flip_pair=False,
               chat_style=True):

    if hasattr(tokenizer, 'name_or_path') and tokenizer.name_or_path in ['mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Llama-3.1-8B-Instruct']:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif tokenizer.name_or_path in ['google/gemma-2-2b-it', 'google/gemma-2-9b-it']:
        print("token id: ", tokenizer.pad_token_id)
    else:
        raise ValueError("Not implemented_config: padding tokens for other models besides mistral-7b-instruct-v.0.2")

    template_config = config.PROMPTS[prompt_config]
    prompt_template = prompt.Prompt(template=template_config["template"],
                                    zero_shot=template_config["zero_shot"])

    prop, concepts = load_concepts()

    df = pd.read_csv(csv_filepath)
    # BUT we really should filter out where the premise and conclusion are same (14 cases)
    df = df[df['premise'] != df['conclusion']]

    # Shuffle and optional sampling
    dfshuffled = df.sample(frac=data_fraction, random_state=shuffle_seed).reset_index(drop=True)
    print("Total dataset size: ", len(dfshuffled))

#    if tokenizer.name_or_path in ['google/gemma-2-2b-it', 'google/gemma-2-9b-it', 'meta-llama/Llama-3.1-8B-Instruct']:
#        chat_format = tokenizer #This is to feed into `create_stimulus`
#    else:
        # use prompt as is without applying `apply_chat_template`.
#        chat_format = None
#    chat_format = None
    if chat_style:
        chat_format = tokenizer #This is to feed into `create_stimulus`
    else:
        chat_format = None

    #NOTE the 'Yes' and 'No' tokens might be '▁Yes' so do this to check!
    example_prompt = prompt_template.create_stimulus(premise = concepts[dfshuffled['conclusion'].values[0]], conclusion=concepts[dfshuffled['premise'].values[0]], prop=prop, tokenizer=chat_format)
    yes_token = tokenizer.convert_ids_to_tokens(tokenizer.encode(example_prompt + " " + "Yes")[-1])
    no_token = tokenizer.convert_ids_to_tokens(tokenizer.encode(example_prompt + " " + "No")[-1])
    print("Yes token: ", yes_token)
    print("No token: ", no_token)

    #TEMPORARY TODO
    #yes_token = "Yes"
    #no_token = "No"

    map_yn = {'yes': yes_token,
              'no': no_token}

    samples = []
    for index, row in dfshuffled.iterrows():
        if flip_pair:
            # flip the concepts used for premise/conclusion. Due to the way the csv is built, label all as No.
            samples.append(
                    {"conclusion": row['premise_form'],
                     "premise": row['conclusion_form'],
                     "is_hyper": row['hypernymy'], #this now means there is a hypo relationship
                     "sim": row['similarity_binary'],
                     "prompt": prompt_template.create_stimulus(premise = concepts[row['conclusion']], conclusion=concepts[row['premise']], prop=prop, tokenizer=chat_format),
                     "label": no_token#'No'
                     } )
        else:
            samples.append(
                    {"conclusion": row['conclusion_form'],
                     "premise": row['premise_form'],
                     "is_hyper": row['hypernymy'],
                     "sim": row['similarity_binary'],
                     "prompt": prompt_template.create_stimulus(premise = concepts[row['premise']], conclusion=concepts[row['conclusion']], prop=prop, tokenizer=chat_format),# + " ",
                     "label": map_yn[row['hypernymy']] #NOTE same as is_hyper for now... may change later!
                     #"label": row['hypernymy'].capitalize() #NOTE same as is_hyper for now... may change later!
                     } )

    def get_token_inds(tokens, premise, conclusion):

        """ Get token indices for the premise and the conclusion.

        Careful, I assume the anchor and conclusion words only appear once in the string.
        """
        # NOTE: it is index 1 for first token because first index is <s> token in mistral;
        # generalize later for other tokenizers if needed.
        # tokens = tokenizer(example).input_ids
        anchor_tokens = tokenizer(premise).input_ids[1:]
        concl_tokens  = tokenizer(conclusion).input_ids[1:]

        anchor_tokens_pos = find_sublist_indices(tokens.tolist(), anchor_tokens)
        concl_tokens_pos = find_sublist_indices(tokens.tolist(), concl_tokens)

        #Make robust to tokenizers like Gemma -- put space in front:
        if anchor_tokens_pos == []:
            anchor_tokens = tokenizer(" "+premise).input_ids[1:]
            anchor_tokens_pos = find_sublist_indices(tokens.tolist(), anchor_tokens)
        if concl_tokens_pos == []:
            concl_tokens = tokenizer(" "+conclusion).input_ids[1:]
            concl_tokens_pos = find_sublist_indices(tokens.tolist(), concl_tokens)

        assert len(anchor_tokens_pos)!=0
        assert len(concl_tokens_pos) !=0


        first_tok_anchor, last_tok_anchor = anchor_tokens_pos[0], anchor_tokens_pos[-1]
        first_tok_concl, last_tok_concl = concl_tokens_pos[0], concl_tokens_pos[-1]

        return (first_tok_anchor, last_tok_anchor), (first_tok_concl, last_tok_concl)

    if tokenizer.name_or_path in ['google/gemma-2-2b-it', 'google/gemma-2-9b-it', 'meta-llama/Llama-3.1-8B-Instruct']:
        inputs = tokenizer([i['prompt'] for i in samples], return_tensors='pt', padding=True, add_special_tokens=False)
    elif tokenizer.name_or_path in ['meta-llama/Llama-3.1-8B-Instruct']:
        tokenizer.padding_side = 'left'
        print(tokenizer.padding_side)
        inputs = tokenizer([i['prompt'] for i in samples], return_tensors='pt', padding=True, add_special_tokens=False)
    else:
        #TODO double-check this is fine with mistral
        inputs = tokenizer([i['prompt'] for i in samples], return_tensors='pt', padding=True)

    input_ids = inputs.input_ids
    attention_masks = inputs.attention_mask

    label_ids = [tokenizer.convert_tokens_to_ids(s['label']) for s in samples]
    output_ids = (torch.ones(input_ids.shape) * -100).long()
    output_ids[:, -1 ] = torch.tensor(label_ids)

    positions = [get_token_inds(input_id, samples[ii]['premise'], samples[ii]['conclusion']) for ii,input_id in enumerate(input_ids)]

    for ii,s in enumerate(samples):
        s['position'] = positions[ii]
        s['input_ids'] = input_ids[ii]
        s['output_ids'] = output_ids[ii]
        s['attention_masks'] = attention_masks[ii]

    return samples, yes_token, no_token

if __name__ == "__main__":

    csv_filepath = "../data/things/stimuli-pairs/things-inheritance-sense_based_sim-pairs.csv"
    model_name = "google/gemma-2-2b-it"
    device = "cuda:0"

    prompt_config = "variation-qa-2"
    chat_style=True
    batch_size = 8 # batch size for inference
    num_samples = None #100 #None

    model, tokenizer = load_model(model_name, device)

    samples, yes_token, no_token  = load_data(csv_filepath, tokenizer,
                                    data_fraction=1,
                                    prompt_config=prompt_config,
                                    chat_style=chat_style)
    if num_samples is None:
        test = samples[:]
    else:
        test = samples[:num_samples]

    prealign_dataloader_test = make_inputs_labels_dataset([i['input_ids'] for i in test],
                                                          [i['output_ids'] for i in test],
                                                          [i['attention_masks'] for i in test],
                                                          batch_size)

    a, gold, pred, yn = icl_accuracy(prealign_dataloader_test, model, tokenizer,  yes_token, no_token)


