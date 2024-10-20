import os
import sklearn.metrics
import argparse
import sys
import pandas as pd
import csv
import torch
# for plotting results
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm, trange
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from collections import defaultdict, Counter, namedtuple

from transformers import AutoTokenizer, AutoModelForCausalLM


pyvene_dir = os.getenv('PYVENE')
sys.path.append(pyvene_dir)
import pyvene
from pyvene import (
    IntervenableModel,
    BoundlessRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from pyvene import create_llama, create_mistral
from pyvene import set_seed, count_parameters
#TODO set_seed is unused

import config
import lexicon
import utils
import prompt
from utils_bdas import find_sublist_indices, save_results, load_results, plot_heatmap

#cuda_gpu = "1"
LEMMA_PATH = "../data/things/things-lemmas-annotated.csv"

#topdir = 'models'
#csv_filepath = "../data/things/stimuli-pairs/things-inheritance-sense_based_sim-pairs.csv"

#topdir = 'models-spose'
#csv_filepath = "../data/things/stimuli-pairs/things-inheritance-SPOSE_prototype_sim-pairs.csv"

def load_model(model_name, cuda_gpu):
    #_config, tokenizer, model = create_mistral()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == "google/gemma-2-9b-it":
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    _ = model.to("cuda:"+cuda_gpu)  # single gpu
    _ = model.eval()  # always no grad on the model
    return model, tokenizer
#TODO check this ok



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


def load_data(csv_filepath,
               tokenizer,
               shuffle_seed=42,
               data_fraction=.05,
               prompt_config='initial-qa',
               flip_pair=False,
               chat_style=True,
               yes_token=None,
               no_token=None):

    if hasattr(tokenizer, 'name_or_path') and tokenizer.name_or_path in ['mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Meta-Llama-3-8B-Instruct']:
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


    if chat_style:
        chat_format = tokenizer #This is to feed into `create_stimulus`
    else:
        chat_format = None

#    if tokenizer.name_or_path in ['google/gemma-2-2b-it', 'google/gemma-2-9b-it', 'meta-llama/Meta-Llama-3-8B-Instruct']:
#        chat_format = tokenizer #This is to feed into `create_stimulus`
#    else:
#        # use prompt as is without applying `apply_chat_template`.
#        chat_format = None
#    chat_format = None

    #NOTE the 'Yes' and 'No' tokens might be '▁Yes' so do this to check!
#    dfshuffled['conclusion'].values[0]
#    example_prompt = prompt_template.create_stimulus(premise = concepts[dfshuffled['conclusion'].values[0]], conclusion=concepts[dfshuffled['premise'].values[0]], prop=prop, tokenizer=chat_format)
#    yes_token = tokenizer.convert_ids_to_tokens(tokenizer.encode(example_prompt + " " + "Yes")[-1])
#    no_token = tokenizer.convert_ids_to_tokens(tokenizer.encode(example_prompt + " " + "No")[-1])

    if yes_token is None and no_token is None:
        #sensible defaults, checked
        if tokenizer.name_or_path == 'meta-llama/Meta-Llama-3-8B-Instruct':
            yes_token = "ĠYes"
            no_token = "ĠNo"
        else:
            yes_token = "Yes"
            no_token = "No"
    print("Yes token: ", yes_token)
    print("No token: ", no_token)

    map_yn = {'yes': yes_token,
              'no': no_token}

    samples = []
    for index, row in dfshuffled.iterrows():
        #These are the "lemma" versions, only need them to make the prompt
        #_cl = row['conclusion']
        #_pl = row['premise']
        if flip_pair:
            # flip the concepts used for premise/conclusion. Due to the way the csv is built, label all as No.
            samples.append(
                    {"conclusion": row['premise_form'],
                     "premise": row['conclusion_form'],
                     "is_hyper": row['hypernymy'], #this now means there is a hypo relationship
                     "sim": row['similarity_binary'],
                     "prompt": prompt_template.create_stimulus(premise = concepts[row['conclusion']], conclusion=concepts[row['premise']], prop=prop, tokenizer=chat_format),
                     "label": map_yn[row['hypernymy']] #yes_token#'No'
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

    if tokenizer.name_or_path in ['google/gemma-2-9b-it']:
        inputs = tokenizer([i['prompt'] for i in samples], return_tensors='pt', padding=True, add_special_tokens=False)
    elif tokenizer.name_or_path in ['meta-llama/Meta-Llama-3-8B-Instruct', 'google/gemma-2-2b-it']:
        tokenizer.padding_side = 'left'
        print(tokenizer.padding_side)
        inputs = tokenizer([i['prompt'] for i in samples], return_tensors='pt', padding=True, add_special_tokens=True)
    else:
        #TODO double-check this is fine with mistral
        inputs = tokenizer([i['prompt'] for i in samples], return_tensors='pt', padding=True)

    #inputs = tokenizer(prompts, return_tensors='pt', padding=True)
    input_ids = inputs.input_ids
    #
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

    return samples, yes_token, no_token#, input_ids


def make_inputs_labels_dataset(input_ids, labels, attention_masks):
    prealign_dataset = Dataset.from_dict(
        {"input_ids": input_ids,
         "labels": labels,
         "attention_mask": attention_masks
         } )
    prealign_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    prealign_dataloader = DataLoader(prealign_dataset, batch_size=8)
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
    print(f"[WARNING: THIS NEEDS TO BE GOOD!] prealign task accuracy: {current_acc}")


    gold = flatten(gold)
    preds = flatten(preds)
    gold = [tokenizer.decode(i) for i in gold]
    preds = [tokenizer.decode(i) for i in preds]

    yes_or_no_according_to_logitdiff = ['Yes' if i else 'No' for i in yes_over_no]
    a3 = sklearn.metrics.accuracy_score([i.strip() for i in gold], yes_or_no_according_to_logitdiff)
    print("Accuracy according to logit diff between 'yes' and 'no' token: ", a3)


    return current_acc, gold, preds, yes_over_no


def keep_condition(sim, taxo, mode):
    if mode=='balanced':
        return True # no filtering, train on all
    if mode=='high-sim-pos': # i.e., same as 'ambiguous' setting   # positive class is taxo & high sim; negative class is non-taxo & low-sim; other two combs removed.
        return (sim=='high' and taxo=='yes') or (sim=='low' and taxo=='no')
    if mode=='low-sim-pos': # i.e., all combinations not covered in 'ambiguous' setting
        return (sim=='low' and taxo=='yes') or (sim=='high' and taxo=='no')
    if mode=='only-taxo':
        return taxo=='yes'


def format_source_base(samples, mode, sample_with_replacement=False):
    """
    e.g., all samples for train regardless of whether high sim, taxonomic etc, "balanced"
    Ltrain  = format_source_base(train, 'balanced')

    say we want test set which is high sim only for positive (taxonomic) examples:
    L  = format_source_base(test, 'high-sim-pos')

    Returns
    -------

    List of namedtuples; each contains a base and source example.
    """
    #Filter samples according to criteria:
    samples = [s for s in samples if keep_condition(s['sim'], s['is_hyper'], mode)]
    #NOTE for now use same filtering mode for source as for base.

    #Now, sample from WITHIN the filtered samples, to get source examples for intervention.
    if sample_with_replacement:
        random.seed(42)
        source = [random.choice(samples) for _ in range(len(samples))]
    else:
        random.seed(42)
        source = random.sample(samples, len(samples))

    Pair = namedtuple('Pair', ['base', 'source'])
    L = [Pair(samples[ii], source[ii]) for ii in range(len(source))]
    return L #list(zip(samples, source))


def _get_pos(i, relative_pos, offset, input_ids):
    # position i
    if relative_pos=='conclusion_last':
        pos = i[1][1] + offset

    if relative_pos=='conclusion_first': #first token
        pos = i[1][0] + offset

    if relative_pos=='premise_last':
        pos = i[0][1] + offset

    if relative_pos=='premise_first': #first token
        pos = i[0][0] + offset

    if relative_pos == 'last':
        if offset >0:
            raise ValueError("if testing last token, offset must be <= 0")
        #NOTE this is ok because I padded source and base to same length but generally be careful here
        pos = input_ids.shape[-1] - 1

    return pos


def create_dataset_for_intervention(samples, relative_pos, offset, filter_mode, batch_size):
    sb_pairs  = format_source_base(samples, filter_mode)

    dataset = Dataset.from_dict(
        {
            "input_ids": [i.base['input_ids'].tolist() for i in sb_pairs], # first group is "base"  input example text, tokenized,
            "source_input_ids": [i.source['input_ids'].tolist() for i in sb_pairs], # second group is source input example text, tokenized (intervention)
            "attention_mask": [i.base['attention_masks'].tolist() for i in sb_pairs], # first group is "base"  input example text, tokenized,
            "source_attention_mask": [i.source['attention_masks'].tolist() for i in sb_pairs], # second group is source input example text, tokenized (intervention)
            "labels": [i.source['output_ids'].tolist() for i in sb_pairs], # labels ([-100, -100 .... yes-or-no-id ] ) <- counterfactual.]
            "intervention_ids": [0]*len(sb_pairs) ,  # we will not use this field
            "source_pos": [_get_pos(i.source['position'], relative_pos, offset, i.source['input_ids']) for i in sb_pairs],
            "base_pos": [_get_pos(i.base['position'], relative_pos, offset, i.base['input_ids']) for i in sb_pairs]
        }
    ).with_format("torch")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
    )

    return dataloader


def simple_boundless_das_position_config(model_type, intervention_type, layer):
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer,              # layer
                intervention_type,  # intervention type
            ),
        ],
        intervention_types=BoundlessRotatedSpaceIntervention,
    )
    return config

def compute_metrics(eval_preds, eval_labels, Yes_id, No_id):
    total_count = 0
    correct_count = 0
    correct_count_yn = 0
    #TODO
    gold_labels_all = []
    pred_labels_all = []

    #not against max output, but rather if p(yes) > p(no) matches ground truth.
    yes_over_no = []

    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        actual_test_labels = eval_label[:, -1]
        pred_test_labels = torch.argmax(eval_pred[:, -1], dim=-1)
        correct_labels = actual_test_labels == pred_test_labels

        yes_greater_than_no = eval_pred[:, -1,  Yes_id] >=  eval_pred[:, -1,  No_id]
        yes_greater_than_no = yes_greater_than_no#.tolist()
        #yes_over_no.extend(yes_greater_than_no)
        # convert True and False to Yes_id and No_id:
        pred_test_labels_yn = torch.where(yes_greater_than_no, Yes_id, No_id)
        correct_labels_yn = actual_test_labels == pred_test_labels_yn

        correct_count_yn += correct_labels_yn.sum().tolist()

        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()

        gold_labels_all.extend(actual_test_labels.tolist())
        pred_labels_all.extend(pred_test_labels.tolist())

    accuracy = round(correct_count / total_count, 2)
    accuracy_yn = round(correct_count_yn / total_count, 2)
    return {"accuracy": accuracy, "accuracy_yn": accuracy_yn}, gold_labels_all, pred_labels_all

def calculate_loss(logits, labels, intervenable):
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, intervenable.model_config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    for k, v in intervenable.interventions.items():
        boundary_loss = 1.0 * v[0].intervention_boundaries.sum()
    loss += boundary_loss

    return loss

#def load_intervention(intervention_dir, model):
#    inter = IntervenableModel.load(intervention_dir, model=model)
#    return inter


def train_boundless_das(layer_to_investigate,
                        model,
                        Yes_id,
                        No_id,
                        train_dataloader,
                        test_dataloader,
                        detail_string,
                        verbose=False,
                        epochs=1,
                        evaluateit=False,
                        outputdirname='output',
                        filter_name = 'filter',
                        cuda_gpu="0",
                        control=False,
                        topdir='models'):#

    if evaluateit and test_dataloader is None:
        raise ValueError("Need to provide the test_dataloader!")

    config = simple_boundless_das_position_config(
        type(model), "block_output", layer_to_investigate
    )
    intervenable = IntervenableModel(config, model)
    intervenable.set_device("cuda:"+cuda_gpu)
    intervenable.disable_model_gradients()

    #epochs = 1 #3   #2#3
    #t_total = int(len(train_dataloader) * 3)
    #NOTE I think this is the right thing?

    t_total = int(len(train_dataloader) * epochs)

    warm_up_steps = 0.1 * t_total
    optimizer_params = []
    for k, v in intervenable.interventions.items():
        optimizer_params += [{"params": v[0].rotate_layer.parameters()}]
        optimizer_params += [{"params": v[0].intervention_boundaries, "lr": 1e-2}]
    optimizer = torch.optim.Adam(optimizer_params, lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
    )

    #epochs = 2#3
    gradient_accumulation_steps = 4
    total_step = 0
    target_total_step = len(train_dataloader) * epochs
    temperature_start = 50.0
    temperature_end = 0.1
    temperature_schedule = (
        torch.linspace(temperature_start, temperature_end, target_total_step)
        .to(torch.bfloat16)
        .to("cuda:"+cuda_gpu)
    )
    intervenable.set_temperature(temperature_schedule[total_step])

    intervenable.model.train()  # train enables drop-off but no grads
    print("model trainable parameters: ", count_parameters(intervenable.model))
    print("intervention trainable parameters: ", intervenable.count_parameters())
    train_iterator = trange(0, int(epochs), desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True
        )
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda:"+cuda_gpu)
            b_s = inputs["input_ids"].shape[0]
            if verbose:
                print("BATCH_SIZE: ", b_s)
                print("WHAT SHAPE IS THIS WTF", inputs["source_pos"].shape)
                print(inputs['source_pos'])
                print(inputs['base_pos'])
            _, counterfactual_outputs = intervenable(
                {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]   },
                [{"input_ids": inputs["source_input_ids"], "attention_mask": inputs["source_attention_mask"] }],
                {"sources->base": ([[[i] for i in  inputs['source_pos'].tolist()   ]] ,    [[[i] for i in  inputs['base_pos'].tolist()  ]]    ) },
             #   {"sources->base": ( [ [[tok_pos_to_investigate]]*b_s ], [ [[tok_pos_to_investigate]]*b_s ]        )  },
             #   {"sources->base": tok_pos_to_investigate},  # swap 80th token
            )
            eval_metrics, gold_labels_all, pred_labels_all = compute_metrics(
                [counterfactual_outputs.logits], [inputs["labels"]], Yes_id, No_id
            )

            # loss and backprop
            loss = calculate_loss(counterfactual_outputs.logits, inputs["labels"], intervenable)
            loss_str = round(loss.item(), 2)
            epoch_iterator.set_postfix({"loss": loss_str, "acc": eval_metrics["accuracy"]})

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            if total_step % gradient_accumulation_steps == 0:
                if not (gradient_accumulation_steps > 1 and total_step == 0):
                    optimizer.step()
                    scheduler.step()
                    intervenable.set_zero_grad()
                    intervenable.set_temperature(temperature_schedule[total_step])
            total_step += 1

    if evaluateit:
        #eval_metrics = evaluate(intervenable, test_dataloader)
        eval_metrics, gold_labels_all, pred_labels_all = evaluate(intervenable, test_dataloader)
    else:
        eval_metrics = None

    if control:
        #intervenable.save('models/gemma-2-9b-it/all/' + detail_string)
        intervenable.save(os.path.join(topdir, 'control', outputdirname, filter_name, detail_string))

        #JUST IN CASE
        #intervenable.save_intervention('models/gemma-2-9b-it/all-si/' + detail_string)
        intervenable.save(os.path.join(topdir, 'control', outputdirname, filter_name+"-si", detail_string))
    else:
        #intervenable.save('models/gemma-2-9b-it/all/' + detail_string)
        intervenable.save(os.path.join(topdir, outputdirname, filter_name, detail_string))

        #JUST IN CASE
        #intervenable.save_intervention('models/gemma-2-9b-it/all-si/' + detail_string)
        intervenable.save(os.path.join(topdir, outputdirname, filter_name+"-si", detail_string))

    return intervenable, eval_metrics


# EVALUATE:
def evaluate(intervenable, test_dataloader, Yes_id, No_id, cuda_gpu="0"):
    #set_seed(42)
    intervenable.set_device("cuda:"+cuda_gpu)
    #intervenable.disable_model_gradients()
    #intervenable.disable_intervention_gradients()
    # evaluation on the test set
    eval_labels = []
    eval_preds = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, desc=f"Test")
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda:"+cuda_gpu)
            b_s = inputs["input_ids"].shape[0]
            _, counterfactual_outputs = intervenable(
                {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]   },
                [{"input_ids": inputs["source_input_ids"], "attention_mask": inputs["source_attention_mask"] }],
                #{"input_ids": inputs["input_ids"]},
                #[{"input_ids": inputs["source_input_ids"]}],
                {"sources->base": ([[[i] for i in  inputs['source_pos'].tolist()   ]] ,    [[[i] for i in  inputs['base_pos'].tolist()  ]]    ) },
                #{"sources->base": ( [ [[tok_pos_to_investigate]]*b_s ], [ [[tok_pos_to_investigate]]*b_s ]        )  },  # swap
           #     {"sources->base": tok_pos_to_investigate},  # swap 80th token
            )
            eval_labels += [inputs["labels"]]
            eval_preds += [counterfactual_outputs.logits]

    eval_metrics, gold_labels_all, pred_labels_all = compute_metrics(eval_preds, eval_labels, Yes_id, No_id)
    print(eval_metrics)
    return eval_metrics, gold_labels_all, pred_labels_all


#TODO feed in Yes_id, No_id here if want to use this... and cuda_gpu also
def eval_all(test, model, exp_dir, test_name='low-sim-pos'):#, Yes_id, No_id):
    batch_size = 5#8
    offset = 0
    layers = [0, 5, 10, 15, 20, 25, 31]
    #layers = [10]
    #relpos = ['conclusion_last']
    relpos = ['premise_first', 'premise_last', 'conclusion_first', 'conclusion_last', 'last']
    results = np.empty((len(layers), len(relpos)))

    for ii,layer in enumerate(layers):
        for jj, relative_pos in enumerate(relpos):
            print(layer, relative_pos)
            #NOTE: test_name should be called test_filter
            #train_dataloader = create_dataset_for_intervention(train, relative_pos, offset, 'high-sim-pos') #'balanced')
            test_dataloader  = create_dataset_for_intervention(test,  relative_pos, offset, test_name, batch_size)

            detail_string = 'relpos-'+relative_pos+'-offset-'+str(offset)+'-layer-'+str(layer)

            intervenable = IntervenableModel.load(os.path.join('models/',exp_dir, detail_string), model=model)
            #intervenable, eval_metrics = train_boundless_das(layer, train_dataloader, test_dataloader, detail_string, verbose=False, epochs=epochs, evaluateit=True)

            #eval_metrics = evaluate(intervenable, test_dataloader)
            eval_metrics, gold_labels_all, pred_labels_all = evaluate(intervenable, test_dataloader, Yes_id, No_id, cuda_gpu=cuda_gpu)

            result =  eval_metrics['accuracy']
            print(result)
            results[ii,jj] =  eval_metrics['accuracy']
    return results


#NOTE this is a weird experiment... maybe don't do this..?!
def reverse_direction(test, testflip):
    offset=0
    exp_dir = 'all'

    layers = [0, 5, 10, 15, 20, 25, 31]
    #layers = [10]
    #relpos = ['conclusion_last']
    relpos = ['premise_first', 'premise_last', 'conclusion_first', 'conclusion_last', 'last']
    results = np.empty((len(layers), len(relpos)))
    Ns =  np.empty((len(layers), len(relpos)))
    for ii,layer in enumerate(layers):
        for jj, relative_pos in enumerate(relpos):
            print(layer, relative_pos)
            #NOTE: test_name should be called test_filter 
            test_dataloader = create_dataset_for_intervention(test,  relative_pos, offset, test_name)
            detail_string = 'relpos-'+relative_pos+'-offset-'+str(offset)+'-layer-'+str(layer)
            intervenable = IntervenableModel.load(os.path.join('models/',exp_dir, detail_string), model=model)

            eval_preds, gold, pred  = evaluate(intervenable, test_dataloader)


            inds = [ii for ii in range(len(gold)) if gold[ii] ==pred[ii]]
            inds_source_yes = [ii for ii,i in enumerate(test_dataloader.dataset['labels']) if tokenizer.decode(i[-1])=='Yes']
            inds2 = sorted(list(set(inds) & set(inds_source_yes)) )

            test_dataloaderslip = create_dataset_for_intervention(testflip,  relative_pos, offset, test_name)

            dl2flip = DataLoader(test_dataloaderslip.dataset.select(inds2), batch_size=8)
            N = len(inds2)
            eval_metrics, gold2df, pred2df  = evaluate(intervenable, dl2flip)

            results[ii,jj] =  eval_metrics['accuracy']
            Ns[ii,jj] = int(N)

            return results, Ns


# reverse:
"""
In [72]: samples_flip = load_data(csv_filepath, tokenizer,
    ...:                           data_fraction=1,
    ...: 
    ...:                           prompt_config="variation-qa-1-mistral-special", flip_pair=True)


"""
#testflip = samples_flip[num_train:]
#exp_dir = 'all'
#results = eval_all(testflip, model, exp_dir, test_name = 'balanced')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DAS.")
    parser.add_argument('relative_pos', type=str, help="position name")
    parser.add_argument('layer', type=int, help="Layer index")
    parser.add_argument('train_filter', type=str, help="train filter")
    parser.add_argument('cuda_gpu', type=int, help="gpu num")
    parser.add_argument('sim', type=str, help="sense or spose")
    parser.add_argument('model', type=str, help="mistral or llama or gemma")
    parser.add_argument("--control", action="store_true", help="Control setting with mapping yes and no to other tokens!")
    #parser.add_argument('test_filter', type=str, help="test filter")
    args = parser.parse_args()

    control = args.control
    cuda_gpu = str(args.cuda_gpu)
    layer = int(args.layer)
    relative_pos = args.relative_pos
    print("LAYER: ", layer)
    print("rel pos: ", relative_pos)

    sim = args.sim
    #topdir = 'models-spose'
    if sim == 'spose':
        csv_filepath = "../data/things/stimuli-pairs/things-inheritance-SPOSE_prototype_sim-pairs.csv"
        topdir = 'models-spose'
    elif sim == 'sense':
        csv_filepath = "../data/things/stimuli-pairs/things-inheritance-sense_based_sim-pairs.csv"
        topdir = 'models'

    modelname = args.model

    if modelname=='mistral':
        modelname = "mistralai/Mistral-7B-Instruct-v0.2"
    elif modelname=='gemma':
        modelname = "google/gemma-2-9b-it"
    elif modelname=='llama':
        modelname = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif modelname=='gemma2b':
        modelname = "google/gemma-2-2b-it"
    else:
        raise ValueError("gemma, mistral or llama")

    #prompt_config = "variation-qa-2" # use for gemma-2-9B + chat
    #modelname = "google/gemma-2-9b-it"

    #NOTE these were the best settings we found on our data
    if modelname == "meta-llama/Meta-Llama-3-8B-Instruct":
        prompt_config = "variation-qa-2"
        chat_style=False
    if modelname == "google/gemma-2-9b-it":
        prompt_config = "variation-qa-2"
        chat_style=True
    if modelname == "mistralai/Mistral-7B-Instruct-v0.2":
        prompt_config = "variation-qa-1-mistral-special"
        chat_style = False
    if modelname == "google/gemma-2-2b-it":
        prompt_config = "variation-qa-1"
        chat_style=False

    data_fraction = 1
    num_train = 3000 #500 #3000 # $2000 # 3000 #800 #2000 #800
    epochs = 2
    batch_size = 16#8#4
    offset = 0
    compute_behavioral = True #False
    compute_all_combinations = False
    eval_during_train = False

    train_filter = args.train_filter #'balanced'

    test_filter = 'balanced'

    print("EPOCHS: ",epochs)
    print("BATCH SIZE: ", batch_size)

    #output_results_name = 'gemma-2-9b-variationqa2-balanced_oct5.pkl'
    output_dir_name = modelname.replace("/",'-')+ "-"+ prompt_config
    output_results_name = output_dir_name + ".pkl"

    if eval_during_train:
        print("SAVING TO: ", output_results_name)

    model, tokenizer = load_model(modelname, cuda_gpu)

    if control:
        #CONTROL map yes and no to these tokens:
        yes_token = "chart"
        no_token = "view"
    else:
        # force yes and no defaults in load_data
        yes_token = None
        no_token = None

    samples, yes_token, no_token = load_data(csv_filepath, tokenizer,
                              data_fraction=data_fraction,#.2,
                              prompt_config=prompt_config,
                              chat_style=chat_style,
                              yes_token=yes_token,
                              no_token=no_token
                              )
                              #prompt_config="variation-qa-1-mistral-special")

    print("YES TOKEN: ", yes_token)
    print("NO TOKEN: ", no_token)
    Yes_id = tokenizer.convert_tokens_to_ids(yes_token)
    No_id = tokenizer.convert_tokens_to_ids(no_token)
    print("YES ID: ", Yes_id)
    print("NO ID: ", No_id)

    if num_train > len(samples): #input_ids.shape[0]:
        raise ValueError("num_train must be smaller than size of dataset")

    train = samples[:num_train]
    test = samples[num_train:]
    print("TRAIN: ", len(train))
    print("TEST: ", len(test))

    #icl_accuracy(prealign_dataloader, model, tokenizer)

    # ICL accuracy on the test set:
    #prealign_dataloader_test = make_inputs_labels_dataset([i['input_ids'] for i in test], [i['output_ids'] for i in test] )
    if compute_behavioral:
        prealign_dataloader_test = make_inputs_labels_dataset([i['input_ids'] for i in test], [i['output_ids'] for i in test] , [i['attention_masks'] for i in test])
        icl_accuracy, actual_test_labels, pred_test_labels, yn = icl_accuracy(prealign_dataloader_test, model, tokenizer, yes_token, no_token)

    #relative_pos = 'conclusion_last' #'last' # 'last' # 'conclusion_last' 'premise_last'
    #offset = 0
    #layer = 10 #31 # last layer is 31


    if compute_all_combinations:
        #layers = [0, 5, 10, 15, 20, 25, 31]
        # for gemma-2-2b:
        #layers = [0, 5, 10, 15, 20, 25]#, 30, 35, 41]
        # for gemma-2-9b
        layers = [0, 5, 10, 15, 20, 25, 30, 35, 41]
        #layers = [10]
        #relpos = ['conclusion_last']
        relpos = ['premise_first', 'premise_last', 'conclusion_first', 'conclusion_last', 'last']
        results = np.empty((len(layers), len(relpos)))

        for ii,layer in enumerate(layers):
            for jj, relative_pos in enumerate(relpos):
                print(layer, relative_pos)

                train_dataloader = create_dataset_for_intervention(train, relative_pos, offset, train_filter , batch_size) #'balanced')
                if eval_during_train:
                    test_dataloader  = create_dataset_for_intervention(test,  relative_pos, offset, test_filter , batch_size)
                else:
                    test_dataloader = None

                detail_string = 'relpos-'+relative_pos+'-offset-'+str(offset)+'-layer-'+str(layer)

                intervenable, eval_metrics = train_boundless_das(layer, model, Yes_id, No_id, train_dataloader, test_dataloader, detail_string, verbose=False, epochs=epochs, evaluateit=eval_during_train, outputdirname=output_dir_name, filter_name = train_filter, cuda_gpu=cuda_gpu, control=control, topdir=topdir)
                if eval_metrics is not None:
                    result =  eval_metrics['accuracy']
                    print(result)
                    results[ii,jj] =  eval_metrics['accuracy']


        if eval_metrics is not None:
            #print(result)
            save_results(results, output_results_name)
            #plot_heatmap(results, [str(i) for i in relpos], [str(i) for i in layers])
    else:
        print(layer, relative_pos)

        train_dataloader = create_dataset_for_intervention(train, relative_pos, offset, train_filter , batch_size) #'balanced')
        if eval_during_train:
            test_dataloader = create_dataset_for_intervention(test,  relative_pos, offset, test_filter , batch_size)
        else:
            test_dataloader = None

        detail_string = 'relpos-'+relative_pos+'-offset-'+str(offset)+'-layer-'+str(layer)

        intervenable, eval_metrics = train_boundless_das(layer, model, Yes_id, No_id, train_dataloader, test_dataloader, detail_string, verbose=False, epochs=epochs, evaluateit=eval_during_train, outputdirname = output_dir_name, filter_name = train_filter, cuda_gpu=cuda_gpu, control=control, topdir=topdir)
        if eval_metrics is not None:
            result =  eval_metrics['accuracy']
            print(result)
            results[ii,jj] =  eval_metrics['accuracy']

        #TODO how results are saved
        if eval_metrics is not None:
            #print(result)
            save_results(results, output_results_name)
            #plot_heatmap(results, [str(i) for i in relpos], [str(i) for i in layers])

