import os
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

sys.path.append("/data/users/jdr/concepts/pyvene")
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

from boundless_das2 import load_data, create_dataset_for_intervention, evaluate

#cuda_gpu = "1"
LEMMA_PATH = "../data/things/things-lemmas-annotated.csv"
csv_filepath = "../data/things/stimuli-pairs/things-inheritance-sense_based_sim-pairs.csv"
#csv_filepath = "../data/things/stimuli-pairs/things-inheritance-SPOSE_prototype_sim-pairs.csv"
num_train = 3000
offset = 0


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="eval boundless das")
    parser.add_argument('relative_pos', type=str, help="position name")
    parser.add_argument('layer', type=int, help="Layer index")
    parser.add_argument('train_filter', type=str, help="train filter")
    parser.add_argument('test_filter', type=str, help="test filter")
    parser.add_argument('portion', type=int, help="portion")
    parser.add_argument('cuda_gpu', type=int, help="cuda gpu")
    parser.add_argument('top_model_dir', type=str, help="top directory where model is")
    parser.add_argument('sim', type=str, help="sense or spose")
    parser.add_argument('model', type=str, help="mistral or llama or gemma")
    parser.add_argument("--control", action="store_true", help="Control setting with mapping yes and no to other tokens!")
    #parser.add_argument('test_filter', type=str, help="test filter")
    args = parser.parse_args()

    control = args.control
    cuda_gpu = str(args.cuda_gpu)
    layer = int(args.layer)
    relative_pos = args.relative_pos
    portion = int(args.portion)#4
    print("LAYER: ", layer)
    print("rel pos: ", relative_pos)
    print("portion: ", portion)

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
    else:
        raise ValueError("gemma, mistral or llama")

#    modelname = "mistralai/Mistral-7B-Instruct-v0.2"
    #modelname = "google/gemma-2-9b-it"
    #first_half = True

    if modelname == "meta-llama/Meta-Llama-3-8B-Instruct":
        prompt_config = "variation-qa-2"
        chat_style=False
    if modelname == "google/gemma-2-9b-it":
        prompt_config = "variation-qa-2"
        chat_style=True
    if modelname == "mistralai/Mistral-7B-Instruct-v0.2":
        prompt_config = "variation-qa-1-mistral-special"
        chat_style = False

    model, tokenizer = load_model(modelname, cuda_gpu)

    if control:
        #CONTROL map yes and no to these tokens:
        yes_token = "chart"
        no_token = "view"
    else:
        # force yes and no defaults in load_data
        yes_token = None
        no_token = None

    samples, yes_token, no_token = load_data(csv_filepath, tokenizer, data_fraction=1,  prompt_config=prompt_config, chat_style=chat_style, yes_token=yes_token, no_token=no_token)

    print("YES TOKEN: ", yes_token)
    print("NO TOKEN: ", no_token)
    Yes_id = tokenizer.convert_tokens_to_ids(yes_token)
    No_id = tokenizer.convert_tokens_to_ids(no_token)
    print("YES ID: ", Yes_id)
    print("NO ID: ", No_id)

    test = samples[num_train:]

    if portion==1:
        print("Let's see if the whole thing fits in memory!")
#        test = test[:254]
#    elif portion==2:
#        test = test[254:508]
#    elif portion==3:
#        test = test[508:762]
#    elif portion==4:
#        test = test[762:]
    else:
        raise ValueError("!!")
    print("Evaluating on num samples: ", len(test))

    batch_size = 24 #8 #6 #5 #4

    train_filter = args.train_filter #'balanced'
    test_filter = args.test_filter #'balanced'

    top_model_dir = args.top_model_dir #'models/'
    print("TOP MODEL DIR: ", top_model_dir)

    #layer = 10
    #relative_pos = 'conclusion_last'

    detail_string = 'relpos-'+relative_pos+'-offset-'+str(offset)+'-layer-'+str(layer)
    exp_dir = modelname.replace("/",'-')+ "-"+ prompt_config

    test_dataloader = create_dataset_for_intervention(test,  relative_pos, offset, test_filter, batch_size)
    imdir = os.path.join(top_model_dir, exp_dir, train_filter, detail_string)

    intervenable = IntervenableModel.load(imdir, model=model)

    eval_metrics, gold_labels_all, pred_labels_all = evaluate(intervenable, test_dataloader, Yes_id, No_id, cuda_gpu=cuda_gpu)

    results_path = os.path.join('results', top_model_dir, exp_dir, train_filter+"-"+test_filter)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

#    with open(os.path.join(results_path, 'iia-argmax-'+str(portion)+'.csv'), 'a') as fd:
#        fd.write(str(layer) + '\t' + relative_pos + '\t' + str(eval_metrics['accuracy']) + '\n')

#    with open(os.path.join(results_path, 'iia-yn-'+str(portion)+'.csv'), 'a') as fd:
#        fd.write(str(layer) + '\t' + relative_pos + '\t' + str(eval_metrics['accuracy_yn']) + '\n')

    with open(os.path.join(results_path, 'iia-argmax'+'.csv'), 'a') as fd:
        fd.write(str(layer) + '\t' + relative_pos + '\t' + str(eval_metrics['accuracy']) + '\n')

    with open(os.path.join(results_path, 'iia-yn'+'.csv'), 'a') as fd:
        fd.write(str(layer) + '\t' + relative_pos + '\t' + str(eval_metrics['accuracy_yn']) + '\n')

