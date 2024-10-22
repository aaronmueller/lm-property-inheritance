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

from boundless_das import load_data, create_dataset_for_intervention, evaluate, make_inputs_labels_dataset, icl_accuracy

#cuda_gpu = "1"
LEMMA_PATH = "../data/things/things-lemmas-annotated.csv"
#csv_filepath = "../data/things/stimuli-pairs/things-inheritance-sense_based_sim-pairs.csv"
csv_filepath = "../data/things/stimuli-pairs/things-inheritance-SPOSE_prototype_sim-pairs.csv"
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
    #parser.add_argument('test_filter', type=str, help="test filter")
    args = parser.parse_args()

    cuda_gpu = str(args.cuda_gpu)
    layer = int(args.layer)
    relative_pos = args.relative_pos
    portion = int(args.portion)#4
    print("LAYER: ", layer)
    print("rel pos: ", relative_pos)
    print("portion: ", portion)

    #modelname = "meta-llama/Meta-Llama-3-8B-Instruct"
    #modelname = "mistralai/Mistral-7B-Instruct-v0.2"
    modelname = "google/gemma-2-2b-it"
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
    if modelname == "google/gemma-2-2b-it":
        prompt_config = "variation-qa-1"
        chat_style=False

    print("MODEL: ", modelname)

    model, tokenizer = load_model(modelname, cuda_gpu)

    # FLIP THEM!
    samplesf, yes_token, no_token = load_data(csv_filepath, tokenizer, data_fraction=1,  prompt_config=prompt_config, chat_style=chat_style, flip_pair=True)

    samples, yes_token, no_token = load_data(csv_filepath, tokenizer, data_fraction=1,  prompt_config=prompt_config, chat_style=chat_style, flip_pair=False)

    print("YES TOKEN: ", yes_token)
    print("NO TOKEN: ", no_token)
    Yes_id = tokenizer.convert_tokens_to_ids(yes_token)
    No_id = tokenizer.convert_tokens_to_ids(no_token)
    print("YES ID: ", Yes_id)
    print("NO ID: ", No_id)

    test = samples[num_train:]
    testflip = samplesf[num_train:]

    test_yes = [i for i in test if i['is_hyper']=='yes']
    test_no = [i for i in test if i['is_hyper']=='no']
    test_yesf = [i for i in testflip if i['is_hyper']=='yes']
    test_nof = [i for i in testflip if i['is_hyper']=='no']

    #NOTE: Filter for the ones where the model gets it right in deduction:
    prealign_dataloader_test_yes = make_inputs_labels_dataset([i['input_ids'] for i in test_yes], [i['output_ids'] for i in test_yes], [i['attention_masks'] for i in test_yes] )

    prealign_dataloader_test_no = make_inputs_labels_dataset([i['input_ids'] for i in test_no], [i['output_ids'] for i in test_no], [i['attention_masks'] for i in test_no] )

    icl_acc, actual_test_labels, pred_test_labels, yn_yes = icl_accuracy(prealign_dataloader_test_yes, model, tokenizer, yes_token, no_token)
    predyn = [yes_token if i else no_token for i in yn_yes]

    if modelname == "meta-llama/Meta-Llama-3-8B-Instruct":
        actual_test_labels = [i.replace(" ", "Ġ") for i in actual_test_labels]

    good_inds_yes = [ii for ii,i in enumerate(predyn) if predyn[ii]==actual_test_labels[ii]]

    icl_acc, actual_test_labels, pred_test_labels, yn_no = icl_accuracy(prealign_dataloader_test_no, model, tokenizer, yes_token, no_token)
    predyn = [yes_token if i else no_token for i in yn_no]

    if modelname == "meta-llama/Meta-Llama-3-8B-Instruct":
        actual_test_labels = [i.replace(" ", "Ġ") for i in actual_test_labels]

    good_inds_no = [ii for ii,i in enumerate(predyn) if predyn[ii]==actual_test_labels[ii]]

    # filter
    test_yes = [test_yes[i] for i in good_inds_yes]
    test_no = [test_no[i] for i in good_inds_no]

    test_yesf = [test_yesf[i] for i in good_inds_yes]
    test_nof = [test_nof[i] for i in good_inds_no]

    minl = min(len(test_yes), len(test_no))

    Pair = namedtuple('Pair', ['base', 'source'])
    sb_pairs = [Pair(test_no[ii] , test_yes[ii] ) for ii in range(minl)  ] # base is not taxo, source is yes (taxo)
    sb_pairs_rev = [Pair(test_nof[ii] , test_yesf[ii] ) for ii in range(minl)  ] # base is not taxo, source is yes (taxo)

#    create_dataset_for_intervention(None, relative_pos, offset, filter_mode, batch_size, sb_pairs=None)

    #breakpoint()

    print("SAMPLES so far: ", len(sb_pairs))

#    testflip = [testflip[i] for i in good_inds]
#    test = [test[i] for i in good_inds]

#    assert all([testflip[ii]['conclusion']==test[ii]['premise'] and testflip[ii]['premise']==test[ii]['conclusion'] for ii in range(len(test))] )



    if portion==1:
        print("Let's see if the whole thing fits in memory!")
#        test = test[:254]
#        testflip = testflip[:254]
        #test = test[:48]
        #testflip = testflip[:48]
#    elif portion==2:
#        testflip = testflip[254:508]
#    elif portion==3:
#        testflip = testflip[508:762]
#    elif portion==4:
#        testflip = testflip[762:]
#    else:
#        raise ValueError("!!")
#    print("Evaluating on num samples: ", len(testflip))

    batch_size = 24 #8 #6 #5 #4

    train_filter = args.train_filter #'balanced'
    test_filter = args.test_filter #'balanced'

    top_model_dir = args.top_model_dir #'models/'
    print("TOP MODEL DIR: ", top_model_dir)

    #layer = 10
    #relative_pos = 'conclusion_last'

    detail_string = 'relpos-'+relative_pos+'-offset-'+str(offset)+'-layer-'+str(layer)
    exp_dir = modelname.replace("/",'-')+ "-"+ prompt_config

    #NOTE evaluated the flipped test set, but filter out non-taxo
    #test_dataloader = create_dataset_for_intervention(testflip,  relative_pos, offset, 'only-taxo', batch_size)
    #test_dataloader = create_dataset_for_intervention(testflip,  relative_pos, offset, 'only-taxo', batch_size)

    test_dataloader = create_dataset_for_intervention(None,  relative_pos, offset, 'balanced', batch_size, sb_pairs)



    #NOTE use all , not sure taxonomic!
    #test_dataloader = create_dataset_for_intervention(testflip,  relative_pos, offset, 'balanced', batch_size)

    print('NUM IN TEST: ',  len(test_dataloader.dataset['input_ids']))

    imdir = os.path.join(top_model_dir, exp_dir, train_filter, detail_string)

    intervenable = IntervenableModel.load(imdir, model=model)

    eval_metrics, gold_labels_all, pred_labels_all = evaluate(intervenable, test_dataloader, Yes_id, No_id, cuda_gpu=cuda_gpu)


#    breakpoint()

    das_good_inds = [ii for ii,i in enumerate(pred_labels_all) if pred_labels_all[ii] == gold_labels_all[ii]]

    print("DAS succeeded in how many cases? ", len(das_good_inds))

    # filter based on where DAS got it right in deduction direction:

    sb_pairs_rev = [sb_pairs_rev[ii] for ii in das_good_inds]

    test_dataloader_rev = create_dataset_for_intervention(None,  relative_pos, offset, 'balanced', batch_size, sb_pairs_rev)
    eval_metrics2, gold_labels_all2, pred_labels_all2 = evaluate(intervenable, test_dataloader_rev, Yes_id, No_id, cuda_gpu=cuda_gpu)

    breakpoint()

    # filter sb_pairs_rev based on where DAS gets it right in forward!

    results_path = os.path.join('results', top_model_dir, 'rev', exp_dir, train_filter+"-"+test_filter)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with open(os.path.join(results_path, 'iia-argmax-'+str(portion)+'.csv'), 'a') as fd:
        fd.write(str(layer) + '\t' + relative_pos + '\t' + str(eval_metrics['accuracy']) + '\n')

    with open(os.path.join(results_path, 'iia-yn-'+str(portion)+'.csv'), 'a') as fd:
        fd.write(str(layer) + '\t' + relative_pos + '\t' + str(eval_metrics['accuracy_yn']) + '\n')

#    with open(os.path.join(results_path, 'iia-argmax'+'.csv'), 'a') as fd:
#        fd.write(str(layer) + '\t' + relative_pos + '\t' + str(eval_metrics['accuracy']) + '\n')

#    with open(os.path.join(results_path, 'iia-yn'+'.csv'), 'a') as fd:
#        fd.write(str(layer) + '\t' + relative_pos + '\t' + str(eval_metrics['accuracy_yn']) + '\n')

