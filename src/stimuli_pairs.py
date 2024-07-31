'''
filename: things-inheritance-<similarity_identifier>-pairs.csv
premise, conclusion, hypernymy, similarity_raw, similarity_binary, premise_form, conclusion_form
'''

import argparse
import csv
import json
import utils

from collections import defaultdict

def main(args):
    taxonomic_path = args.triples_path
    negative_samples_path = args.ns_path
    output_path = args.output_path
    similarity_path = args.similarity_path