import csv
import json


def read_csv_dict(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
