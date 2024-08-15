from string import Template

NONCE_WORDS = ["wug", "dax", "fep", "blicket", "kiki", "bouba", "toma", "glorp", "zek"]

PROPERTIES = {
    "can dax": {"singular": "can dax", "plural": "can dax"},
    "is daxable": {"singular": "is daxable", "plural": "are daxable"},
    "has daxes": {"singular": "has daxes", "plural": "have daxes"},
    "is fepish": {"singular": "is fepish", "plural": "are fepish"},
}


PROMPTS = {
    "initial-qa": {
        "template": Template(
            "Given that $premise, is it true that $conclusion? Answer with Yes/No:"
        ),
        "zero_shot": Template("Is it true that $conclusion? Answer with Yes/No:"),
        "qa": True,
        "label-separator": " ",
    },
    "initial-phrasal": {
        "template": Template(
            "Given a premise, produce a conclusion that is true.\nPremise: $premise\nConclusion:"
        ),
        "zero_shot": Template("$conclusion"),
        "qa": False,
        "label-separator": None,
    },
    "variation-qa-1": {
        "template": Template(
            "Answer the question. Given that $premise, is it true that $conclusion?\nAnswer with Yes/No.\n"
        ),
        "zero_shot": Template(
            "Answer the question. Is it true that $conclusion?\nAnswer with Yes/No.\n"
        ),
        "qa": True,
        "label-separator": " ",
    },
    "variation-qa-1-mistral-special": {
        "template": Template(
            "Answer the question. Given that $premise, is it true that $conclusion?<0x0A>Answer with Yes/No.<0x0A>"
        ),
        "zero_shot": Template(
            "Answer the question. Is it true that $conclusion?<0x0A>Answer with Yes/No.<0x0A>"
        ),
        "qa": True,
        "label-separator": "",
    },
    "variation-qa-2": {
        "template": Template(
            "Answer the question. Given that $premise, is it true that $conclusion? Answer with Yes/No. The answer is:"
        ),
        "zero_shot": Template(
            "Answer the question. Is it true that $conclusion? Answer with Yes/No. The answer is:"
        ),
        "qa": True,
        "label-separator": " ",
    },
}


CATEGORY_REPLACEMENTS = {
    'breakfast food': 'breakfast',
    'clothing accessory': 'accessory',
    'farm animal': 'livestock',
    'part of car': 'auto part',
    'personal hygiene item': 'toiletry',
    'watercraft': 'water vehicle',
    'kitchen appliance': 'kitchen equipment',
    'hardware': 'tool'
}