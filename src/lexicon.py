from dataclasses import dataclass


@dataclass
class Property:
    property_name: str
    singular: str
    plural: str


@dataclass
class Concept:
    lemma: str
    singular: str
    plural: str
    article: str
    generic: str
    taxonomic_phrase: str

    def is_a(self, hypernym, split=False):
        if self.generic == "p":
            n1 = self.plural
        else:
            n1 = self.article

        # hypernym is an instance of Concept.
        # hypernym is always singular
        if not split:
            return f"{n1} {self.taxonomic_phrase} {hypernym.singular}"
        else:
            # return n1, f"{self.taxonomic_phrase} {hypernym.singular}"
            return f"{n1} {self.taxonomic_phrase}", hypernym.singular

    def property_sentence(self, prop: Property):
        if self.generic == "p":
            n1 = self.plural
            prop = prop.plural
        else:
            n1 = self.article
            prop = prop.singular

        return f"{n1} {prop}"
