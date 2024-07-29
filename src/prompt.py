import lexicon

from dataclasses import dataclass
from string import Template


@dataclass
class Prompt:
    template: Template
    zero_shot: Template

    def create_sample(
        self,
        premise: lexicon.Concept,
        conclusion: lexicon.Concept,
        prop=lexicon.Concept,
        control_sentence="nothing is daxable",
    ):
        premise_sentence = premise.property_sentence(prop)
        conclusion_sentence = conclusion.property_sentence(prop)

        zero_shot_prompt = self.zero_shot.substitute(conclusion=conclusion_sentence)
        control_prompt = self.template.substitute(
            premise=control_sentence, conclusion=conclusion_sentence
        )
        reasoning_prompt = self.template.substitute(
            premise=premise_sentence, conclusion=conclusion_sentence
        )

        return (zero_shot_prompt, control_prompt, reasoning_prompt)

    def create_sample_tokenized(
        self,
        premise: lexicon.Concept,
        conclusion: lexicon.Concept,
        prop=lexicon.Concept,
        control_sentence="nothing is daxable",
        tokenizer=None,
    ):
        premise_sentence = premise.property_sentence(prop)
        conclusion_sentence = conclusion.property_sentence(prop)

        def _chat_template(sequence):
            formatted = [{"role": "user", "content": sequence.strip()}]
            return tokenizer.apply_chat_template(
                formatted, tokenize=False, add_generation_prompt=True
            )

        # print(self.zero_shot.substitute(conclusion=conclusion_sentence))
        zero_shot_prompt = _chat_template(
            self.zero_shot.substitute(conclusion=conclusion_sentence)
        )
        control_prompt = _chat_template(
            self.template.substitute(
                premise=control_sentence, conclusion=conclusion_sentence
            )
        )
        reasoning_prompt = _chat_template(
            self.template.substitute(
                premise=premise_sentence, conclusion=conclusion_sentence
            )
        )

        return (zero_shot_prompt, control_prompt, reasoning_prompt)
