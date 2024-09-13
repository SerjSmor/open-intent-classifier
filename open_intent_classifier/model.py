from typing import List

from transformers import T5ForConditionalGeneration, T5Tokenizer

from open_intent_classifier.consts import INTENT_CLASSIFIER_80M_FLAN_T5_SMALL, DEFAULT_MAX_LENGTH, DEFAULT_TRUNCATION, \
    PROMPT_TEMPLATE

import logging

# Create a logger with the filename
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


class IntentClassifier:
    def __init__(self, model_name: str = INTENT_CLASSIFIER_80M_FLAN_T5_SMALL, device: str = None, verbose=False):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.device = device
        if verbose:
            logger.setLevel(logging.DEBUG)

    def _build_prompt(self, text: str, labels: List[str], prompt_template=PROMPT_TEMPLATE):
        prompt_options = "Options:\n"
        for label in labels:
            prompt_options += f"# {label} \n"

        # first replace {labels} this way we know that {text} can't be misused with tokens that can overlap with {text}
        prompt = prompt_template.replace("{labels}", prompt_options)
        prompt = prompt.replace("{text}", text)

        return prompt

    def predict(self, text, labels: List[str], **kwargs):
        prompt = self._build_prompt(text, labels)
        max_length = kwargs.get("max_length", DEFAULT_MAX_LENGTH)
        truncation = kwargs.get("truncation", DEFAULT_TRUNCATION)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", max_length=max_length,
                                          truncation=truncation).to(self.device)
        # Generate the output
        output = self.model.generate(input_ids, **kwargs)
        # Decode the output tokens
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        logger.debug(f"Full prompt: {prompt}")
        logger.debug(f"Decoded output: {decoded_output}")
        return decoded_output

