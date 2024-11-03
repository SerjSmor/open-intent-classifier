import os
from dataclasses import dataclass
from typing import List
import logging

from transformers import T5ForConditionalGeneration, T5Tokenizer
from openai import OpenAI
from dotenv import load_dotenv

from open_intent_classifier.consts import INTENT_CLASSIFIER_80M_FLAN_T5_SMALL, DEFAULT_MAX_LENGTH, DEFAULT_TRUNCATION, \
    PROMPT_TEMPLATE, OPENAI_PROMPT_TEMPLATE

from open_intent_classifier.utils import join_labels

TEXT_PLACEHOLDER = "{text}"
LABELS_PLACEHOLDER = "{labels}"

# Create a logger with the filename
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

@dataclass
class ClassificationResult:
    class_name: str
    reasoning: str


class Classifier:
    def predict(self, text: str, labels: List[str], **kwargs) -> ClassificationResult:
        raise NotImplementedError()

class OpenAiIntentClassifier(Classifier):
    def __init__(self, model_name: str, openai_api_key: str = ""):
        if not openai_api_key:
            load_dotenv()
            openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name



    def predict(self, text: str, labels: List[str], **kwargs) -> ClassificationResult:
        joined_labels = join_labels(labels)
        prompt = OPENAI_PROMPT_TEMPLATE.replace(LABELS_PLACEHOLDER, joined_labels).replace(TEXT_PLACEHOLDER, text)

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        response_dict = eval(completion.choices[0].message.content)
        return ClassificationResult(response_dict["class_name"], response_dict["reasoning"])

class IntentClassifier:
    def __init__(self, model_name: str = INTENT_CLASSIFIER_80M_FLAN_T5_SMALL, device: str = None, verbose=False):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.device = device
        if verbose:
            logger.setLevel(logging.DEBUG)

    def _build_prompt(self, text: str, labels: List[str], prompt_template=PROMPT_TEMPLATE):
        prompt_options = join_labels(labels)

        # first replace {labels} this way we know that {text} can't be misused with tokens that can overlap with {text}
        prompt = prompt_template.replace(LABELS_PLACEHOLDER, prompt_options)
        prompt = prompt.replace(TEXT_PLACEHOLDER, text)

        return prompt

    def predict(self, text: str, labels: List[str], **kwargs) -> ClassificationResult:
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
        return ClassificationResult(class_name=decoded_output, reasoning="")

