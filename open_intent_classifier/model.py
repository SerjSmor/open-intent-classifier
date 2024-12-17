import os
from dataclasses import dataclass
from typing import List
import logging

import dspy
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from dotenv import load_dotenv

from open_intent_classifier.consts import INTENT_CLASSIFIER_80M_FLAN_T5_SMALL, DEFAULT_MAX_LENGTH, DEFAULT_TRUNCATION, \
    PROMPT_TEMPLATE, OPENAI_PROMPT_TEMPLATE, SMOLLM2_1_7B, SMOLLM2_PROMPT_TEMPLATE

from open_intent_classifier.utils import join_labels

CUDA = "cuda"

GPT_4O_MINI = 'gpt-4o-mini'

TEXT_PLACEHOLDER = "{text}"
LABELS_PLACEHOLDER = "{labels}"
EXAMPLES_PLACEHOLDER = "{examples}"

# Create a logger with the filename
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


@dataclass
class ClassificationResult:
    class_name: str
    reasoning: str


@dataclass
class ClassificationExample:
    text: str
    intent: str
    intent_labels_str: str

    def __str__(self):
        return f'''
            -----
            Example
            Text: {self.text}
            Intents list: {self.intent_labels_str}
            Answer: {self.intent}
            -----
        '''


DEFAULT_EXAMPLE = ClassificationExample(
    text="I want to cancel subscription",
    intent="Cancel Subscription",
    intent_labels_str="Refund Request%Delivery Late%Cancel Subscription"
)


class Classifier:
    def predict(self, text: str, labels: List[str], **kwargs) -> ClassificationResult:
        raise NotImplementedError()


class Classification(dspy.Signature):
    """Classify the customer message into one of the intent labels.
    The output should be only the predicted class as a single intent label."""

    customer_message = dspy.InputField(desc="Customer message during customer service interaction")
    intent_labels = dspy.InputField(desc="Labels that represent customer intent")
    intent_class = dspy.OutputField(desc="a label best matching customer's intent ")


class DSPyClassifier(dspy.Module, Classifier):
    def __init__(self, model_name=GPT_4O_MINI):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(Classification)
        lm = dspy.OpenAI(model=model_name)
        dspy.settings.configure(lm=lm)

    def forward(self, customer_message: str, labels: str):
        return self.generate_answer(customer_message=customer_message, labels=labels)

    def predict(self, text: str, labels: List[str], **kwargs) -> ClassificationResult:
        labels = join_labels(labels)
        pred = self.forward(customer_message=text, labels=labels)
        return ClassificationResult(pred.intent_class, pred.rationale)


class OpenAiIntentClassifier(Classifier):
    def __init__(self, model_name: str, openai_api_key: str = "", few_shot_examples: List[ClassificationExample] = None):
        if not openai_api_key:
            load_dotenv()
            openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name
        self.few_shot_examples = [] if few_shot_examples is None else []

    def predict(self, text: str, labels: List[str], **kwargs) -> ClassificationResult:
        joined_labels = join_labels(labels)

        if len(self.few_shot_examples) == 0:
            self.few_shot_examples.append(DEFAULT_EXAMPLE)
        examples_str = ""
        for example in self.few_shot_examples:
            examples_str += str(example)

        prompt = OPENAI_PROMPT_TEMPLATE.replace(LABELS_PLACEHOLDER, joined_labels).replace(TEXT_PLACEHOLDER, text).\
            replace(EXAMPLES_PLACEHOLDER, examples_str)

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
    def __init__(self, model_name: str = INTENT_CLASSIFIER_80M_FLAN_T5_SMALL, device: str = None,
                 verbose: bool = False):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.device = device
        if verbose:
            logger.setLevel(logging.DEBUG)

    def _build_prompt(self, text: str, labels: List[str], prompt_template: str = PROMPT_TEMPLATE):
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


class SmolLm2Classifier(Classifier):
    def __init__(self, model_name: str = SMOLLM2_1_7B, device: str = CUDA, verbose: bool = False,
                 few_shot_examples: List[ClassificationExample] = None):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.verbose = verbose
        self.few_shot_examples = [] if few_shot_examples is None else few_shot_examples

    def predict(self, text: str, labels: List[str], **kwargs) -> ClassificationResult:
        labels_str = "%".join(labels)

        if len(self.few_shot_examples) == 0:
            self.few_shot_examples.append(DEFAULT_EXAMPLE)
        examples_str = ""
        for example in self.few_shot_examples:
            examples_str += str(example)

        prompt = SMOLLM2_PROMPT_TEMPLATE.replace("{text}", text).replace("{labels}", labels_str).replace("{examples}", examples_str)
        if self.verbose:
            logger.info(prompt)

        messages = [{"role": "system",
                     "content": "You are a customer service expert. Your goal is to predict what is the intent of the user from a predfined list"},
                    {"role": "user", "content": prompt}]

        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs, max_new_tokens=350)
        output = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

        if self.verbose:
            logger.info(output)

        last_answer_occurrence = output.rsplit("Answer: ", 2)[-1]
        return ClassificationResult(last_answer_occurrence, "")


