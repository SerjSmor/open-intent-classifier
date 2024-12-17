PROMPT_TEMPLATE = "Topic %% Customer: {text}.\nEND MESSAGE\nChoose one topic that matches customer's issue.\n {labels} \nClass name: "

OPENAI_PROMPT_TEMPLATE = ''' 
    You are an expert in customer service domain. You need to classify a customer message into one of
    the following classes: {labels} 
    Please return json object with the following structure: {'{class_name: '', reasoning: ''}'} class name should not contain a number.
    {examples}
    Customer message: {text} %
    Answer: .
'''

SMOLLM2_PROMPT_TEMPLATE = '''Input format: 
    Text: the message of the customer.
    Intent list: all of the possible labels, delimeted by a %.
    Answer: the name of the intent from the list that matches the intent from the message.

    Answer format is one of the intent from the list.
    Example
    ---
    {examples}
    --- 
    Text: {text}
    Intent List: {labels}
    Answer: 
'''

INTENT_CLASSIFIER_248M_FLAN_T5_BASE = "Serj/intent-classifier"
INTENT_CLASSIFIER_80M_FLAN_T5_SMALL = "Serj/intent-classifier-flan-t5-small"

DEFAULT_MAX_LENGTH = 512
DEFAULT_TRUNCATION = True

SMOLLM2_1_7B = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
SMOLLM2_360M = "HuggingFaceTB/SmolLM2-360M-Instruct"