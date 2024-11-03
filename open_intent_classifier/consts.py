PROMPT_TEMPLATE = "Topic %% Customer: {text}.\nEND MESSAGE\nChoose one topic that matches customer's issue.\n {labels} \nClass name: "

OPENAI_PROMPT_TEMPLATE = ''' 
    You are an expert in customer service domain. You need to classify a customer message into one of
    the following classes: {labels} % Customer message: {text} % Please return json object with the following structure: {'{class_name: '', reasoning: ''}'} class name should not contain a number.
'''

INTENT_CLASSIFIER_248M_FLAN_T5_BASE = "Serj/intent-classifier"
INTENT_CLASSIFIER_80M_FLAN_T5_SMALL = "Serj/intent-classifier-flan-t5-small"

DEFAULT_MAX_LENGTH = 512
DEFAULT_TRUNCATION = True

