from typing import List


def join_labels(labels):
    prompt_options = "Options:\n"
    for label in labels:
        prompt_options += f"# {label} \n"
    return prompt_options

def labels_to_str(labels: List[str]):
    return "%".join(labels)