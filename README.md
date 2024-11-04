# Open Intent Classification
Closed intent classification uses a set of predefined labels to identify an intent. 
In comparison, open intent classification allows you to define as many labels you want, without fine tuning the model.

This project implements different components that support open intent classification such as an embedder, a T5 based fine tuned model for intent classification and a verbalizer. 
If you are interested in finer detailes you can read my [blog post](https://medium.com/80-20-hacking-ai/components-that-optimize-n-shot-text-classification-f574184e0b81).

The goal of this library is to enable you test your assumptions about your data as fast as possible and to be a one stop shop for everything "classification like", similarly to how [Bertopic](https://maartengr.github.io/BERTopic/index.html) is for clustering. 

## Why should you use this?
1. You are researching nlp classification problems and want to test different embeddings, verbalizers and components with plug-and-play feel
2. You want to detect semantically user intents in text but either don't want to commit to pre-defined classes OR just want to test out the fastest way to classify text other than through an LLM


[!IMPORTANT]
> open-intent-classification project is in Alpha stage. 
> 1. Expect API changes
> 2. Milage may vary. Quality of classifiers have been tested on Atis and Banking77


## Usage
A full example is under [Atis Notebook](https://github.com/SerjSmor/open-intent-classifier/blob/main/notebooks/atis_example.ipynb)

### T5 Based Intent Classification 
````python
from open_intent_classifier.model import IntentClassifier
model = IntentClassifier()
labels = ["Cancel Subscription", "Refund Requests", "Broken Item", "And More..."]
text = "I don't want to continue this subscription"
predicted_label = model.predict(text, labels)
````

By default, the IntentClassifier is loading a small model with 80M parameters.

For higher accuracy you can initialize the model with: 
```python
from open_intent_classifier.model import IntentClassifier
from open_intent_classifier.consts import INTENT_CLASSIFIER_248M_FLAN_T5_BASE
model = IntentClassifier(INTENT_CLASSIFIER_248M_FLAN_T5_BASE)
```
This will increase model latency as well.


### Embeddings Based Classification
```python
from open_intent_classifier.embedder import StaticLabelsEmbeddingClassifier
labels = ["Cancel Subscription", "Refund Requests", "Broken Item", "And More..."]
text = "I don't want to continue this subscription"
embeddings_classifier = StaticLabelsEmbeddingClassifier(labels)
predicted_label = embeddings_classifier.predict(text)

```

### LLM Based Classification
Using LLM for classification is a viable option that sometimes provides the highest quality.
Currently we have implemented Open AI based LLMs.

```python
from open_intent_classifier.model import OpenAiIntentClassifier
labels = ["Cancel Subscription", "Refund Requests", "Broken Item", "And More..."]
text = "I don't want to continue this subscription"
model_name = "gpt-4o-mini"
classifier = OpenAiIntentClassifier(model_name)
result = classifier.predict(text=text, labels=labels)
```


## Training the T5 base classifier 
The details of training of the classifier is in another repository. I have separated training from inference in order to allow each repository to be focused and extended.

You can read about the training in the training repo: https://github.com/SerjSmor/intent_classification

# Roadmap

- [x] Add LLM based classification
- [ ] Add embeddings filtering stage for classifiers
- [ ] Add small language models as classifiers
- [ ] Add multithreading for LLM based classifiers
- [ ] Add an option to ensemble embeddings and T5 (and additional models)
- [ ] Create a recommender for fine-tuning