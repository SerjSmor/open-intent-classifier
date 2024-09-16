from setuptools import setup

from open_intent_classifier import VERSION

setup(
    name="open_intent_classifier",
    version=VERSION,
    packages=["open_intent_classifier"],
    author="Serj Smorodinsky",
    author_email="serjsmor@gmail.com",
    description="This library has two purposes: 1. allow to easily test semantic classification with open labels (not pre defined) for intent recognition. "
                "2. allow to experiment with different n-shot classification components.",
    install_requires=["transformers", "torch", "sentencepiece", "scikit-learn", "sentence-transformers", "pandas"]
)
