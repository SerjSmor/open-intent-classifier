import unittest

from open_intent_classifier.model import DSPyClassifier


class TestDSPyClassifier(unittest.TestCase):

    def test_basic(self):
        classifier = DSPyClassifier()
        labels = ["Cancel subscription", "Refund request"]
        text = "I want to cancel my subscription"
        result = classifier.predict(text, labels)
        self.assertEquals("Cancel subscription".lower(), result.class_name.lower())

