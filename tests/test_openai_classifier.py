import unittest

from open_intent_classifier.model import OpenAiIntentClassifier


class TestOpenAiIntentClassifier(unittest.TestCase):

    def test_prediction(self):
        classifier = OpenAiIntentClassifier(model_name="gpt-4o-mini")
        result = classifier.predict(text="I want to cancel subscription",
                                    labels=["cancel subscription", "cancel refund", "permission issue"])

        self.assertEquals("cancel subscription", result.class_name)
