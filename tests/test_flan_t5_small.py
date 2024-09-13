from unittest import TestCase

from open_intent_classifier.consts import INTENT_CLASSIFIER_80M_FLAN_T5_SMALL
from open_intent_classifier.model import IntentClassifier


class TestFlanT5Small(TestCase):

    def setUp(self) -> None:
        self.model = IntentClassifier(model_name=INTENT_CLASSIFIER_80M_FLAN_T5_SMALL, verbose=True)

    def test_sanity(self):
        output = self.model.predict("I want to cancel subscription", ["Cancel Subscription", "Refund"])
        self.assertTrue("Cancel Subscription", output)

    def test_cuda(self):
        output = self.model.predict("I want a refund", ["Refund Request", "Issues", "Bug Report"])
        self.assertTrue("Refund Request", output)

    def test_cpu(self):
        output = self.model.predict("I don't remember my password, can you please help me?", ["Cancel Subscription",
                                                                                         "Refund Request", "Transfer Request", "Login Issues"])
        self.assertTrue("Login Issues", output)