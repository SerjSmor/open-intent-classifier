from unittest import TestCase

from open_intent_classifier.model import IntentClassifier


class TestFlanT5Base(TestCase):

    def setUp(self) -> None:
        self.model = IntentClassifier()

    def test_sanity(self):
        output = self.model.predict("I want to cancel subscription", ["Cancel Subscription", "Refund"])
        self.assertTrue("Cancel Subscription", output)

    def test_cuda(self):
        output = self.model.predict("I want a refund", ["Refund Request", "Issues", "Bug Report"])
        self.assertTrue("Refund Request", output)

    def test_cpu(self):
        output = self.model.predict("I don't remember my password, can you please help me?",
                                    ["Cancel Subscription", "Refund Request", "Transfer Request", "Login Issues"])
        self.assertTrue("Login Issues", output)
