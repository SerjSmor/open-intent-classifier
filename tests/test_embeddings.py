from unittest import TestCase

from open_intent_classifier.embedder import StaticLabelsEmbeddingClassifier


class TestEmbeddings(TestCase):
    def setUp(self) -> None:
        self.model = StaticLabelsEmbeddingClassifier(["Cancel Subscription", "Refund Requests"])

    def test_top_1(self):
        labels, probabilities = self.model.predict("I want to cancel my subscription", n=1)
        self.assertTrue(1, len(labels))
        self.assertTrue(1, len(probabilities))
        self.assertTrue("Cancel Subscription", labels[0])

    def test_top_2(self):
        labels, probabilities = self.model.predict("I want to cancel subscription", n=2)
        self.assertTrue(2, len(labels))
        self.assertTrue(2, len(probabilities))
        self.assertTrue("Cancel Subscription", labels[0])
        self.assertTrue("Refund Requests", labels[1])

