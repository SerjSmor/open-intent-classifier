from unittest import TestCase

import torch.cuda

from open_intent_classifier.model import IntentClassifier


class TestDevices(TestCase):

    def test_default(self):
        self.model = IntentClassifier()

    def test_cuda(self):
        if torch.cuda.is_available():
            self.model = IntentClassifier(device="cuda")

    def test_cpu(self):
        self.model = IntentClassifier(device="cpu")

    def test_wrong_device(self):
        with self.assertRaises(RuntimeError):
            self.model = IntentClassifier(device="wrong")