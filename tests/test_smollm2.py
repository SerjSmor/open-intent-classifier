import unittest

from open_intent_classifier.model import SmolLm2Classifier, ClassificationExample


class TestSmolLm2(unittest.TestCase):

    def test_basic(self):
        labels = ["Cancel subscription", "Refund request"]
        text = "I want to cancel my subscription"

        c = SmolLm2Classifier()
        result = c.predict(text, labels)
        self.assertEquals("Cancel subscription".lower(), result.class_name.lower())

    def test_few_shot(self):
        labels = ["Order flight ticket", "Abbrevations", "Fare code", "Cheap price", "Meals questions", "Seating questions"]
        text = "I want to get a flight ticket to NYC asap"

        example_flight = ClassificationExample("Do you have an open ticket to a flight to Chicago?",
                                               "Order flight ticket", labels)

        example_price = ClassificationExample("What is the cheapest route to London?", "Price questions", labels)

        c = SmolLm2Classifier(verbose=True, few_shot_examples=[example_flight, example_price])

        result = c.predict(text, labels)
        self.assertEquals("Order flight ticket".lower(), result.class_name.lower())


    def test_hierarchy(self):
        labels = ["flight ticket", "price", "meal", "seats"]
        text = "I want to get a flight ticket to NYC asap"

        c = SmolLm2Classifier(verbose=True)
        example_flight = ClassificationExample("Do you have an open ticket to a flight to Chicago?", "flight ticket",
                                               labels)

        example_price = ClassificationExample("What is the price for a flight to London?", "price", labels)

        result = c.predict(text, labels, few_shot_examples=[example_flight, example_price])
        self.assertEquals("flight ticket".lower(), result.class_name.lower())