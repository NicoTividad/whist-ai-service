import unittest
from pydantic import ValidationError
from api.models import CardModel, GameStateModel, toCard, fromCard


class TestCardModel(unittest.TestCase):

    def testAValidCardBuilds(self):
        card = CardModel(suit="spades", rank=14)
        self.assertEqual(card.suit, "spades")
        self.assertEqual(card.rank, 14)

    def testAnUnknownSuitIsRejected(self):
        with self.assertRaises(ValidationError):
            CardModel(suit="swords", rank=10)

    def testARankBelowTwoIsRejected(self):
        with self.assertRaises(ValidationError):
            CardModel(suit="hearts", rank=1)

    def testARankAboveFourteenIsRejected(self):
        with self.assertRaises(ValidationError):
            CardModel(suit="hearts", rank=15)


class TestCardConversion(unittest.TestCase):

    def testAModelConvertsToACard(self):
        card = toCard(CardModel(suit="clubs", rank=9))
        self.assertEqual(card.suit, "clubs")
        self.assertEqual(card.rank, 9)

    def testACardRoundTripsBackToTheSameValues(self):
        # the two conversions must agree with each other
        original = CardModel(suit="diamonds", rank=11)
        asDict = fromCard(toCard(original))
        self.assertEqual(asDict, {"suit": "diamonds", "rank": 11})


class TestGameStateModel(unittest.TestCase):

    def validState(self, **changes):
        # a valid state that each test can alter one field of
        state = {
            "deckSeed": 1,
            "playerSeed": 1,
            "seats": ["human", "search", "rule", "search"],
            "handsPlayed": 0,
            "scores": [0, 0],
            "moves": [],
        }
        state.update(changes)
        return state

    def testAValidStateBuilds(self):
        state = GameStateModel(**self.validState())
        self.assertEqual(state.handsPlayed, 0)
        self.assertEqual(state.moves, [])

    def testSampleCountDefaultsToTheOperatingPoint(self):
        state = GameStateModel(**self.validState())
        self.assertEqual(state.sampleCount, 20)

    def testThreeSeatsIsRejected(self):
        with self.assertRaises(ValidationError):
            GameStateModel(**self.validState(seats=["human", "rule", "rule"]))

    def testFiveSeatsIsRejected(self):
        with self.assertRaises(ValidationError):
            GameStateModel(**self.validState(seats=["human", "rule", "rule", "rule", "rule"]))

    def testAnUnknownSeatNameIsRejected(self):
        # a typo must fail loudly rather than quietly seating a random agent
        with self.assertRaises(ValidationError):
            GameStateModel(**self.validState(seats=["human", "serch", "rule", "rule"]))

    def testANegativeSeedIsRejected(self):
        with self.assertRaises(ValidationError):
            GameStateModel(**self.validState(deckSeed=-1))

    def testASampleCountAboveTheCapIsRejected(self):
        with self.assertRaises(ValidationError):
            GameStateModel(**self.validState(sampleCount=1000))

    def testABadCardInTheMoveListIsRejected(self):
        with self.assertRaises(ValidationError):
            GameStateModel(**self.validState(moves=[{"suit": "spades", "rank": 99}]))


if __name__ == "__main__":
    unittest.main()