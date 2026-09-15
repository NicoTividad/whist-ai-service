import unittest

from whist.engine.game import WhistGame
from whist.engine.rules import GameConfig
from whist.engine.cards import Card


# ----- FOLLOW-SUIT RULE IN legalMoves -----

class TestLegalMoves(unittest.TestCase):

    def setUp(self):
        self.config = GameConfig()
        self.game = WhistGame(self.config, deckSeed=1)
        self.game.hands = [[] for seat in range(self.config.numberOfPlayers)]
        self.game.currentTrick = []

    def testLeaderMayPlayAnyCard(self):
        self.game.hands[0] = [Card("spades", 14), Card("hearts", 4), Card("clubs", 9)]
        expected = [Card("spades", 14), Card("hearts", 4), Card("clubs", 9)]
        self.assertCountEqual(self.game.legalMoves(0), expected)

    def testMustFollowLedSuitWhenAble(self):
        self.game.currentTrick = [(1, Card("hearts", 12))] # hearts is the led suit
        self.game.hands[2] = [Card("hearts", 10), Card("hearts", 4), Card("clubs", 9), Card("spades", 13)]
        expected = [Card("hearts", 10), Card("hearts", 4)]
        self.assertCountEqual(self.game.legalMoves(2), expected)
        self.assertNotIn(Card("clubs", 9), self.game.legalMoves(2))

    def testMayPlayAnyCardWhenVoidInLedSuit(self):
        self.game.currentTrick = [(1, Card("hearts", 12))]
        self.game.hands[2] = [Card("clubs", 9), Card("spades", 13), Card("diamonds", 2)]
        expected = [Card("clubs", 9), Card("spades", 13), Card("diamonds", 2)]
        self.assertCountEqual(self.game.legalMoves(2), expected)


# ----- TRICK WINNER IN resolveTrick -----

class TestTrickWinner(unittest.TestCase):

    def setUp(self):
        self.config = GameConfig()
        self.game = WhistGame(self.config, deckSeed=1)
        self.game.trumpSuit = "spades" # spades is trump in every test below
        self.game.tricksWon = [0 for side in self.config.partnerships]
        self.game.trickHistory = []
        self.game.currentTrick = []

    def testHighestOfLedSuitWinsWithNoTrump(self):
        self.game.currentTrick = [(0, Card("hearts", 9)), (1, Card("hearts", 13)), (2, Card("hearts", 5)), (3, Card("hearts", 11))]
        self.game.resolveTrick()
        self.assertEqual(self.game.leader, 1) # the king of hearts is the highest

    def testAnyTrumpBeatsLedSuit(self):
        self.game.currentTrick = [(0, Card("hearts", 14)), (1, Card("spades", 2)), (2, Card("hearts", 13)), (3, Card("hearts", 12))]
        self.game.resolveTrick()
        self.assertEqual(self.game.leader, 1) # the two of spades is a trump and beats the ace of hearts

    def testHighestTrumpWins(self):
        self.game.currentTrick = [(0, Card("hearts", 14)), (1, Card("spades", 7)), (2, Card("spades", 13)), (3, Card("spades", 4))]
        self.game.resolveTrick()
        self.assertEqual(self.game.leader, 2) # the king of spades is the highest trump

    def testOffSuitDiscardNeverWins(self):
        self.game.currentTrick = [(0, Card("hearts", 9)), (1, Card("hearts", 4)), (2, Card("clubs", 14)), (3, Card("hearts", 6))]
        self.game.resolveTrick()
        self.assertEqual(self.game.leader, 0) # the ace of clubs is neither hearts (led) nor spades (trump), so the nine of hearts wins


# ----- SCORING IN scoreHand -----

class TestScoring(unittest.TestCase):

    def setUp(self):
        self.config = GameConfig()
        self.game = WhistGame(self.config, deckSeed=1)
        self.game.scores = [0 for side in self.config.partnerships]
        self.game.tricksWon = [0 for side in self.config.partnerships]

    def testSideAboveBookScoresOddTricks(self):
        self.game.tricksWon = [9, 4]
        self.game.scoreHand()
        self.assertEqual(self.game.scores, [3, 0]) # 9 minus 6 is 3, and 4 is not above the book

    def testSideAtOrBelowBookScoresZero(self):
        self.game.tricksWon = [7, 6] # the narrowest possible win
        self.game.scoreHand()
        self.assertEqual(self.game.scores, [1, 0]) # 7 minus 6 is 1, and exactly 6 tricks scores nothing


# ----- REPRODUCIBILITY IN playGame -----

class TestReproducibleGame(unittest.TestCase):

    def testSameSeedGivesSameResult(self):
        config = GameConfig()
        firstGame = WhistGame(config, deckSeed=42)
        secondGame = WhistGame(config, deckSeed=42)
        firstResult = firstGame.playGame()
        secondResult = secondGame.playGame()
        self.assertEqual(firstResult, secondResult) # the seed fully determines the game


if __name__ == "__main__":
    unittest.main()
