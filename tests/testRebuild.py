import unittest

from whist.engine.rules import GameConfig
from api.rebuild import rebuildGame


class TestRebuild(unittest.TestCase):

    def setUp(self):
        self.config = GameConfig()
        self.seats = ["human", "rule", "rule", "rule"]

    def build(self, handsPlayed=0, scores=None, moves=None, deckSeed=1):
        # a short way to call rebuildGame with sensible defaults in every test
        if scores is None:
            scores = [0, 0]
        if moves is None:
            moves = []
        return rebuildGame(deckSeed, 1, self.seats, handsPlayed, scores, moves)

    def legalSequence(self, count):
        # plays a few legal cards out of a fresh game and returns the list
        game = self.build()
        cards = []
        for step in range(count):
            seat = game.whoseTurn
            card = game.legalMoves(seat)[0]
            cards.append(card)
            game.applyMove(seat, card)
        return cards

    def testFreshDealGivesThirteenCardsPerSeat(self):
        game = self.build()
        for seat in range(self.config.numberOfPlayers):
            self.assertEqual(len(game.hands[seat]), self.config.cardsPerHand)

    def testSameStateRebuildsIdentically(self):
        first = self.build()
        second = self.build()
        self.assertEqual(first.hands, second.hands)
        
    def testDifferentDeckSeedGivesDifferentHands(self):
        first = self.build()
        second = self.build(deckSeed=2)
        self.assertNotEqual(first.hands, second.hands)

    def testReplayedMovesRemoveCards(self):
        moves = self.legalSequence(4)
        game = self.build(moves=moves)
        # one card from each seat, so every hand is one shorter
        for seat in range(self.config.numberOfPlayers):
            self.assertEqual(len(game.hands[seat]), self.config.cardsPerHand - 1)
        self.assertEqual(len(game.cardsPlayedSoFar()), 4)

    def testCompletedDealsChangeTheDealAndTheDealer(self):
        fresh = self.build()
        later = self.build(handsPlayed=2)
        self.assertNotEqual(fresh.hands, later.hands)
        self.assertEqual(later.dealerSeat, 2)
        self.assertEqual(later.handsPlayed, 2)

    def testScoresCarryThrough(self):
        game = self.build(scores=[3, 1])
        self.assertEqual(game.scores, [3, 1])

    def testAnIllegalMoveRaises(self):
        moves = self.legalSequence(3)

        # a card from another seat's hand can never be legal for the seat on the move
        game = self.build(moves=moves)
        otherSeat = (game.whoseTurn + 1) % self.config.numberOfPlayers
        impossible = game.hands[otherSeat][0]

        with self.assertRaises(ValueError):
            self.build(moves=moves + [impossible])