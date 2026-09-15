import unittest

from whist.engine.game import WhistGame
from whist.engine.rules import GameConfig
from whist.players.human import HumanPlayer # a seat that is not ready until a move is handed in


# ----- PLAYING ONE MOVE AT A TIME -----

class TestPlayOneMove(unittest.TestCase):

    def setUp(self):
        self.config = GameConfig()

    def testMovePlayedReportsSeatAndRemovesCard(self):
        game = WhistGame(self.config, deckSeed=1)
        game.dealHand()
        seat = game.whoseTurn

        event = game.playOneMove()
        self.assertEqual(event["what"], "movePlayed")
        self.assertEqual(event["seat"], seat)
        self.assertNotIn(event["card"], game.hands[seat])

    def testFourthMoveFinishesTheTrick(self):
        game = WhistGame(self.config, deckSeed=1)
        game.dealHand()

        # the first three plays do not finish a trick, so they report no trick
        for step in range(3):
            event = game.playOneMove()
            self.assertIsNone(event["trickCards"])
            self.assertIsNone(event["trickWinnerSeat"])

        fourth = game.playOneMove()
        self.assertEqual(len(fourth["trickCards"]), self.config.numberOfPlayers)
        self.assertIsNotNone(fourth["trickWinnerSeat"])

    def testNotReadyChangesNothingUntilAMoveIsHandedIn(self):
        humans = [] # a human in every seat, none of them ready yet
        for seat in range(self.config.numberOfPlayers):
            humans.append(HumanPlayer())
        game = WhistGame(self.config, deckSeed=1, players=humans)
        game.dealHand()

        seat = game.whoseTurn
        handBefore = list(game.hands[seat])

        event = game.playOneMove()
        self.assertEqual(event["what"], "notReady")
        self.assertEqual(game.whoseTurn, seat)
        self.assertEqual(game.hands[seat], handBefore)

        legalCard = game.legalMoves(seat)[0]
        game.players[seat].setMove(legalCard)
        after = game.playOneMove()
        self.assertEqual(after["what"], "movePlayed")

    def testHandFinishesExactlyOnceThenReportsHandOver(self):
        game = WhistGame(self.config, deckSeed=1)
        game.dealHand()

        handFinishedCount = 0
        event = game.playOneMove()
        while event["what"] != "handOver":
            if event["handFinished"]:
                handFinishedCount = handFinishedCount + 1
            event = game.playOneMove()
        self.assertEqual(handFinishedCount, 1)

        again = game.playOneMove()
        self.assertEqual(again["what"], "handOver") # keeps reporting handOver rather than scoring again

    def testPlayOutHandRaisesWhenASeatIsNotReady(self):
        humans = []
        for seat in range(self.config.numberOfPlayers):
            humans.append(HumanPlayer())
        game = WhistGame(self.config, deckSeed=1, players=humans)
        game.dealHand()

        # playOutHand cannot drive a not-ready seat, so it must raise rather than return early or hang
        with self.assertRaises(ValueError):
            game.playOutHand()


if __name__ == "__main__":
    unittest.main()
