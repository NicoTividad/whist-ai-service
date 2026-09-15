import unittest

from whist.engine.game import WhistGame, gameFromPosition
from whist.engine.rules import GameConfig
from whist.engine.cards import Card
from whist.players.ruleBasedPlayer import RuleBasedPlayer


# ----- BUILDING A GAME FROM A POSITION -----

class TestGameFromPosition(unittest.TestCase):

    def setUp(self):
        self.config = GameConfig()

    def makeRulePlayers(self):
        # four deterministic rule-based players, so a play-out gives the same result every time
        players = []
        for seat in range(self.config.numberOfPlayers):
            players.append(RuleBasedPlayer())
        return players

    def buildPartwayGame(self):
        # deals and plays part of a hand so there is a real mid-trick position to snapshot
        game = WhistGame(self.config, deckSeed=1, players=self.makeRulePlayers())
        game.dealHand()
        for step in range(10): # ten moves leaves two completed tricks and two cards in the next
            seat = game.whoseTurn
            view = game.viewFor(seat)
            card = game.players[seat].getMove(view)
            game.applyMove(seat, card)
        return game

    def snapshotOf(self, game):
        # copies the visible state of a game into a plain dict, so it can be rebuilt independently
        hands = []
        for hand in game.hands:
            hands.append(list(hand))
        trickHistory = []
        for trick in game.trickHistory:
            trickHistory.append(list(trick))
        snapshot = {
            "hands": hands,
            "trumpSuit": game.trumpSuit,
            "currentTrick": list(game.currentTrick),
            "trickHistory": trickHistory,
            "tricksWon": list(game.tricksWon),
            "scores": list(game.scores),
            "whoseTurn": game.whoseTurn,
        }
        return snapshot

    def gameFromSnapshot(self, snapshot):
        # with its own fresh rule-based players
        return gameFromPosition(
            self.config,
            snapshot["hands"],
            snapshot["trumpSuit"],
            snapshot["whoseTurn"],
            currentTrick=snapshot["currentTrick"],
            trickHistory=snapshot["trickHistory"],
            tricksWon=snapshot["tricksWon"],
            scores=snapshot["scores"],
            players=self.makeRulePlayers(),
        )

    def testRebuiltGameReportsSameVisibleState(self):
        played = self.buildPartwayGame()
        snapshot = self.snapshotOf(played)
        rebuilt = self.gameFromSnapshot(snapshot)

        seat = snapshot["whoseTurn"]
        self.assertEqual(played.cardsPlayedSoFar(), rebuilt.cardsPlayedSoFar())
        self.assertEqual(played.legalMoves(seat), rebuilt.legalMoves(seat))

        viewA = played.viewFor(seat)
        viewB = rebuilt.viewFor(seat)
        self.assertEqual(viewA.ownHand, viewB.ownHand)
        self.assertEqual(viewA.trumpSuit, viewB.trumpSuit)
        self.assertEqual(viewA.currentTrick, viewB.currentTrick)
        self.assertEqual(viewA.cardsPlayedSoFar, viewB.cardsPlayedSoFar)
        self.assertEqual(viewA.scores, viewB.scores)
        self.assertEqual(viewA.whoseTurn, viewB.whoseTurn)
        self.assertEqual(viewA.legalMoves, viewB.legalMoves)

    def testBothGamesPlayOutIdentically(self):
        played = self.buildPartwayGame()
        snapshot = self.snapshotOf(played)
        rebuilt = self.gameFromSnapshot(snapshot)

        played.playOutHand()
        rebuilt.playOutHand()
        self.assertEqual(played.tricksWon, rebuilt.tricksWon)

    def testHelperDoesNotChangeCallersLists(self):
        played = self.buildPartwayGame()
        snapshot = self.snapshotOf(played)

        originalHands = [] # an independent copy to compare against afterwards
        for hand in snapshot["hands"]:
            originalHands.append(list(hand))
        originalTrick = list(snapshot["currentTrick"])

        rebuilt = self.gameFromSnapshot(snapshot)
        rebuilt.playOutHand() # this removes cards from the built game's hands as it plays

        self.assertEqual(snapshot["hands"], originalHands)
        self.assertEqual(snapshot["currentTrick"], originalTrick)

    def testLeaderIsWorkedOutFromTheTrick(self):
        emptyHands = [] # any hands will do, since we are not playing here
        for seat in range(self.config.numberOfPlayers):
            emptyHands.append([])

        # an empty trick means the seat on the move is leading
        leadingGame = gameFromPosition(self.config, emptyHands, "spades", 2)
        self.assertEqual(leadingGame.leader, 2)

        # a trick with cards already in it means the leader is the seat that played first
        trick = [(1, Card("hearts", 5)), (2, Card("hearts", 9))]
        followingGame = gameFromPosition(self.config, emptyHands, "spades", 3, currentTrick=trick)
        self.assertEqual(followingGame.leader, 1)


if __name__ == "__main__":
    unittest.main()
