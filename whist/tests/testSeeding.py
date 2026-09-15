import unittest

from whist.engine.game import WhistGame
from whist.engine.rules import GameConfig
from whist.players.ruleBasedPlayer import RuleBasedPlayer # a non-random player to prove the deal ignores who is seated


# ----- THE DECK SEED AND PLAYER SEED ARE SEPARATE -----

class TestSeeding(unittest.TestCase):

    def setUp(self):
        self.config = GameConfig()

    def testSameDeckSeedGivesSameDealWhoeverIsSeated(self):
        gameA = WhistGame(self.config, deckSeed=1) # fallback random players
        gameA.dealHand()

        rulePlayers = [RuleBasedPlayer() for seat in range(self.config.numberOfPlayers)]
        gameB = WhistGame(self.config, deckSeed=1, players=rulePlayers)
        gameB.dealHand()

        self.assertEqual(gameA.hands, gameB.hands)

    def testChangingPlayerSeedDoesNotChangeDeal(self):
        gameA = WhistGame(self.config, deckSeed=1, playerSeed=1)
        gameA.dealHand()
        gameB = WhistGame(self.config, deckSeed=1, playerSeed=999)
        gameB.dealHand()
        self.assertEqual(gameA.hands, gameB.hands)

    def testChangingDeckSeedChangesDeal(self):
        gameA = WhistGame(self.config, deckSeed=1, playerSeed=1)
        gameA.dealHand()
        gameB = WhistGame(self.config, deckSeed=2, playerSeed=1)
        gameB.dealHand()
        self.assertNotEqual(gameA.hands, gameB.hands)

    def testEachSeatHasItsOwnRandomSource(self):
        game = WhistGame(self.config, deckSeed=1, playerSeed=1)
        sequences = []
        for seat in range(self.config.numberOfPlayers):
            numbers = [game.seatRngs[seat].randrange(1000) for draw in range(5)]
            sequences.append(numbers)
        # they should not all be identical; if they were, the seats would share one source
        allTheSame = True
        for seat in range(1, self.config.numberOfPlayers):
            if sequences[seat] != sequences[0]:
                allTheSame = False
        self.assertFalse(allTheSame)


if __name__ == "__main__":
    unittest.main()
