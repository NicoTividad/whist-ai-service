import random
import unittest

from whist.engine.game import WhistGame
from whist.engine.rules import GameConfig
from whist.players.searchPlayer import SearchPlayer
from whist.players.randomPlayer import RandomPlayer
from whist.players.ruleBasedPlayer import RuleBasedPlayer


# ----- THE PLAIN SAMPLER -----

class TestPlainSampleDeal(unittest.TestCase):

    def setUp(self):
        self.config = GameConfig()
        self.game = WhistGame(self.config, deckSeed=1)
        self.game.dealHand()

    def playLegalMoves(self, count):
        # plays a few legal moves so mid-trick cases with uneven hand sizes are covered
        for step in range(count):
            seat = self.game.whoseTurn
            move = self.game.legalMoves(seat)[0]
            self.game.applyMove(seat, move)

    def testHandSizesMatchCardsLeftPerSeat(self):
        self.playLegalMoves(3)
        seat = self.game.whoseTurn
        view = self.game.viewFor(seat)
        player = SearchPlayer(random.Random(7), self.config)

        left = player.cardsLeftPerSeat(view)
        deal = player.plainSampleDeal(view)
        for otherSeat in range(self.config.numberOfPlayers):
            self.assertEqual(len(deal[otherSeat]), left[otherSeat])

    def testOwnHandMatchesView(self):
        seat = self.game.whoseTurn
        view = self.game.viewFor(seat)
        player = SearchPlayer(random.Random(7), self.config)

        deal = player.plainSampleDeal(view)
        self.assertCountEqual(deal[seat], view.ownHand) # this seat's own hand is kept exactly, not guessed

    def testNoDuplicatesAndNoPlayedCards(self):
        self.playLegalMoves(3) # create some already-played cards to check against
        seat = self.game.whoseTurn
        view = self.game.viewFor(seat)
        player = SearchPlayer(random.Random(7), self.config)

        deal = player.plainSampleDeal(view)
        allCards = []
        for otherSeat in range(self.config.numberOfPlayers):
            for card in deal[otherSeat]:
                allCards.append(card)
        self.assertEqual(len(allCards), len(set(allCards)))

        playedCards = []
        for play in view.cardsPlayedSoFar:
            playedCards.append(play[1])
        for card in playedCards:
            self.assertNotIn(card, allCards)

    def testSeedReproducesDealAndDifferentSeedDiffers(self):
        seat = self.game.whoseTurn
        view = self.game.viewFor(seat)

        playerA = SearchPlayer(random.Random(7), self.config)
        playerB = SearchPlayer(random.Random(7), self.config)
        dealA = playerA.plainSampleDeal(view)
        dealB = playerB.plainSampleDeal(view)
        self.assertEqual(dealA, dealB)

        playerC = SearchPlayer(random.Random(999), self.config)
        dealC = playerC.plainSampleDeal(view)
        self.assertNotEqual(dealA, dealC)


# ----- READING VOIDS AND THE VOID-AWARE SAMPLER -----

class TestVoidAwareSampling(unittest.TestCase):

    def setUp(self):
        self.config = GameConfig()
        self.game = WhistGame(self.config, deckSeed=1)
        self.game.dealHand()

    def playMoves(self, count):
        for step in range(count):
            seat = self.game.whoseTurn
            move = self.game.legalMoves(seat)[0]
            self.game.applyMove(seat, move)

    def playUntilVoid(self):
        # plays legal moves until some seat fails to follow the led suit, then stops
        while not self.game.handOver():
            seat = self.game.whoseTurn
            ledSuit = None
            if self.game.currentTrick:
                ledSuit = self.game.currentTrick[0][1].suit
            move = self.game.legalMoves(seat)[0]
            self.game.applyMove(seat, move)
            if ledSuit is not None and move.suit != ledSuit:
                return seat, ledSuit
        return None, None # the hand ended without a void being forced

    def testTricksFromHistorySplitsCorrectly(self):
        # four players, so six plays make one full trick of four plus a short trick of two
        self.playMoves(6)
        seat = self.game.whoseTurn
        view = self.game.viewFor(seat)
        player = SearchPlayer(random.Random(7), self.config)

        tricks = player.tricksFromHistory(view)
        self.assertEqual(len(tricks), 2)
        self.assertEqual(len(tricks[0]), 4)
        self.assertEqual(len(tricks[1]), 2)

    def testNoVoidsReportedBeforeAnyFailureToFollow(self):
        seat = self.game.whoseTurn
        view = self.game.viewFor(seat)
        player = SearchPlayer(random.Random(7), self.config)

        voids = player.voidSuitsPerSeat(view)
        for seatVoids in voids:
            self.assertEqual(seatVoids, [])

    def testVoidReportedForTheRightSeatOnly(self):
        voidSeat, voidSuit = self.playUntilVoid()
        self.assertIsNotNone(voidSeat) # the fixture must actually force a void

        view = self.game.viewFor(self.game.whoseTurn)
        player = SearchPlayer(random.Random(7), self.config)
        voids = player.voidSuitsPerSeat(view)

        self.assertIn(voidSuit, voids[voidSeat])
        totalVoids = 0
        for seatVoids in voids:
            totalVoids = totalVoids + len(seatVoids)
        self.assertEqual(totalVoids, 1) # we stopped at the first, so seats that followed show nothing

    def testSampledDealRespectsAKnownVoid(self):
        voidSeat, voidSuit = self.playUntilVoid()
        self.assertIsNotNone(voidSeat)

        # view from a seat other than the void seat, so the void seat's hand is guessed rather than known
        viewingSeat = (voidSeat + 1) % self.config.numberOfPlayers
        view = self.game.viewFor(viewingSeat)

        # check over several seeds so this is not a lucky single draw
        for seed in range(20):
            player = SearchPlayer(random.Random(seed), self.config)
            deal = player.sampleDeal(view)
            for card in deal[voidSeat]:
                self.assertNotEqual(card.suit, voidSuit)

    def testVoidInferenceIsOnByDefault(self):
        # this guards every result measured so far, which all assume the inference is on
        player = SearchPlayer(random.Random(7), self.config)
        self.assertTrue(player.useVoidInference)

    def testSwitchedOffSamplerMatchesPlainSampler(self):
        # with the switch off, sampleDeal must be exactly the plain sampler, drawn from the same seed
        seat = self.game.whoseTurn
        view = self.game.viewFor(seat)

        offPlayer = SearchPlayer(random.Random(7), self.config)
        offPlayer.useVoidInference = False
        offDeal = offPlayer.sampleDeal(view)

        plainPlayer = SearchPlayer(random.Random(7), self.config) # a fresh source with the same seed
        plainDeal = plainPlayer.plainSampleDeal(view)

        for otherSeat in range(self.config.numberOfPlayers):
            self.assertEqual(offDeal[otherSeat], plainDeal[otherSeat])

    def testSwitchedOffSamplerCanBreakAKnownVoid(self):
        voidSeat, voidSuit = self.playUntilVoid()
        self.assertIsNotNone(voidSeat)

        viewingSeat = (voidSeat + 1) % self.config.numberOfPlayers
        view = self.game.viewFor(viewingSeat)

        # with the switch off, at least one of twenty seeds should break the void
        brokeTheVoid = False
        for seed in range(20):
            player = SearchPlayer(random.Random(seed), self.config)
            player.useVoidInference = False
            deal = player.sampleDeal(view)
            for card in deal[voidSeat]:
                if card.suit == voidSuit:
                    brokeTheVoid = True
        self.assertTrue(brokeTheVoid) # proves the constraint is genuinely gone, not just unused


# ----- CHOOSING A MOVE BY SEARCH -----

class TestSearchGetMove(unittest.TestCase):

    def setUp(self):
        self.config = GameConfig()
        self.game = WhistGame(self.config, deckSeed=1)
        self.game.dealHand()

    def playMoves(self, count):
        for step in range(count):
            seat = self.game.whoseTurn
            move = self.game.legalMoves(seat)[0]
            self.game.applyMove(seat, move)

    def makeSearchPlayer(self, seed):
        # a search player with only a few samples so the tests stay fast
        player = SearchPlayer(random.Random(seed), self.config)
        player.numberOfSamples = 3
        return player

    def testGetMoveReturnsALegalCard(self):
        view = self.game.viewFor(self.game.whoseTurn)
        player = self.makeSearchPlayer(5)
        move = player.getMove(view)
        self.assertIn(move, view.legalMoves)

    def testSingleLegalMoveIsReturnedDirectly(self):
        # play down to the last trick, where every seat holds exactly one card
        self.playMoves((self.config.cardsPerHand - 1) * self.config.numberOfPlayers)
        view = self.game.viewFor(self.game.whoseTurn)
        self.assertEqual(len(view.legalMoves), 1) # a sanity check that only one move is possible

        player = self.makeSearchPlayer(5)
        move = player.getMove(view)
        self.assertEqual(move, view.legalMoves[0])

    def testSameSeedChoosesTheSameCard(self):
        view = self.game.viewFor(self.game.whoseTurn)
        playerA = self.makeSearchPlayer(5)
        playerB = self.makeSearchPlayer(5)
        self.assertEqual(playerA.getMove(view), playerB.getMove(view))

    def testSimulateOnceReturnsAPossibleScore(self):
        view = self.game.viewFor(self.game.whoseTurn)
        player = self.makeSearchPlayer(5)
        move = view.legalMoves[0]

        tricksLeft = self.config.cardsPerHand - len(player.completedTricks(view))
        result = player.simulateOnce(view, move)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, tricksLeft)

    def testGetMoveDoesNotChangeTheRealState(self):
        seat = self.game.whoseTurn
        view = self.game.viewFor(seat)
        ownBefore = list(view.ownHand)
        handsBefore = []
        for hand in self.game.hands:
            handsBefore.append(list(hand))

        player = self.makeSearchPlayer(5)
        player.getMove(view)
        self.assertEqual(view.ownHand, ownBefore) # the view's hand is untouched, so simulations used copies
        self.assertEqual(self.game.hands, handsBefore)


# ----- MEASURING TIME PER MOVE -----

class TestSearchTiming(unittest.TestCase):

    def setUp(self):
        self.config = GameConfig()
        self.game = WhistGame(self.config, deckSeed=1)
        self.game.dealHand()

    def testCountersStartAtZero(self):
        player = SearchPlayer(random.Random(5), self.config)
        self.assertEqual(player.movesMade, 0)
        self.assertEqual(player.totalMoveSeconds, 0.0)

    def testOneMoveUpdatesTheCounters(self):
        view = self.game.viewFor(self.game.whoseTurn)
        player = SearchPlayer(random.Random(5), self.config)
        player.numberOfSamples = 3

        player.getMove(view)
        self.assertEqual(player.movesMade, 1)
        self.assertGreater(player.totalMoveSeconds, 0.0)

    def testAverageIsZeroBeforeAnyMove(self):
        player = SearchPlayer(random.Random(5), self.config)
        self.assertEqual(player.averageMoveSeconds(), 0.0) # returns zero rather than dividing by zero


# ----- COUNTING THE SIMULATED MOVES (THE MACHINE-INDEPENDENT COST) -----

class TestSimulatedMoveCount(unittest.TestCase):

    def setUp(self):
        self.config = GameConfig()
        self.game = WhistGame(self.config, deckSeed=1)
        self.game.dealHand()

    def makeSearchPlayer(self, seed):
        player = SearchPlayer(random.Random(seed), self.config)
        player.numberOfSamples = 3
        return player

    def playMoves(self, count):
        for step in range(count):
            seat = self.game.whoseTurn
            move = self.game.legalMoves(seat)[0]
            self.game.applyMove(seat, move)

    def testCountStartsAtZero(self):
        player = self.makeSearchPlayer(5)
        self.assertEqual(player.simulatedMovesPlayed, 0)
        self.assertEqual(player.simulatedMovesPerRealMove(), 0.0)

    def testCountMatchesTheCardsActuallyPlayedOut(self):
        # a fresh deal makes the count exact, samples times moves times cards held
        view = self.game.viewFor(self.game.whoseTurn)
        cardsStillHeld = 0
        for hand in self.game.hands:
            cardsStillHeld = cardsStillHeld + len(hand)
        self.assertEqual(cardsStillHeld, 52) # a sanity check that this really is a fresh deal

        player = self.makeSearchPlayer(5)
        player.getMove(view)

        expected = player.numberOfSamples * len(view.legalMoves) * cardsStillHeld
        self.assertGreater(player.simulatedMovesPlayed, 0)
        self.assertEqual(player.simulatedMovesPlayed, expected)
        self.assertEqual(player.simulatedMovesPerRealMove(), expected) # one decision, so the mean is the total

    def testSingleLegalMoveCountsNoSimulatedMoves(self):
        # a single legal move skips the search, so no simulated move is counted
        self.playMoves((self.config.cardsPerHand - 1) * self.config.numberOfPlayers)
        view = self.game.viewFor(self.game.whoseTurn)
        self.assertEqual(len(view.legalMoves), 1)

        player = self.makeSearchPlayer(5)
        player.getMove(view)
        self.assertEqual(player.movesMade, 1) # the decision still counts as a real move
        self.assertEqual(player.simulatedMovesPlayed, 0)


# ----- COMBINING SAMPLES BY THE SELECTION RULE -----

class TestSelectionRule(unittest.TestCase):

    def setUp(self):
        self.config = GameConfig() # default full-game settings
        self.scores = [2, 5, 3, 1] # one list with clearly different values, so the three rules differ

    def testSelectionRuleDefaultsToAverage(self):
        # this guards every result measured so far, which all assume averaging
        player = SearchPlayer(random.Random(5), self.config)
        self.assertEqual(player.selectionRule, "average")

    def testAverageRuleReturnsTheMean(self):
        player = SearchPlayer(random.Random(5), self.config)
        player.selectionRule = "average"
        self.assertEqual(player.aggregateScores(self.scores), 11 / 4) # the mean of 2, 5, 3 and 1

    def testBestCaseReturnsTheHighestAndWorstCaseTheLowest(self):
        player = SearchPlayer(random.Random(5), self.config)
        player.selectionRule = "bestCase"
        self.assertEqual(player.aggregateScores(self.scores), 5)
        player.selectionRule = "worstCase"
        self.assertEqual(player.aggregateScores(self.scores), 1)

    def testUnknownSelectionRuleRaises(self):
        player = SearchPlayer(random.Random(5), self.config)
        player.selectionRule = "nonsense"
        with self.assertRaises(ValueError): # a bad rule must fail loudly, not average silently
            player.aggregateScores(self.scores)


# ----- CHOOSING THE PLAYOUT POLICY -----

class TestPlayoutPolicy(unittest.TestCase):

    def setUp(self):
        self.config = GameConfig()

    def testRulePlayoutsAreOffByDefault(self):
        # this guards every result measured so far, which all assume random playouts
        player = SearchPlayer(random.Random(5), self.config)
        self.assertFalse(player.useRulePlayouts)

    def testDefaultPlayoutsAreRandomPlayers(self):
        player = SearchPlayer(random.Random(5), self.config)
        players = player.playoutPlayers()
        self.assertEqual(len(players), self.config.numberOfPlayers)
        for seatPlayer in players:
            self.assertIsInstance(seatPlayer, RandomPlayer)

    def testRulePlayoutsAreRuleBasedPlayers(self):
        player = SearchPlayer(random.Random(5), self.config)
        player.useRulePlayouts = True
        players = player.playoutPlayers()
        self.assertEqual(len(players), self.config.numberOfPlayers)
        for seatPlayer in players:
            self.assertIsInstance(seatPlayer, RuleBasedPlayer)


if __name__ == "__main__":
    unittest.main()
