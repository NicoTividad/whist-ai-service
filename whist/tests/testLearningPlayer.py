# the positions here are built by hand rather than dealt, so every expected feature can be worked
# out on paper and checked exactly. nothing in these tests reads another seat's real hand.

import os
import random
import tempfile
import unittest

from whist.engine.rules import GameConfig
from whist.engine.cards import Card
from whist.engine.playerView import PlayerView
from whist.engine.game import WhistGame, makeSeatRngs
from whist.players.learningPlayer import LearningPlayer, numberOfFeatures, numberOfWeights, biasIndex


# ----- SHARED POSITION BUILDERS -----

class LearningPlayerTestBase(unittest.TestCase):

    def setUp(self):
        self.config = GameConfig() # so trump is a real suit and there are 4 seats
        self.player = LearningPlayer(random.Random(11), self.config)

    def makeView(self, seat, ownHand, trumpSuit, currentTrick, cardsPlayedSoFar, scores):
        # builds one position by hand; legalMoves is set to the whole hand
        return PlayerView(
            seat=seat,
            ownHand=ownHand,
            trumpSuit=trumpSuit,
            currentTrick=currentTrick,
            cardsPlayedSoFar=cardsPlayedSoFar,
            scores=scores,
            whoseTurn=seat,
            legalMoves=list(ownHand),
        )

    def leadingView(self):
        # position A: seat 0 leads an untouched deal, trump is spades, scores level
        hand = [Card("hearts", 14), Card("hearts", 2), Card("spades", 5), Card("clubs", 9)]
        return self.makeView(0, hand, "spades", [], [], [0, 0])

    def followingView(self):
        # position B: seat 3 follows its winning partner and is void in hearts
        firstTrick = [
            (1, Card("hearts", 13)),
            (2, Card("hearts", 3)),
            (3, Card("clubs", 4)),
            (0, Card("hearts", 5)),
        ]
        currentTrick = [(1, Card("diamonds", 10)), (2, Card("diamonds", 2))]
        history = firstTrick + currentTrick # cardsPlayedSoFar includes the trick in progress
        hand = [Card("diamonds", 14), Card("diamonds", 6), Card("spades", 7), Card("clubs", 9)]
        return self.makeView(3, hand, "spades", currentTrick, history, [1, 3])

    def learningGame(self, deckSeed, playerSeed=1):
        # a game with a learning agent in every seat, so any seat can be inspected
        seatRngs = makeSeatRngs(playerSeed, self.config.numberOfPlayers)
        players = []
        for seat in range(self.config.numberOfPlayers):
            players.append(LearningPlayer(seatRngs[seat], self.config))
        return WhistGame(self.config, deckSeed=deckSeed, playerSeed=playerSeed, players=players)

    def playOneDeal(self, game):
        # the engine does not push events to players itself, so the caller has to
        game.dealHand()
        while not game.handOver():
            event = game.playOneMove()
            for player in game.players:
                player.observe(event)

    def makeFeatures(self, pairs):
        # synthetic rather than dealt, so the arithmetic in the update tests can be done on paper
        features = [0.0 for slot in range(numberOfFeatures)]
        for index in pairs:
            features[index] = pairs[index]
        return features

    def makeExample(self, pairs, dealReturn):
        example = {}
        example["features"] = self.makeFeatures(pairs)
        example["dealReturn"] = dealReturn
        return example

    def totalSquaredError(self, player, examples):
        total = 0.0
        for example in examples:
            error = example["dealReturn"] - player.scoreFromFeatures(example["features"])
            total = total + error * error
        return total

    def voidShownView(self):
        # position C: seat 2 has shown a hearts void and seat 0 leads
        firstTrick = [
            (1, Card("hearts", 3)),
            (2, Card("clubs", 4)),
            (3, Card("hearts", 8)),
            (0, Card("hearts", 13)), # highest heart, no trump played, so seat 0 won and leads next
        ]
        hand = [Card("hearts", 14), Card("hearts", 2), Card("spades", 7), Card("clubs", 9)]
        return self.makeView(0, hand, "spades", [], firstTrick, [0, 0])


# ----- SHAPE AND RANGE -----

class TestFeatureShape(LearningPlayerTestBase):

    def testVectorHasTheRightLength(self):
        for view in [self.leadingView(), self.followingView(), self.voidShownView()]:
            for move in view.ownHand:
                features = self.player.featuresFor(view, move)
                self.assertEqual(len(features), numberOfFeatures)

    def testEveryValueIsBetweenZeroAndOne(self):
        # a feature outside this range would dominate the linear model by scale alone
        for view in [self.leadingView(), self.followingView(), self.voidShownView()]:
            for move in view.ownHand:
                for value in self.player.featuresFor(view, move):
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

    def testEveryValueIsAFloat(self):
        # the model multiplies these by weights, so they must all be numbers of one kind
        view = self.followingView()
        for value in self.player.featuresFor(view, Card("diamonds", 14)):
            self.assertIsInstance(value, float)


# ----- THE MOVE ITSELF (FEATURES 1 TO 5) -----

class TestMoveFeatures(LearningPlayerTestBase):

    def testTrumpFlagIsSetForATrumpMove(self):
        view = self.leadingView()
        features = self.player.featuresFor(view, Card("spades", 5))
        self.assertEqual(features[1], 1.0)

    def testTrumpFlagIsClearForANonTrumpMove(self):
        view = self.leadingView()
        features = self.player.featuresFor(view, Card("hearts", 14))
        self.assertEqual(features[1], 0.0)

    def testScaledRankRunsFromZeroToOne(self):
        view = self.leadingView() # deckRanks is 2..14, so 2 scales to 0.0 and 14 to 1.0
        self.assertEqual(self.player.featuresFor(view, Card("hearts", 2))[0], 0.0)
        self.assertEqual(self.player.featuresFor(view, Card("hearts", 14))[0], 1.0)
        self.assertAlmostEqual(self.player.featuresFor(view, Card("spades", 5))[0], 0.25) # (5-2)/12

    def testHighestAndLowestOfSuitFlags(self):
        view = self.leadingView() # the hand holds hearts 14 and hearts 2
        highest = self.player.featuresFor(view, Card("hearts", 14))
        lowest = self.player.featuresFor(view, Card("hearts", 2))
        self.assertEqual(highest[2], 1.0)
        self.assertEqual(highest[3], 0.0)
        self.assertEqual(lowest[2], 0.0)
        self.assertEqual(lowest[3], 1.0)

    def testASingletonIsBothHighestAndLowestOfItsSuit(self):
        view = self.leadingView() # only one spade and one club are held
        features = self.player.featuresFor(view, Card("clubs", 9))
        self.assertEqual(features[2], 1.0)
        self.assertEqual(features[3], 1.0)

    def testALeadingMoveNeverWinsTheTrickYet(self):
        view = self.leadingView()
        for move in view.ownHand:
            self.assertEqual(self.player.featuresFor(view, move)[4], 0.0)

    def testAFollowingMoveThatTakesTheTrickSetsTheFlag(self):
        view = self.followingView() # diamonds 10 is winning, no trump played
        self.assertEqual(self.player.featuresFor(view, Card("diamonds", 14))[4], 1.0)
        self.assertEqual(self.player.featuresFor(view, Card("spades", 7))[4], 1.0)

    def testAFollowingMoveThatLosesTheTrickClearsTheFlag(self):
        view = self.followingView()
        self.assertEqual(self.player.featuresFor(view, Card("diamonds", 6))[4], 0.0)
        self.assertEqual(self.player.featuresFor(view, Card("clubs", 9))[4], 0.0)


# ----- THE TRICK IN PROGRESS (FEATURES 6 TO 8) -----

class TestTrickFeatures(LearningPlayerTestBase):

    def testTrickFullnessScalesOverTheSeats(self):
        self.assertEqual(self.player.featuresFor(self.leadingView(), Card("hearts", 14))[5], 0.0)
        self.assertEqual(self.player.featuresFor(self.followingView(), Card("diamonds", 14))[5], 0.5) # 2 of 4

    def testTrumpPlayedFlagIsClearWhenNoTrumpIsInTheTrick(self):
        view = self.followingView() # the trick holds two diamonds and trump is spades
        self.assertEqual(self.player.featuresFor(view, Card("diamonds", 14))[6], 0.0)

    def testTrumpPlayedFlagIsSetWhenATrumpIsInTheTrick(self):
        # the same position but with a spade led into the trick instead
        currentTrick = [(1, Card("spades", 4)), (2, Card("spades", 9))]
        hand = [Card("spades", 7), Card("clubs", 9)]
        view = self.makeView(3, hand, "spades", currentTrick, list(currentTrick), [0, 0])
        self.assertEqual(self.player.featuresFor(view, Card("spades", 7))[6], 1.0)

    def testPartnerWinningFlagIsSetWhenThePartnerLeads(self):
        # seat 3's partner is seat 1, which played the winning diamonds 10
        view = self.followingView()
        self.assertEqual(self.player.featuresFor(view, Card("diamonds", 14))[7], 1.0)

    def testPartnerWinningFlagIsClearWhenAnOpponentLeads(self):
        # the same trick, but seat 2 now holds the winning card, and seat 2 is an opponent of seat 3
        currentTrick = [(1, Card("diamonds", 2)), (2, Card("diamonds", 10))]
        hand = [Card("diamonds", 14), Card("spades", 7)]
        view = self.makeView(3, hand, "spades", currentTrick, list(currentTrick), [0, 0])
        self.assertEqual(self.player.featuresFor(view, Card("diamonds", 14))[7], 0.0)

    def testPartnerWinningFlagIsClearWhenLeading(self):
        view = self.leadingView()
        self.assertEqual(self.player.featuresFor(view, Card("hearts", 14))[7], 0.0)


# ----- THE HAND AND THE HISTORY (FEATURES 9 TO 12) -----

class TestHandAndHistoryFeatures(LearningPlayerTestBase):

    def testSuitAndTrumpSharesOfTheHand(self):
        view = self.leadingView() # 4 cards: two hearts, one spade, one club
        features = self.player.featuresFor(view, Card("hearts", 14))
        # scaled by a full suit so the meaning holds as the hand shrinks
        self.assertAlmostEqual(features[8], 2 / 13)
        self.assertAlmostEqual(features[9], 1 / 13)

    def testShareOfTheSuitAlreadyPlayed(self):
        view = self.voidShownView() # three hearts were played in the completed trick
        features = self.player.featuresFor(view, Card("hearts", 14))
        self.assertAlmostEqual(features[10], 3 / 13)

    def testNothingPlayedGivesAZeroShare(self):
        view = self.leadingView()
        self.assertEqual(self.player.featuresFor(view, Card("hearts", 14))[10], 0.0)

    def testVoidFlagFiresWhenASeatHasShownVoid(self):
        # seat 2 discarded a club on a hearts lead, so a hearts void is public knowledge
        view = self.voidShownView()
        self.assertEqual(self.player.featuresFor(view, Card("hearts", 14))[11], 1.0)

    def testVoidFlagIsClearForASuitNobodyHasFailedToFollow(self):
        # clubs was never led in that position, so no club void has been shown
        view = self.voidShownView()
        self.assertEqual(self.player.featuresFor(view, Card("clubs", 9))[11], 0.0)

    def testVoidFlagIsClearOnAnUntouchedDeal(self):
        view = self.leadingView()
        for move in view.ownHand:
            self.assertEqual(self.player.featuresFor(view, move)[11], 0.0)


# ----- THE SCORE (FEATURE 13) -----

class TestScoreFeature(LearningPlayerTestBase):

    def testLevelScoreGivesAHalf(self):
        view = self.leadingView()
        self.assertEqual(self.player.featuresFor(view, Card("hearts", 14))[12], 0.5)

    def testBeingAheadGivesOne(self):
        # seat 3 is on side 1, and side 1 leads 3 to 1 in the following position
        view = self.followingView()
        self.assertEqual(self.player.featuresFor(view, Card("diamonds", 14))[12], 1.0)

    def testBeingBehindGivesZero(self):
        # the same seat and side, with the scores turned around
        hand = [Card("diamonds", 14), Card("spades", 7)]
        view = self.makeView(3, hand, "spades", [], [], [4, 1])
        self.assertEqual(self.player.featuresFor(view, Card("diamonds", 14))[12], 0.0)


# ----- THE WHOLE VECTOR ON ONE KNOWN POSITION -----

class TestWholeVector(LearningPlayerTestBase):

    def testLeadingPositionGivesTheExpectedVector(self):
        # every value here was worked out by hand from position A, so the order is pinned down too
        view = self.leadingView()
        expected = [
            1.0, # 1. rank 14 is the top of the 2..14 range
            0.0, # 2. hearts is not the trump suit
            1.0, # 3. the ace is the highest heart held
            0.0, # 4. and not the lowest
            0.0, # 5. leading, so nothing is being beaten
            0.0, # 6. the trick is empty
            0.0, # 7. no trump has been played into it
            0.0, # 8. nobody is winning an empty trick
            2 / 13, # 9. two of a full suit of hearts are held
            1 / 13, # 10. one of a full suit of trumps is held
            0.0, # 11. no hearts have been played this deal
            0.0, # 12. no void has been shown
            0.5, # 13. the scores are level
        ]
        actual = self.player.featuresFor(view, Card("hearts", 14))
        self.assertEqual(len(actual), len(expected))
        for index in range(len(expected)):
            self.assertAlmostEqual(actual[index], expected[index])


# ----- GETMOVE RETURNS A LEGAL MOVE -----

class TestGetMoveUnchanged(LearningPlayerTestBase):

    def testGetMoveReturnsALegalMove(self):
        game = WhistGame(self.config, deckSeed=1)
        game.dealHand()
        seat = game.whoseTurn
        view = game.viewFor(seat)
        player = LearningPlayer(random.Random(3), self.config)
        move = player.getMove(view)
        self.assertIn(move, game.legalMoves(seat))

    def testGetMoveIsReproducibleFromTheSeatRng(self):
        game = WhistGame(self.config, deckSeed=1)
        game.dealHand()
        view = game.viewFor(game.whoseTurn)
        first = LearningPlayer(random.Random(3), self.config)
        second = LearningPlayer(random.Random(3), self.config)
        self.assertEqual(first.getMove(view), second.getMove(view))


# ----- THE WEIGHTS VECTOR -----

class TestWeightsVector(LearningPlayerTestBase):

    def testWeightsHaveOneSlotPerFeaturePlusABias(self):
        self.assertEqual(numberOfWeights, numberOfFeatures + 1)
        self.assertEqual(len(self.player.weights), numberOfWeights)
        self.assertEqual(biasIndex, numberOfFeatures)

    def testEveryWeightStartsAtZero(self):
        # a default guard, a changed initialisation changes every trained result
        for weight in self.player.weights:
            self.assertEqual(weight, 0.0)


# ----- SCORING A MOVE -----

class TestScoreMove(LearningPlayerTestBase):

    def testScoreIsBiasPlusTheWeightedSum(self):
        # 0.5 + 2.0 * 1.0 + 3.0 * 1.0 + 10.0 * 0.5 = 10.5
        view = self.leadingView()
        self.player.weights[0] = 2.0 # rewards the move's scaled rank
        self.player.weights[2] = 3.0 # rewards being the highest card of its suit in hand
        self.player.weights[12] = 10.0 # rewards this seat's side being ahead
        self.player.weights[biasIndex] = 0.5
        self.assertAlmostEqual(self.player.scoreMove(view, Card("hearts", 14)), 10.5)

    def testScoreIsTheBiasAloneWhenAllFeatureWeightsAreZero(self):
        view = self.leadingView()
        self.player.weights[biasIndex] = -1.5
        for move in view.ownHand:
            self.assertAlmostEqual(self.player.scoreMove(view, move), -1.5)

    def testEveryMoveScoresZeroWithUntrainedWeights(self):
        view = self.leadingView()
        for move in view.legalMoves:
            self.assertEqual(self.player.scoreMove(view, move), 0.0)


# ----- GREEDY CHOICE AND THE TIE BREAK -----

class TestGreedyChoice(LearningPlayerTestBase):

    def testZeroWeightsStillReturnALegalMove(self):
        # every move ties at zero, so the tie break alone decides, and it must stay inside the rules
        view = self.leadingView()
        for seed in range(20):
            player = LearningPlayer(random.Random(seed), self.config)
            self.assertIn(player.getMove(view), view.legalMoves)

    def testZeroWeightsChoiceIsNotAlwaysTheFirstLegalMove(self):
        # the tie break must use the seat rng, not the first legal move
        view = self.leadingView()
        chosen = []
        for seed in range(50):
            player = LearningPlayer(random.Random(seed), self.config)
            chosen.append(player.getMove(view))
        self.assertGreater(len(set(chosen)), 1)
        self.assertEqual(len(set(chosen)), len(view.legalMoves))

    def testWeightsThatRewardTrumpsPickTheTrump(self):
        # the only trump in hand becomes the single best move, no tie
        view = self.leadingView()
        self.player.weights[1] = 1.0 # feature 2 is the trump flag
        self.assertEqual(self.player.getMove(view), Card("spades", 5))

    def testWeightsThatRewardRankPickTheHighestCard(self):
        # rewarding feature 1 makes the highest ranked card the unique best, which is the ace of hearts
        view = self.leadingView()
        self.player.weights[0] = 1.0 # feature 1 is the scaled rank
        self.assertEqual(self.player.getMove(view), Card("hearts", 14))

    def testNegativeWeightsPickTheOppositeCard(self):
        # the same reward turned around must pick the lowest ranked card instead, which is the two of hearts
        view = self.leadingView()
        self.player.weights[0] = -1.0
        self.assertEqual(self.player.getMove(view), Card("hearts", 2))


# ----- SAVING AND LOADING THE WEIGHTS -----

class TestSaveAndLoadWeights(LearningPlayerTestBase):

    def someTrainedLookingWeights(self):
        # a spread of positive, negative and fractional values, so a lazy save would show up
        weights = []
        for slot in range(numberOfWeights):
            weights.append((slot + 1) * 0.37 - 2.0)
        weights[3] = -0.1
        weights[7] = 1 / 3 # a value with no exact decimal form, to check the round trip
        return weights

    def testSaveThenLoadGivesIdenticalWeights(self):
        saved = LearningPlayer(random.Random(1), self.config)
        saved.weights = self.someTrainedLookingWeights()
        with tempfile.TemporaryDirectory() as folder: # nothing is left on disk, and never in results/
            filePath = os.path.join(folder, "learningWeights.txt")
            saved.saveWeights(filePath)
            loaded = LearningPlayer(random.Random(2), self.config)
            loaded.loadWeights(filePath)
            self.assertEqual(loaded.weights, saved.weights) # identical, not merely close

    def testLoadedPlayerScoresAPositionIdentically(self):
        view = self.followingView()
        saved = LearningPlayer(random.Random(1), self.config)
        saved.weights = self.someTrainedLookingWeights()
        with tempfile.TemporaryDirectory() as folder:
            filePath = os.path.join(folder, "learningWeights.txt")
            saved.saveWeights(filePath)
            loaded = LearningPlayer(random.Random(2), self.config)
            loaded.loadWeights(filePath)
            for move in view.legalMoves:
                self.assertEqual(loaded.scoreMove(view, move), saved.scoreMove(view, move))

    def testSavedFileHasOneNumberPerLine(self):
        # the file is meant to be readable by eye, so check its shape rather than just its round trip
        with tempfile.TemporaryDirectory() as folder:
            filePath = os.path.join(folder, "learningWeights.txt")
            self.player.saveWeights(filePath)
            handle = open(filePath, "r")
            lines = handle.readlines()
            handle.close()
            self.assertEqual(len(lines), numberOfWeights)
            for line in lines:
                float(line.strip())

    def testLoadingAWrongSizedFileFails(self):
        # a short file must raise rather than quietly leave the agent scoring nothing like the saved one
        with tempfile.TemporaryDirectory() as folder:
            filePath = os.path.join(folder, "tooFewWeights.txt")
            handle = open(filePath, "w")
            handle.write("0.5\n1.5\n")
            handle.close()
            player = LearningPlayer(random.Random(1), self.config)
            self.assertRaises(ValueError, player.loadWeights, filePath)

    def testAFailedLoadLeavesTheWeightsAlone(self):
        # the size check happens before the weights are replaced, so a bad file cannot half-load
        with tempfile.TemporaryDirectory() as folder:
            filePath = os.path.join(folder, "tooFewWeights.txt")
            handle = open(filePath, "w")
            handle.write("0.5\n")
            handle.close()
            player = LearningPlayer(random.Random(1), self.config)
            player.weights[0] = 9.0
            self.assertRaises(ValueError, player.loadWeights, filePath)
            self.assertEqual(player.weights[0], 9.0)
            self.assertEqual(len(player.weights), numberOfWeights)


if __name__ == "__main__":
    unittest.main()


# ----- RECORDING TRAINING EXAMPLES -----

class TestRecordingExamples(LearningPlayerTestBase):

    def testOneExamplePerMoveTheSeatMade(self):
        # a four player deal is 13 tricks, so each seat plays 13 cards and records 13 examples
        game = self.learningGame(1)
        self.playOneDeal(game)
        for player in game.players:
            self.assertEqual(len(player.getTrainingExamples()), self.config.cardsPerHand)

    def testEveryExampleHasTheRightFeatureLength(self):
        # the examples must be shaped for the update, which expects one weight per feature
        game = self.learningGame(1)
        self.playOneDeal(game)
        for example in game.players[0].getTrainingExamples():
            self.assertEqual(len(example["features"]), numberOfFeatures)

    def testEveryExampleFromOneDealCarriesTheSameReturn(self):
        # monte carlo control pairs every move of a deal with that deal's single outcome
        game = self.learningGame(1)
        self.playOneDeal(game)
        returns = []
        for example in game.players[0].getTrainingExamples():
            returns.append(example["dealReturn"])
        self.assertEqual(len(set(returns)), 1)

    def testReturnEqualsTheTricksThatSideWon(self):
        # checked against the engine's own count, which is the independent source of truth
        game = self.learningGame(1)
        self.playOneDeal(game)
        for seat in range(self.config.numberOfPlayers):
            player = game.players[seat]
            expected = game.tricksWon[game.sideOf(seat)]
            self.assertEqual(player.getTrainingExamples()[0]["dealReturn"], expected)

    def testTheTwoSidesGetDifferentReturns(self):
        # a constant would pass the tests above, the two sides can never tie
        game = self.learningGame(1)
        self.playOneDeal(game)
        northReturn = game.players[0].getTrainingExamples()[0]["dealReturn"] # side 0
        eastReturn = game.players[1].getTrainingExamples()[0]["dealReturn"] # side 1
        self.assertEqual(northReturn + eastReturn, self.config.cardsPerHand)
        self.assertNotEqual(northReturn, eastReturn) # an odd total cannot split evenly

    def testPerDealListIsClearedAtTheDealBoundary(self):
        game = self.learningGame(1)
        self.playOneDeal(game)
        for player in game.players:
            self.assertEqual(player.dealFeatures, [])
            self.assertEqual(player.dealTrickWinners, [])

    def testASecondDealRecordsOnlyItsOwnMoves(self):
        # two deals must not bleed into one another: the store grows by exactly one deal's worth
        game = self.learningGame(1)
        self.playOneDeal(game)
        afterFirst = len(game.players[0].getTrainingExamples())

        game.finishHand() # score the first deal and rotate the dealer, as playHand would
        self.playOneDeal(game)
        afterSecond = len(game.players[0].getTrainingExamples())

        self.assertEqual(afterFirst, self.config.cardsPerHand)
        self.assertEqual(afterSecond, 2 * self.config.cardsPerHand)
        self.assertEqual(game.players[0].dealFeatures, [])

    def testTheTwoDealsCanCarryDifferentReturns(self):
        game = self.learningGame(1)
        self.playOneDeal(game)
        firstReturn = game.players[0].getTrainingExamples()[0]["dealReturn"]
        firstDealTricks = game.tricksWon[game.sideOf(0)]

        game.finishHand()
        self.playOneDeal(game)
        secondReturn = game.players[0].getTrainingExamples()[-1]["dealReturn"]
        secondDealTricks = game.tricksWon[game.sideOf(0)]

        self.assertEqual(firstReturn, firstDealTricks)
        self.assertEqual(secondReturn, secondDealTricks)

    def testNothingIsRecordedUntilTheDealEnds(self):
        # the examples are only created at the deal boundary, so a part-played deal has none
        game = self.learningGame(1)
        game.dealHand()
        for step in range(4):
            event = game.playOneMove()
            for player in game.players:
                player.observe(event)
        self.assertEqual(game.players[0].getTrainingExamples(), [])
        self.assertEqual(len(game.players[0].dealFeatures), 1) # but this seat's own move is remembered

    def testClearTrainingExamplesEmptiesTheStore(self):
        game = self.learningGame(1)
        self.playOneDeal(game)
        player = game.players[0]
        self.assertGreater(len(player.getTrainingExamples()), 0)
        player.clearTrainingExamples()
        self.assertEqual(player.getTrainingExamples(), [])

    def testGetTrainingExamplesReturnsACopy(self):
        # a caller must not be able to disturb the store by holding on to what it was given
        game = self.learningGame(1)
        self.playOneDeal(game)
        player = game.players[0]
        taken = player.getTrainingExamples()
        taken.append("not a real example")
        self.assertEqual(len(player.getTrainingExamples()), self.config.cardsPerHand)

    def testRecordedFeaturesMatchTheMovePlayed(self):
        # check the recorded vector against a fresh calculation on the same view
        game = self.learningGame(1)
        game.dealHand()
        seat = game.whoseTurn
        player = game.players[seat]
        view = game.viewFor(seat)
        expected = player.featuresFor(view, player.getMove(view)) # getMove records as it chooses
        self.assertEqual(player.dealFeatures[-1], expected)

    def testWeightsAreUntouchedByRecording(self):
        game = self.learningGame(1)
        self.playOneDeal(game)
        for player in game.players:
            for weight in player.weights:
                self.assertEqual(weight, 0.0)


# ----- THE LEARNING UPDATE -----

class TestLearningUpdate(LearningPlayerTestBase):

    def testLearningRateDefault(self):
        # a default guard: the rate is a setting, and changing it would change every trained result
        self.assertEqual(self.player.learningRate, 0.01)

    def testOneUpdateMovesTheWeightsInTheRightDirection(self):
        # a positive error must raise the bias and every weight with a positive feature
        example = self.makeExample({0: 1.0, 1: 0.5, 2: 0.25}, 7.0)
        self.player.updateFromExample(example["features"], example["dealReturn"])
        self.assertGreater(self.player.weights[0], 0.0)
        self.assertGreater(self.player.weights[1], 0.0)
        self.assertGreater(self.player.weights[2], 0.0)
        self.assertGreater(self.player.weights[biasIndex], 0.0)

    def testWeightsWithAZeroFeatureDoNotMove(self):
        # a feature that was zero took no part in the prediction, so it earns no credit or blame
        example = self.makeExample({0: 1.0}, 7.0)
        self.player.updateFromExample(example["features"], example["dealReturn"])
        for index in range(1, numberOfFeatures):
            self.assertEqual(self.player.weights[index], 0.0)

    def testNegativeErrorMovesTheWeightsDown(self):
        self.player.weights[biasIndex] = 10.0 # so the prediction starts well above the return
        example = self.makeExample({0: 1.0}, 2.0)
        self.player.updateFromExample(example["features"], example["dealReturn"])
        self.assertLess(self.player.weights[0], 0.0)
        self.assertLess(self.player.weights[biasIndex], 10.0)

    def testOneUpdateMatchesTheHandWorkedArithmetic(self):
        # error 4.0 at rate 0.1 gives weight0 0.4, weight1 0.2, bias 0.4
        self.player.learningRate = 0.1
        example = self.makeExample({0: 1.0, 1: 0.5}, 4.0)
        self.player.updateFromExample(example["features"], example["dealReturn"])
        self.assertAlmostEqual(self.player.weights[0], 0.4)
        self.assertAlmostEqual(self.player.weights[1], 0.2)
        self.assertAlmostEqual(self.player.weights[biasIndex], 0.4)

    def testASecondUpdateMatchesTheHandWorkedArithmetic(self):
        # prediction 0.9 and error 3.1 give weight0 0.71, weight1 0.355, bias 0.71
        self.player.learningRate = 0.1
        example = self.makeExample({0: 1.0, 1: 0.5}, 4.0)
        self.player.updateFromExample(example["features"], example["dealReturn"])
        self.player.updateFromExample(example["features"], example["dealReturn"])
        self.assertAlmostEqual(self.player.weights[0], 0.71)
        self.assertAlmostEqual(self.player.weights[1], 0.355)
        self.assertAlmostEqual(self.player.weights[biasIndex], 0.71)

    def testEveryWeightStepsFromTheSamePrediction(self):
        # one prediction before any weight moves, re-predicting would lower weight0
        self.player.learningRate = 0.1
        example = self.makeExample({0: 1.0}, 4.0)
        self.player.updateFromExample(example["features"], example["dealReturn"])
        self.assertAlmostEqual(self.player.weights[0], 0.4) # not 0.36, which a re-predicted step would give
        self.assertAlmostEqual(self.player.weights[biasIndex], 0.4)

    def testAnExactPredictionLeavesTheWeightsUnchanged(self):
        self.player.weights[0] = 2.0
        self.player.weights[biasIndex] = 1.0
        before = list(self.player.weights)
        example = self.makeExample({0: 1.0}, 3.0) # prediction is 1.0 + 2.0 * 1.0 = 3.0, the return exactly
        self.player.updateFromExample(example["features"], example["dealReturn"])
        self.assertEqual(self.player.weights, before)

    def testZeroLearningRateLeavesTheWeightsUnchanged(self):
        self.player.learningRate = 0.0
        before = list(self.player.weights)
        example = self.makeExample({0: 1.0, 1: 0.5}, 9.0)
        self.player.updateFromExample(example["features"], example["dealReturn"])
        self.assertEqual(self.player.weights, before)

    def testRepeatedUpdatesReduceTheSquaredError(self):
        # the core evidence that the update learns, the squared error falls
        examples = [
            self.makeExample({0: 1.0, 1: 0.0}, 8.0),
            self.makeExample({0: 0.0, 1: 1.0}, 4.0),
            self.makeExample({0: 0.5, 1: 0.5}, 6.0),
        ]
        self.player.learningRate = 0.05
        errorAtStart = self.totalSquaredError(self.player, examples)
        for round in range(200):
            self.player.updateFromExamples(examples)
        errorAtEnd = self.totalSquaredError(self.player, examples)
        self.assertLess(errorAtEnd, errorAtStart)
        self.assertLess(errorAtEnd, 0.01)

    def testTheErrorFallsSteadilyRatherThanOnlyAtTheEnd(self):
        # a single before-and-after check could hide wild swings, so watch it come down in stages
        examples = [
            self.makeExample({0: 1.0}, 8.0),
            self.makeExample({1: 1.0}, 4.0),
        ]
        self.player.learningRate = 0.05
        readings = []
        for block in range(5):
            for round in range(20):
                self.player.updateFromExamples(examples)
            readings.append(self.totalSquaredError(self.player, examples))
        for index in range(1, len(readings)):
            self.assertLess(readings[index], readings[index - 1])

    def testBatchUpdateMatchesApplyingEachInTurn(self):
        # the batch method must be exactly the single step repeated, in the order given
        examples = [
            self.makeExample({0: 1.0}, 8.0),
            self.makeExample({1: 0.5}, 3.0),
            self.makeExample({2: 0.25}, 5.0),
        ]
        batchPlayer = LearningPlayer(random.Random(1), self.config)
        oneByOnePlayer = LearningPlayer(random.Random(1), self.config)
        batchPlayer.updateFromExamples(examples)
        for example in examples:
            oneByOnePlayer.updateFromExample(example["features"], example["dealReturn"])
        self.assertEqual(batchPlayer.weights, oneByOnePlayer.weights)

    def testBatchUpdateOnAnEmptyListDoesNothing(self):
        before = list(self.player.weights)
        self.player.updateFromExamples([])
        self.assertEqual(self.player.weights, before)

    def testBatchUpdateAcceptsTheExamplesRecordedInADeal(self):
        # the store and the update have to fit together, so drive one into the other
        game = self.learningGame(1)
        self.playOneDeal(game)
        player = game.players[0]
        examples = player.getTrainingExamples()
        self.assertEqual(len(examples), self.config.cardsPerHand)
        player.updateFromExamples(examples) # the shapes match, so this must not raise
        moved = False
        for weight in player.weights:
            if weight != 0.0:
                moved = True
        self.assertTrue(moved)


# ----- THE AVERAGED UPDATE (STABILISATION EXPERIMENT) -----

class TestAveragedUpdate(LearningPlayerTestBase):

    def testUpdateModeDefaultsToOnline(self):
        # a default guard, online produced every measured number so far
        self.assertEqual(self.player.updateMode, "online")

    def testAveragedUpdateMatchesTheHandWorkedArithmetic(self):
        # both errors use zero weights, giving steps of 0.2, 0.1 and a bias of 0.3
        self.player.learningRate = 0.1
        examples = [
            self.makeExample({0: 1.0}, 4.0),
            self.makeExample({1: 1.0}, 2.0),
        ]
        self.player.updateFromExamplesAveraged(examples)
        self.assertAlmostEqual(self.player.weights[0], 0.2)
        self.assertAlmostEqual(self.player.weights[1], 0.1)
        self.assertAlmostEqual(self.player.weights[biasIndex], 0.3)

    def testEveryExampleIsScoredAgainstTheSameStartingWeights(self):
        # scoring after a partial step would shrink the second error and the bias
        self.player.learningRate = 0.1
        examples = [
            self.makeExample({0: 1.0}, 4.0),
            self.makeExample({0: 1.0}, 4.0), # the same example twice, so a moved prediction would show
        ]
        self.player.updateFromExamplesAveraged(examples)
        self.assertAlmostEqual(self.player.weights[biasIndex], 0.4) # not less, which partway stepping would give
        self.assertAlmostEqual(self.player.weights[0], 0.4)

    def testAveragedDiffersFromOnlineWhenTheExamplesDisagree(self):
        # if the two rules gave the same answer the whole experiment would be doing nothing
        examples = [
            self.makeExample({0: 1.0}, 8.0),
            self.makeExample({0: 1.0}, 2.0), # the same features, a very different return
            self.makeExample({1: 1.0}, 5.0),
        ]
        onlinePlayer = LearningPlayer(random.Random(1), self.config)
        averagedPlayer = LearningPlayer(random.Random(1), self.config)
        onlinePlayer.updateFromExamples(examples)
        averagedPlayer.updateFromExamplesAveraged(examples)
        self.assertNotEqual(onlinePlayer.weights, averagedPlayer.weights)

    def testASingleExampleGivesTheSameResultEitherWay(self):
        # with one example there is nothing to average, so the two rules must agree exactly
        examples = [self.makeExample({0: 1.0, 1: 0.5}, 6.0)]
        onlinePlayer = LearningPlayer(random.Random(1), self.config)
        averagedPlayer = LearningPlayer(random.Random(1), self.config)
        onlinePlayer.updateFromExamples(examples)
        averagedPlayer.updateFromExamplesAveraged(examples)
        self.assertEqual(onlinePlayer.weights, averagedPlayer.weights)

    def testAnEmptyBatchDoesNothing(self):
        before = list(self.player.weights)
        self.player.updateFromExamplesAveraged([])
        self.assertEqual(self.player.weights, before)

    def testTheAveragedStepIsSmallerThanTheOnlineOneOnCorrelatedExamples(self):
        # thirteen shared-return examples take thirteen online steps but one averaged step
        examples = []
        for count in range(13):
            examples.append(self.makeExample({0: 1.0}, 8.0)) # what one deal's worth looks like
        onlinePlayer = LearningPlayer(random.Random(1), self.config)
        averagedPlayer = LearningPlayer(random.Random(1), self.config)
        onlinePlayer.updateFromExamples(examples)
        averagedPlayer.updateFromExamplesAveraged(examples)
        self.assertGreater(onlinePlayer.weights[biasIndex], averagedPlayer.weights[biasIndex])

    def testTheOnlineMethodsAreUntouched(self):
        # the online path must still match the snapshotted results
        self.player.learningRate = 0.1
        example = self.makeExample({0: 1.0, 1: 0.5}, 4.0)
        self.player.updateFromExample(example["features"], example["dealReturn"])
        self.assertAlmostEqual(self.player.weights[0], 0.4)
        self.assertAlmostEqual(self.player.weights[1], 0.2)
        self.assertAlmostEqual(self.player.weights[biasIndex], 0.4)


# ----- SELECTING THE UPDATE MODE -----

class TestApplyUpdateRouting(LearningPlayerTestBase):

    def someDisagreeingExamples(self):
        return [
            self.makeExample({0: 1.0}, 8.0),
            self.makeExample({0: 1.0}, 2.0),
            self.makeExample({1: 1.0}, 5.0),
        ]

    def testOnlineModeRoutesToTheOnlineUpdate(self):
        examples = self.someDisagreeingExamples()
        routed = LearningPlayer(random.Random(1), self.config)
        direct = LearningPlayer(random.Random(1), self.config)
        routed.updateMode = "online"
        routed.applyUpdate(examples)
        direct.updateFromExamples(examples)
        self.assertEqual(routed.weights, direct.weights)

    def testAveragedModeRoutesToTheAveragedUpdate(self):
        examples = self.someDisagreeingExamples()
        routed = LearningPlayer(random.Random(1), self.config)
        direct = LearningPlayer(random.Random(1), self.config)
        routed.updateMode = "averaged"
        routed.applyUpdate(examples)
        direct.updateFromExamplesAveraged(examples)
        self.assertEqual(routed.weights, direct.weights)

    def testAnUnknownModeFailsLoudly(self):
        # a typo in an experiment command must not train by the wrong rule
        self.player.updateMode = "nonsense"
        self.assertRaises(ValueError, self.player.applyUpdate, self.someDisagreeingExamples())


# ----- THE SELF DESCRIBING WEIGHTS HEADER -----

class TestWeightsFileHeader(LearningPlayerTestBase):

    def someTrainedLookingWeights(self):
        weights = []
        for slot in range(numberOfWeights):
            weights.append((slot + 1) * 0.37 - 2.0)
        weights[3] = -0.1
        weights[7] = 1 / 3
        return weights

    def someDetails(self):
        details = {}
        details["updateMode"] = "averaged"
        details["trainingDeals"] = 2000
        details["deckSeed"] = 1
        details["playerSeed"] = 1
        return details

    def testAHeaderedFileLoadsBackToIdenticalWeights(self):
        # the whole point of the header is that it costs nothing: the file must still reload exactly
        saved = LearningPlayer(random.Random(1), self.config)
        saved.weights = self.someTrainedLookingWeights()
        with tempfile.TemporaryDirectory() as folder:
            filePath = os.path.join(folder, "learningWeights.txt")
            saved.saveWeights(filePath, self.someDetails())
            loaded = LearningPlayer(random.Random(2), self.config)
            loaded.loadWeights(filePath)
            self.assertEqual(loaded.weights, saved.weights)

    def testTheHeaderLinesAreWritten(self):
        with tempfile.TemporaryDirectory() as folder:
            filePath = os.path.join(folder, "learningWeights.txt")
            self.player.saveWeights(filePath, self.someDetails())
            handle = open(filePath, "r")
            text = handle.read()
            handle.close()
            self.assertIn("# whist learning agent weights", text)
            self.assertIn("# updateMode: averaged", text) # a bare vector is otherwise unidentifiable
            self.assertIn("# trainingDeals: 2000", text)
            self.assertIn("# deckSeed: 1", text)
            self.assertIn("# playerSeed: 1", text)

    def testCommentLinesAreSkippedAndOnlyNumbersAreRead(self):
        # the count check must see the numbers only, or a headered file would look the wrong size
        with tempfile.TemporaryDirectory() as folder:
            filePath = os.path.join(folder, "learningWeights.txt")
            self.player.saveWeights(filePath, self.someDetails())

            handle = open(filePath, "r")
            lines = handle.readlines()
            handle.close()
            numberLines = 0
            for line in lines:
                if line.strip() and not line.strip().startswith("#"):
                    numberLines = numberLines + 1
            self.assertEqual(numberLines, numberOfWeights)
            self.assertGreater(len(lines), numberOfWeights) # so there really was a header to skip

            loaded = LearningPlayer(random.Random(2), self.config)
            loaded.loadWeights(filePath) # must not raise the wrong-size error
            self.assertEqual(len(loaded.weights), numberOfWeights)

    def testAFileWithNoDetailsHasNoHeaderAtAll(self):
        # the old shape is kept when no details are given, so nothing that wrote weights before changes
        with tempfile.TemporaryDirectory() as folder:
            filePath = os.path.join(folder, "learningWeights.txt")
            self.player.saveWeights(filePath)
            handle = open(filePath, "r")
            lines = handle.readlines()
            handle.close()
            self.assertEqual(len(lines), numberOfWeights)

    def testAHeaderedFileStillFailsLoudlyWhenItIsTheWrongSize(self):
        # the size check must count numbers, not lines, so a header cannot mask a short file
        with tempfile.TemporaryDirectory() as folder:
            filePath = os.path.join(folder, "shortWithHeader.txt")
            handle = open(filePath, "w")
            handle.write("# whist learning agent weights\n")
            handle.write("# updateMode: averaged\n")
            handle.write("0.5\n1.5\n")
            handle.close()
            player = LearningPlayer(random.Random(1), self.config)
            self.assertRaises(ValueError, player.loadWeights, filePath)
