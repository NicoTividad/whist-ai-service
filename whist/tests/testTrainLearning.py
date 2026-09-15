# these drive short training runs, so they stay well inside a normal test run. nothing here writes
# to results/, and the only file written is a weights file inside a temporary directory.

import os
import random
import tempfile
import unittest

from whist.engine.rules import GameConfig
from whist.engine.game import makeSeatRngs
from whist.main import makePlayers, evaluateMatchup, makeSearchSettings, defaultLearningWeightsFile
from whist.engine.cards import Card
from whist.engine.playerView import PlayerView
from whist.players.learningPlayer import LearningPlayer, numberOfWeights
from whist.trainLearning import (
    makeLearningSettings,
    makeTrainingSeats,
    playOneTrainingDeal,
    decayedExploreRate,
    trainBySelfPlay,
    evaluateWeights,
    untrainedWeights,
    weightsBaseNameForMode,
    saveTrainedWeights,
    makeCurveRow,
    writeCurveFile,
)


# ----- SHARED HELPERS -----

class TrainLearningTestBase(unittest.TestCase):

    def setUp(self):
        self.config = GameConfig()

    def shortSettings(self, deals):
        # a short run, small enough for the test suite but long enough to show a real gain
        return makeLearningSettings(trainingDeals=deals)

    def leadingView(self):
        # a rank weight makes the ace of hearts the unique best move
        hand = [Card("hearts", 14), Card("hearts", 2), Card("spades", 5), Card("clubs", 9)]
        return PlayerView(
            seat=0,
            ownHand=hand,
            trumpSuit="spades",
            currentTrick=[],
            cardsPlayedSoFar=[],
            scores=[0, 0],
            whoseTurn=0,
            legalMoves=list(hand),
        )

    def rankRewardingPlayer(self, seed):
        player = LearningPlayer(random.Random(seed), self.config)
        player.weights[0] = 1.0 # feature 1 is the scaled rank
        return player


# ----- THE SETTINGS DEFAULTS -----

class TestLearningSettings(TrainLearningTestBase):

    def testDefaultsAreUnchanged(self):
        # a default guard, a silent change would make saved weights unreproducible
        settings = makeLearningSettings()
        self.assertEqual(settings["learningRate"], 0.01)
        self.assertEqual(settings["exploreStartRate"], 1.0)
        self.assertEqual(settings["exploreDecay"], 0.999)
        self.assertEqual(settings["exploreMinRate"], 0.05)
        self.assertEqual(settings["trainingDeals"], 2000)
        self.assertEqual(settings["checkpointEvery"], 250)
        self.assertEqual(settings["evaluationGames"], 200)
        self.assertEqual(settings["evaluationSeed"], 12345)
        self.assertEqual(settings["updateMode"], "online")

    def testSettingsCanBeOverridden(self):
        settings = makeLearningSettings(learningRate=0.5, trainingDeals=10)
        self.assertEqual(settings["learningRate"], 0.5)
        self.assertEqual(settings["trainingDeals"], 10)
        self.assertEqual(settings["exploreStartRate"], 1.0)


# ----- THE SHARED WEIGHTS VECTOR -----

class TestTrainingSeats(TrainLearningTestBase):

    def testAllFourSeatsShareOneWeightsVector(self):
        # checked by identity, since four equal copies would not share learning
        players, sharedWeights = makeTrainingSeats(self.config, 1, self.shortSettings(10))
        for player in players:
            self.assertIs(player.weights, sharedWeights)

    def testAnUpdateBySeatZeroIsSeenByEverySeat(self):
        # the update assigns into the list rather than rebinding it, which is what makes sharing work
        players, sharedWeights = makeTrainingSeats(self.config, 1, self.shortSettings(10))
        features = [0.0 for slot in range(13)]
        features[0] = 1.0
        players[0].updateFromExample(features, 8.0)
        for player in players:
            self.assertEqual(player.weights[0], sharedWeights[0])
            self.assertNotEqual(player.weights[0], 0.0)

    def testEverySeatTakesTheSettingsLearningRate(self):
        players, sharedWeights = makeTrainingSeats(self.config, 1, makeLearningSettings(learningRate=0.25))
        for player in players:
            self.assertEqual(player.learningRate, 0.25)


# ----- THE EXPLORATION SCHEDULE -----

class TestExplorationSchedule(TrainLearningTestBase):

    def testTheRateDecays(self):
        settings = makeLearningSettings()
        self.assertAlmostEqual(decayedExploreRate(1.0, settings), 0.999)
        self.assertAlmostEqual(decayedExploreRate(0.5, settings), 0.4995)

    def testTheRateNeverFallsBelowTheFloor(self):
        settings = makeLearningSettings()
        rate = 1.0
        for step in range(20000): # far more decay than any real run applies
            rate = decayedExploreRate(rate, settings)
        self.assertEqual(rate, settings["exploreMinRate"]) # parked on the floor, not at zero


# ----- EXPLORATION IS TRAINING ONLY -----

class TestExplorationIsTrainingOnly(TrainLearningTestBase):

    def testExploreRateDefaultsToZero(self):
        # a default guard: an agent that has not been put into training explores not at all
        self.assertEqual(LearningPlayer(random.Random(1), self.config).exploreRate, 0.0)

    def testAnEvaluatedAgentIsFullyGreedy(self):
        view = self.leadingView()
        for seed in range(50):
            player = self.rankRewardingPlayer(seed)
            self.assertEqual(player.getMove(view), Card("hearts", 14))

    def testExplorationSometimesPlaysANonGreedyMove(self):
        view = self.leadingView()
        nonGreedy = 0
        for seed in range(50):
            player = self.rankRewardingPlayer(seed)
            player.exploreRate = 1.0 # always explore, so the effect is unmistakable
            if player.getMove(view) != Card("hearts", 14):
                nonGreedy = nonGreedy + 1
        self.assertGreater(nonGreedy, 0)

    def testAnExploredMoveIsStillRecorded(self):
        # exploration changes which move is played, never whether it is learned from
        view = self.leadingView()
        player = self.rankRewardingPlayer(1)
        player.exploreRate = 1.0
        move = player.getMove(view)
        self.assertEqual(len(player.dealFeatures), 1)
        self.assertEqual(player.dealFeatures[0], player.featuresFor(view, move))

    def testEvaluateWeightsLeavesEverySeatGreedy(self):
        # any exploration would draw from the rngs and shift the result
        settings = self.shortSettings(50)
        trained = trainBySelfPlay(settings, 1, 1)
        first = evaluateWeights(trained, 20)
        second = evaluateWeights(trained, 20)
        self.assertEqual(first, second)


# ----- A SHORT TRAINING RUN -----

class TestTrainingRun(TrainLearningTestBase):

    def testAShortRunCompletesAndMovesTheWeightsOffZero(self):
        trained = trainBySelfPlay(self.shortSettings(100), 1, 1)
        self.assertEqual(len(trained), numberOfWeights)
        moved = 0
        for weight in trained:
            if weight != 0.0:
                moved = moved + 1
        self.assertGreater(moved, 0)

    def testTrainingIsReproducibleFromTheSeeds(self):
        settings = self.shortSettings(100)
        first = trainBySelfPlay(settings, 1, 1)
        second = trainBySelfPlay(settings, 1, 1)
        self.assertEqual(first, second)

    def testADifferentPlayerSeedTrainsDifferentWeights(self):
        # without this a run that always gave the same numbers would still pass
        settings = self.shortSettings(100)
        first = trainBySelfPlay(settings, 1, 1)
        other = trainBySelfPlay(settings, 1, 99)
        self.assertNotEqual(first, other)

    def testADifferentDeckSeedTrainsDifferentWeights(self):
        settings = self.shortSettings(100)
        first = trainBySelfPlay(settings, 1, 1)
        other = trainBySelfPlay(settings, 500, 1)
        self.assertNotEqual(first, other)

    def testTrainingDealsSettingIsRespected(self):
        short = trainBySelfPlay(self.shortSettings(20), 1, 1)
        longer = trainBySelfPlay(self.shortSettings(200), 1, 1)
        self.assertNotEqual(short, longer)


# ----- THE TRAINING ACTUALLY HELPS -----

class TestTrainingImproves(TrainLearningTestBase):

    def testUntrainedAgentIsAboutEvenWithRandom(self):
        # zero weights play at random, so the untrained agent lands near half
        wins = evaluateWeights(untrainedWeights(), 100)
        self.assertGreater(wins, 30)
        self.assertLess(wins, 70)

    def testTrainedAgentBeatsRandomMoreThanUntrainedDoes(self):
        # the headline check: training produced a real improvement, not just movement
        untrainedWins = evaluateWeights(untrainedWeights(), 100)
        trained = trainBySelfPlay(self.shortSettings(400), 1, 1)
        trainedWins = evaluateWeights(trained, 100)
        self.assertGreater(trainedWins, untrainedWins + 5) # a clear gain, not noise
        self.assertGreater(trainedWins, 55)


# ----- SAVING THE TRAINED MODEL -----

class TestSavingTrainedWeights(TrainLearningTestBase):

    def testTrainedWeightsSaveAndReloadIdentically(self):
        trained = trainBySelfPlay(self.shortSettings(50), 1, 1)
        saver = LearningPlayer(random.Random(1), self.config)
        saver.weights = trained
        with tempfile.TemporaryDirectory() as folder: # never results/, and nothing left behind
            filePath = os.path.join(folder, "learningWeights.txt")
            saver.saveWeights(filePath)
            reloaded = LearningPlayer(random.Random(2), self.config)
            reloaded.loadWeights(filePath)
            self.assertEqual(reloaded.weights, trained)

    def testAReloadedAgentPlaysTheSameAsTheTrainedOne(self):
        trained = trainBySelfPlay(self.shortSettings(50), 1, 1)
        view = self.leadingView()
        with tempfile.TemporaryDirectory() as folder:
            filePath = os.path.join(folder, "learningWeights.txt")
            saver = LearningPlayer(random.Random(1), self.config)
            saver.weights = trained
            saver.saveWeights(filePath)
            reloaded = LearningPlayer(random.Random(1), self.config)
            reloaded.loadWeights(filePath)
            for move in view.legalMoves:
                self.assertEqual(reloaded.scoreMove(view, move), saver.scoreMove(view, move))


# ----- THE TRAINING CURVE -----

class TestTrainingCurve(TrainLearningTestBase):

    def curveSettings(self, deals=80, every=40, games=20):
        # small on purpose: these check the shape and the plumbing, not the strength of the agent
        return makeLearningSettings(trainingDeals=deals, checkpointEvery=every, evaluationGames=games)

    def runCurve(self, settings, deckSeed=1, playerSeed=1):
        curve = []
        trainBySelfPlay(settings, deckSeed, playerSeed, 0, curve)
        return curve

    def testOnePointPerCheckpointPlusTheUntrainedStart(self):
        # 80 deals checkpointed every 40 gives points at 0, 40 and 80
        curve = self.runCurve(self.curveSettings(80, 40, 20))
        self.assertEqual(len(curve), 3)
        self.assertEqual(curve[0]["dealsTrained"], 0)
        self.assertEqual(curve[1]["dealsTrained"], 40)
        self.assertEqual(curve[2]["dealsTrained"], 80)

    def testNoCurveIsRecordedWhenNoneIsAskedFor(self):
        # leaving curve as None must skip the evaluations entirely, which is what keeps training cheap
        weights = trainBySelfPlay(self.curveSettings(40, 20, 20), 1, 1)
        self.assertEqual(len(weights), numberOfWeights)

    def testEveryValueIsInRange(self):
        curve = self.runCurve(self.curveSettings(40, 20, 20))
        for row in curve:
            self.assertGreaterEqual(row["ruleWins"], 0)
            self.assertLessEqual(row["ruleWins"], row["games"])
            self.assertGreaterEqual(row["ruleWinRate"], 0.0)
            self.assertLessEqual(row["ruleWinRate"], 100.0)
            self.assertGreaterEqual(row["randomWins"], 0)
            self.assertLessEqual(row["randomWins"], row["games"])

    def testTheCurveIsReproducible(self):
        # the same seeds must give the same curve, or no point on it can be quoted
        settings = self.curveSettings(40, 20, 20)
        first = self.runCurve(settings)
        second = self.runCurve(settings)
        self.assertEqual(first, second)

    def testTheUntrainedPointIsNearTheFloorAgainstRule(self):
        # the untrained agent must win almost nothing against the rule agent
        curve = self.runCurve(self.curveSettings(40, 20, 100))
        self.assertLess(curve[0]["ruleWinRate"], 20.0)

    def testTheCurveImprovesOnItsStartingPointAgainstRule(self):
            # checks only that some checkpoint beats the untrained start
            # a steady rise would be untrue, the curve oscillates against the rule agent
        curve = self.runCurve(self.curveSettings(400, 200, 100))
        bestRule = curve[0]["ruleWinRate"]
        for row in curve:
            if row["ruleWinRate"] > bestRule:
                bestRule = row["ruleWinRate"]
        self.assertGreater(bestRule, curve[0]["ruleWinRate"])

    def testTheCurveImprovesOnItsStartingPointAgainstRandom(self):
        # the same check against the weak baseline, where the improvement is much clearer
        curve = self.runCurve(self.curveSettings(400, 200, 100))
        bestRandom = curve[0]["randomWinRate"]
        for row in curve:
            if row["randomWinRate"] > bestRandom:
                bestRandom = row["randomWinRate"]
        self.assertGreater(bestRandom, curve[0]["randomWinRate"])


# ----- CHECKPOINTING MUST NOT DISTURB TRAINING -----

class TestCheckpointingIsSideEffectFree(TrainLearningTestBase):

    def testCheckpointingDoesNotChangeTheTrainedWeights(self):
        # training with and without checkpoints must give identical weights
        settings = makeLearningSettings(trainingDeals=100, checkpointEvery=25, evaluationGames=20)
        withCurve = []
        measured = trainBySelfPlay(settings, 1, 1, 0, withCurve)
        unmeasured = trainBySelfPlay(settings, 1, 1, 0, None)
        self.assertEqual(len(withCurve), 5) # it really did measure five times
        self.assertEqual(measured, unmeasured)

    def testAFrozenCheckpointDoesNotMoveTheWeightsItMeasured(self):
        # makeCurveRow takes a copy, so the vector handed in must come back untouched
        weights = untrainedWeights()
        weights[0] = 0.5
        before = list(weights)
        makeCurveRow(0, weights, makeLearningSettings(evaluationGames=20))
        self.assertEqual(weights, before)

    def testACheckpointRepeatsExactly(self):
        # the evaluation is greedy and seeded, so measuring the same weights twice must agree
        settings = makeLearningSettings(evaluationGames=40)
        weights = trainBySelfPlay(makeLearningSettings(trainingDeals=50), 1, 1)
        first = makeCurveRow(100, weights, settings)
        second = makeCurveRow(100, weights, settings)
        self.assertEqual(first, second)


# ----- WRITING THE CURVE FILE -----

class TestCurveFile(TrainLearningTestBase):

    def smallCurve(self):
        return [
            {"dealsTrained": 0, "games": 20, "ruleWins": 1, "ruleWinRate": 5.0,
             "randomWins": 10, "randomWinRate": 50.0},
            {"dealsTrained": 40, "games": 20, "ruleWins": 3, "ruleWinRate": 15.0,
             "randomWins": 13, "randomWinRate": 65.0},
        ]

    def testWritesTheRowsAndASelfDescribingHeader(self):
        settings = makeLearningSettings()
        with tempfile.TemporaryDirectory() as folder:
            path = writeCurveFile(self.smallCurve(), settings, 1, 1, folder, "learningCurve")
            handle = open(path, "r")
            text = handle.read()
            handle.close()

            # the header must make it impossible to mistake this for a search agent output file
            self.assertIn("LEARNING run, not a search run", text)
            self.assertIn("updateMode: online", text) # the mode must be on the face of the file
            self.assertIn("learningRate: 0.01", text)
            self.assertIn("exploreStartRate: 1.0", text)
            self.assertIn("exploreDecay: 0.999", text)
            self.assertIn("exploreMinRate: 0.05", text)
            self.assertIn("evaluationSeed: 12345", text)
            self.assertIn("training deckSeed: 1", text)
            self.assertIn("training playerSeed: 1", text)
            self.assertIn("dealsTrained,games,ruleWins,ruleWinRate,randomWins,randomWinRate", text)
            self.assertIn("0,20,1,5.0,10,50.0", text)
            self.assertIn("40,20,3,15.0,13,65.0", text)

    def testASecondWriteDoesNotOverwriteTheFirst(self):
        # results/ holds evidence that cost hours to produce, so a repeated run must never clobber it
        settings = makeLearningSettings()
        with tempfile.TemporaryDirectory() as folder:
            first = writeCurveFile(self.smallCurve(), settings, 1, 1, folder, "learningCurve")
            second = writeCurveFile(self.smallCurve(), settings, 1, 2, folder, "learningCurve")
            self.assertNotEqual(first, second)
            self.assertTrue(first.endswith("learningCurve.csv"))
            self.assertTrue(second.endswith("learningCurve2.csv"))
            self.assertTrue(os.path.exists(first))

    def testAThirdWriteStepsPastBothOfThem(self):
        settings = makeLearningSettings()
        with tempfile.TemporaryDirectory() as folder:
            writeCurveFile(self.smallCurve(), settings, 1, 1, folder, "learningCurve")
            writeCurveFile(self.smallCurve(), settings, 1, 2, folder, "learningCurve")
            third = writeCurveFile(self.smallCurve(), settings, 1, 3, folder, "learningCurve")
            self.assertTrue(third.endswith("learningCurve3.csv"))

    def testDoesNotTouchAnythingAlreadyInTheResultsFolder(self):
        # a stand-in results file must come back byte for byte identical
        settings = makeLearningSettings()
        with tempfile.TemporaryDirectory() as folder:
            sentinelPath = os.path.join(folder, "ablationSummary.csv")
            sentinelText = "condition,samples,games\nbaseline,5,500\n"
            handle = open(sentinelPath, "w")
            handle.write(sentinelText)
            handle.close()

            before = sorted(os.listdir(folder))
            writeCurveFile(self.smallCurve(), settings, 1, 1, folder, "learningCurve")
            after = sorted(os.listdir(folder))

            handle = open(sentinelPath, "r")
            self.assertEqual(handle.read(), sentinelText)
            handle.close()
            self.assertEqual(after, sorted(before + ["learningCurve.csv"])) # exactly one new file

    def testCreatesTheResultsFolderIfItIsMissing(self):
        settings = makeLearningSettings()
        with tempfile.TemporaryDirectory() as folder:
            missing = os.path.join(folder, "notYetThere")
            path = writeCurveFile(self.smallCurve(), settings, 1, 1, missing, "learningCurve")
            self.assertTrue(os.path.exists(path))


# ----- NAMING AND SAVING THE TRAINED MODEL -----

class TestWeightsFileNaming(TrainLearningTestBase):

    def someWeights(self):
        # a recognisable vector; nothing here trains, this is about naming and headers only
        weights = []
        for slot in range(numberOfWeights):
            weights.append((slot + 1) * 0.25 - 1.0)
        return weights

    def testTheNameCarriesTheUpdateMode(self):
        # a fixed name let runs overwrite each other, so the mode is in the name
        self.assertEqual(weightsBaseNameForMode("online"), "learningWeightsOnline")
        self.assertEqual(weightsBaseNameForMode("averaged"), "learningWeightsAveraged")

    def testAnOnlineAndAnAveragedRunCannotCollide(self):
        onlineSettings = makeLearningSettings(updateMode="online")
        averagedSettings = makeLearningSettings(updateMode="averaged")
        with tempfile.TemporaryDirectory() as folder:
            onlinePath = saveTrainedWeights(self.someWeights(), onlineSettings, 1, 1, folder)
            averagedPath = saveTrainedWeights(self.someWeights(), averagedSettings, 1, 1, folder)
            self.assertNotEqual(onlinePath, averagedPath)
            self.assertTrue(onlinePath.endswith("learningWeightsOnline.txt"))
            self.assertTrue(averagedPath.endswith("learningWeightsAveraged.txt"))

    def testARepeatRunStepsPastRatherThanOverwriting(self):
        # mirrors the curve file naming test: a second run of the same command must not clobber
        settings = makeLearningSettings(updateMode="averaged")
        with tempfile.TemporaryDirectory() as folder:
            first = saveTrainedWeights(self.someWeights(), settings, 1, 1, folder)
            second = saveTrainedWeights(self.someWeights(), settings, 1, 2, folder)
            third = saveTrainedWeights(self.someWeights(), settings, 1, 3, folder)
            self.assertTrue(first.endswith("learningWeightsAveraged.txt"))
            self.assertTrue(second.endswith("learningWeightsAveraged2.txt"))
            self.assertTrue(third.endswith("learningWeightsAveraged3.txt"))
            self.assertTrue(os.path.exists(first))
            self.assertTrue(os.path.exists(second))

    def testAnOverrideNameIsHonoured(self):
        settings = makeLearningSettings(updateMode="averaged")
        with tempfile.TemporaryDirectory() as folder:
            path = saveTrainedWeights(self.someWeights(), settings, 1, 1, folder, "myOwnModel")
            self.assertTrue(path.endswith("myOwnModel.txt"))

    def testTheSavedFileSaysHowItWasProduced(self):
        settings = makeLearningSettings(updateMode="averaged", trainingDeals=1500)
        with tempfile.TemporaryDirectory() as folder:
            path = saveTrainedWeights(self.someWeights(), settings, 7, 9, folder)
            handle = open(path, "r")
            text = handle.read()
            handle.close()
            self.assertIn("# updateMode: averaged", text)
            self.assertIn("# trainingDeals: 1500", text)
            self.assertIn("# deckSeed: 7", text)
            self.assertIn("# playerSeed: 9", text)

    def testASavedFileLoadsBackToTheWeightsItWasGiven(self):
        settings = makeLearningSettings(updateMode="averaged")
        weights = self.someWeights()
        with tempfile.TemporaryDirectory() as folder:
            path = saveTrainedWeights(weights, settings, 1, 1, folder)
            loaded = LearningPlayer(random.Random(1), self.config)
            loaded.loadWeights(path)
            self.assertEqual(loaded.weights, weights) # the header costs nothing on the way back in

    def testSavingDoesNotDisturbTheWeightsHandedIn(self):
        settings = makeLearningSettings(updateMode="averaged")
        weights = self.someWeights()
        before = list(weights)
        with tempfile.TemporaryDirectory() as folder:
            saveTrainedWeights(weights, settings, 1, 1, folder)
        self.assertEqual(weights, before)


# ----- SEATING THE TRAINED LEARNING AGENT -----

class TestLearningSeat(TrainLearningTestBase):

    def seatRngs(self):
        return makeSeatRngs(1, self.config.numberOfPlayers)

    def testTheSeatedAgentIsTrainedNotZeroWeighted(self):
        # a zero-weight agent plays at random and would make the comparison meaningless
        players = makePlayers(["learning", "rule", "learning", "rule"], self.seatRngs(), self.config)
        allZero = True
        for weight in players[0].weights:
            if weight != 0.0:
                allZero = False
        self.assertFalse(allZero)

    def testTheDefaultModelIsTheCanonicalAveragedFile(self):
        self.assertEqual(defaultLearningWeightsFile, "learningWeightsAveraged.txt")

    def testTheSeatedAgentMatchesTheFileOnDisk(self):
        # not just non-zero: it must be exactly the model that file holds
        players = makePlayers(["learning"], self.seatRngs(), self.config)
        reference = LearningPlayer(random.Random(1), self.config)
        reference.loadWeights(defaultLearningWeightsFile)
        self.assertEqual(players[0].weights, reference.weights)

    def testTheSeatedAgentIsFullyGreedyAndNotTraining(self):
        # confirmed rather than switched off, a fresh player already starts at zero
        players = makePlayers(["learning"], self.seatRngs(), self.config)
        self.assertEqual(players[0].exploreRate, 0.0)
        self.assertEqual(players[0].getTrainingExamples(), [])

    def testAMissingModelFailsLoudly(self):
        # silently seating an untrained agent would produce a misleading comparison that looks normal
        self.assertRaises(ValueError, makePlayers, ["learning"], self.seatRngs(), self.config,
                          None, "noSuchWeightsFile.txt")

    def testADifferentModelFileCanBeAsked(self):
        with tempfile.TemporaryDirectory() as folder:
            filePath = os.path.join(folder, "otherModel.txt")
            other = LearningPlayer(random.Random(1), self.config)
            other.weights = [0.5 for slot in range(numberOfWeights)]
            other.saveWeights(filePath)
            players = makePlayers(["learning"], self.seatRngs(), self.config, None, filePath)
            self.assertEqual(players[0].weights, other.weights)


# ----- THE LEARNING AGENT IN THE FINAL COMPARISON -----

class TestLearningInEvaluateMatchup(TrainLearningTestBase):

    def testLearningVersusRuleRunsThroughTheSharedMachinery(self):
        # the same call path as the other agents, so all three share one table
        result = evaluateMatchup(4, 1, 1, "learning", "rule")
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["aWins"] + result["bWins"], 4)

    def testLearningVersusSearchRunsThroughTheSharedMachinery(self):
        result = evaluateMatchup(2, 1, 1, "learning", "search", makeSearchSettings(sampleCount=2))
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["aWins"] + result["bWins"], 2)

    def testTheMatchupIsReproducible(self):
        # a quoted win rate is worthless if it does not reproduce from its seeds
        first = evaluateMatchup(4, 1, 1, "learning", "rule")
        second = evaluateMatchup(4, 1, 1, "learning", "rule")
        self.assertEqual(first["aWins"], second["aWins"])
