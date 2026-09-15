# only ever creates new files in results/, nothing already there is overwritten

import argparse
import os

from .engine.rules import GameConfig
from .engine.game import WhistGame, makeSeatRngs
from .players.learningPlayer import LearningPlayer
from .players.randomPlayer import RandomPlayer
from .players.ruleBasedPlayer import RuleBasedPlayer


# ----- LEARNING SETTINGS -----
def makeLearningSettings(learningRate=0.01, exploreStartRate=1.0, exploreDecay=0.999,
                         exploreMinRate=0.05, trainingDeals=2000, checkpointEvery=250,
                         evaluationGames=200, evaluationSeed=12345, updateMode="online"):
    # each setting has a default guard test so trained files stay reproducible
    settings = {}
    settings["learningRate"] = learningRate
    settings["exploreStartRate"] = exploreStartRate
    settings["exploreDecay"] = exploreDecay # the rate is multiplied by this after every deal
    settings["exploreMinRate"] = exploreMinRate # so some exploration always remains
    settings["trainingDeals"] = trainingDeals
    settings["checkpointEvery"] = checkpointEvery # how often the weights are frozen and evaluated
    settings["evaluationGames"] = evaluationGames # against each opponent
    # one fixed evaluation seed, so the curve shows learning not card luck
    settings["evaluationSeed"] = evaluationSeed
    # online is the baseline every measured number used, so it stays the default
    settings["updateMode"] = updateMode
    return settings


# ----- BUILDING THE TRAINING SEATS -----
def makeTrainingSeats(config: GameConfig, playerSeed: int, settings: dict) -> tuple:
    # all four seats share one weights list, never call loadWeights mid training
    seatRngs = makeSeatRngs(playerSeed, config.numberOfPlayers)
    players = []
    for seat in range(config.numberOfPlayers):
        players.append(LearningPlayer(seatRngs[seat], config))

    sharedWeights = players[0].weights
    for player in players:
        player.weights = sharedWeights
        player.learningRate = settings["learningRate"]
        player.updateMode = settings["updateMode"]
    return players, sharedWeights


# ----- PLAYING ONE TRAINING DEAL -----
def playOneTrainingDeal(game: WhistGame, players: list) -> None:
    # events are forwarded here because the engine does not push them to players
    game.dealHand()
    while not game.handOver():
        event = game.playOneMove()
        for player in players:
            player.observe(event)


# ----- APPLYING THE UPDATE -----
def applyUpdatesAndClear(players: list) -> None:
    # seat order then play order, so a run reproduces from the seeds alone
    for player in players:
        player.applyUpdate(player.getTrainingExamples())
        player.clearTrainingExamples()


# ----- THE EXPLORATION SCHEDULE -----
def decayedExploreRate(rate: float, settings: dict) -> float:
    # decay stops early lock-in, the floor keeps some exploration going
    lowered = rate * settings["exploreDecay"]
    if lowered < settings["exploreMinRate"]:
        return settings["exploreMinRate"]
    return lowered


# ----- ONE CHECKPOINT ON THE TRAINING CURVE -----
def makeCurveRow(dealsTrained: int, weights: list, settings: dict) -> dict:
    # a frozen copy, so measuring never touches the weights being trained
    frozen = list(weights)
    games = settings["evaluationGames"]
    evaluationSeed = settings["evaluationSeed"]

    ruleWins = evaluateWeights(frozen, games, evaluationSeed, evaluationSeed, "rule") # the yardstick
    randomWins = evaluateWeights(frozen, games, evaluationSeed, evaluationSeed, "random") # for the saturation story

    row = {}
    row["dealsTrained"] = dealsTrained
    row["games"] = games
    row["ruleWins"] = ruleWins
    row["ruleWinRate"] = winPercentage(ruleWins, games)
    row["randomWins"] = randomWins
    row["randomWinRate"] = winPercentage(randomWins, games)
    return row


def winPercentage(wins: int, games: int) -> float:
    # guarding an empty batch rather than dividing by zero
    if games == 0:
        return 0.0
    return wins / games * 100


# ----- THE TRAINING RUN -----
def trainBySelfPlay(settings: dict, deckSeed: int = 1, playerSeed: int = 1, progressEvery: int = 0,
                    curve: list | None = None) -> list:
    # pass a list as curve to turn checkpointing on, None keeps training cheap
    config = GameConfig()
    players, sharedWeights = makeTrainingSeats(config, playerSeed, settings)

    exploreRate = settings["exploreStartRate"]
    dealsPlayed = 0
    gameNumber = 0 # so each gets its own deal seed

    # the first point is the untrained agent, so the rise has a true start
    if curve is not None:
        curve.append(makeCurveRow(0, sharedWeights, settings))

    # deals run inside whole games so the dealer rotates and the scores move
    while dealsPlayed < settings["trainingDeals"]:
        game = WhistGame(config, deckSeed=deckSeed + gameNumber, playerSeed=playerSeed, players=players)
        gameNumber = gameNumber + 1

        while not game.gameOver() and dealsPlayed < settings["trainingDeals"]:
            for player in players:
                player.exploreRate = exploreRate

            playOneTrainingDeal(game, players)
            game.finishHand() # as playHand would
            applyUpdatesAndClear(players)

            dealsPlayed = dealsPlayed + 1
            exploreRate = decayedExploreRate(exploreRate, settings)

            # changes nothing about the training
            if curve is not None and dealsPlayed % settings["checkpointEvery"] == 0:
                row = makeCurveRow(dealsPlayed, sharedWeights, settings)
                curve.append(row)
                if progressEvery > 0:
                    print("  checkpoint at", dealsPlayed, "deals: versus rule",
                          round(row["ruleWinRate"], 1), "percent, versus random",
                          round(row["randomWinRate"], 1), "percent", flush=True)

            if progressEvery > 0 and dealsPlayed % progressEvery == 0:
                print("  trained on", dealsPlayed, "of", settings["trainingDeals"],
                      "deals; exploreRate", round(exploreRate, 4), flush=True)

    return sharedWeights


# ----- MEASURING A SET OF WEIGHTS -----
def makeOpponent(opponent: str, rng):
    # rule is the curve's yardstick because the random baseline saturates
    if opponent == "rule":
        return RuleBasedPlayer()
    return RandomPlayer(rng)


def evaluateWeights(weights: list, numberOfGames: int, deckSeed: int = 1, playerSeed: int = 1,
                    opponent: str = "random") -> int:
    # seating alternates like evaluateMatchup and every learning seat stays greedy
    config = GameConfig()
    wins = 0
    for gameNumber in range(numberOfGames):
        if gameNumber % 2 == 0: # learning on seats 0 and 2, which is side 0
            learningSeats = [0, 2]
            learningSide = 0
        else: # learning on seats 1 and 3, which is side 1
            learningSeats = [1, 3]
            learningSide = 1

        seatRngs = makeSeatRngs(playerSeed + gameNumber, config.numberOfPlayers)
        players = []
        for seat in range(config.numberOfPlayers):
            if seat in learningSeats:
                player = LearningPlayer(seatRngs[seat], config)
                player.weights = list(weights) # its own copy, so evaluation can never disturb the original
                players.append(player)
            else:
                players.append(makeOpponent(opponent, seatRngs[seat]))

        game = WhistGame(config, deckSeed=deckSeed + gameNumber, playerSeed=playerSeed + gameNumber, players=players)
        result = game.playGame()
        if result["winningSide"] == learningSide:
            wins = wins + 1
    return wins


# ----- WRITING THE TRAINING CURVE -----
def availableFilePath(folder: str, baseName: str, extension: str) -> str:
    # steps past taken names so nothing in results/ is ever overwritten
    path = os.path.join(folder, baseName + extension)
    if not os.path.exists(path):
        return path
    counter = 2 # the first run took the plain name, so the next free one starts here
    while True:
        path = os.path.join(folder, baseName + str(counter) + extension)
        if not os.path.exists(path):
            return path
        counter = counter + 1


def writeCurveFile(curve: list, settings: dict, deckSeed: int, playerSeed: int,
                   resultsDir: str = "results", baseName: str = "learningCurve") -> str:
    # the header records every setting and seed so the file explains itself
    os.makedirs(resultsDir, exist_ok=True) # touching nothing in it
    filePath = availableFilePath(resultsDir, baseName, ".csv")

    handle = open(filePath, "w")
    handle.write("# learning agent training curve, milestone 7g\n")
    handle.write("# produced by whist/trainLearning.py; this is a LEARNING run, not a search run\n")
    handle.write("# the opponent for the headline column is the rule-based agent, because the random\n")
    handle.write("# baseline saturates and cannot separate one competent agent from another\n")
    handle.write("# updateMode: " + str(settings["updateMode"]) + " (online is the 7g baseline, averaged the stabilisation experiment)\n")
    handle.write("# learningRate: " + str(settings["learningRate"]) + "\n")
    handle.write("# exploreStartRate: " + str(settings["exploreStartRate"]) + "\n")
    handle.write("# exploreDecay: " + str(settings["exploreDecay"]) + "\n")
    handle.write("# exploreMinRate: " + str(settings["exploreMinRate"]) + "\n")
    handle.write("# trainingDeals: " + str(settings["trainingDeals"]) + "\n")
    handle.write("# checkpointEvery: " + str(settings["checkpointEvery"]) + "\n")
    handle.write("# evaluationGames: " + str(settings["evaluationGames"]) + "\n")
    handle.write("# evaluationSeed: " + str(settings["evaluationSeed"]) + " (held constant at every checkpoint)\n")
    handle.write("# training deckSeed: " + str(deckSeed) + "\n")
    handle.write("# training playerSeed: " + str(playerSeed) + "\n")
    handle.write("dealsTrained,games,ruleWins,ruleWinRate,randomWins,randomWinRate\n")

    for row in curve:
        values = []
        values.append(str(row["dealsTrained"]))
        values.append(str(row["games"]))
        values.append(str(row["ruleWins"]))
        values.append(str(round(row["ruleWinRate"], 1)))
        values.append(str(row["randomWins"]))
        values.append(str(round(row["randomWinRate"], 1)))
        handle.write(",".join(values) + "\n")
    handle.close()
    return filePath


# ----- NAMING AND SAVING THE TRAINED MODEL -----
def weightsBaseNameForMode(updateMode: str) -> str:
    # the update mode goes in the name so runs cannot overwrite each other
    return "learningWeights" + updateMode[0].upper() + updateMode[1:]


def makeWeightsDetails(settings: dict, deckSeed: int, playerSeed: int) -> dict:
    # what a weights file needs to say about itself to be identifiable once it is sitting on disk
    details = {} # written into the file as comment lines, in this order
    details["updateMode"] = settings["updateMode"]
    details["trainingDeals"] = settings["trainingDeals"]
    details["learningRate"] = settings["learningRate"]
    details["deckSeed"] = deckSeed
    details["playerSeed"] = playerSeed
    return details


def saveTrainedWeights(weights: list, settings: dict, deckSeed: int, playerSeed: int,
                       folder: str = ".", overrideName: str | None = None) -> str:
    baseName = overrideName # a name asked for by hand wins over the mode-based one
    if baseName is None:
        baseName = weightsBaseNameForMode(settings["updateMode"])
    filePath = availableFilePath(folder, baseName, ".txt")

    saver = LearningPlayer(None, GameConfig()) # only used to reach saveWeights
    saver.weights = list(weights)
    saver.saveWeights(filePath, makeWeightsDetails(settings, deckSeed, playerSeed))
    return filePath


def untrainedWeights() -> list:
    # for the before and after comparison
    config = GameConfig()
    return list(LearningPlayer(None, config).weights) # no rng is drawn from, so None is safe here


# ----- RUNNABLE ENTRY POINT -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deckSeed", type=int, default=None)
    parser.add_argument("--playerSeed", type=int, default=None)
    parser.add_argument("--learningRate", type=float, default=0.01)
    parser.add_argument("--exploreStart", type=float, default=1.0)
    parser.add_argument("--exploreDecay", type=float, default=0.999)
    parser.add_argument("--exploreMin", type=float, default=0.05)
    parser.add_argument("--evalGames", type=int, default=200)
    parser.add_argument("--evalSeed", type=int, default=12345)
    parser.add_argument("--checkpointEvery", type=int, default=250)
    parser.add_argument("--updateMode", type=str, default="online")
    parser.add_argument("--curve", action="store_true")
    parser.add_argument("--curveFile", type=str, default=None)
    parser.add_argument("--weightsFile", type=str, default=None)
    parser.add_argument("--progressEvery", type=int, default=250)
    args = parser.parse_args()

    # treat --seed as a master value, matching main.py, so one number repeats a whole run
    deckSeed = args.deckSeed if args.deckSeed is not None else args.seed
    playerSeed = args.playerSeed if args.playerSeed is not None else args.seed

    settings = makeLearningSettings(
        learningRate=args.learningRate,
        exploreStartRate=args.exploreStart,
        exploreDecay=args.exploreDecay,
        exploreMinRate=args.exploreMin,
        trainingDeals=args.deals,
        checkpointEvery=args.checkpointEvery,
        evaluationGames=args.evalGames,
        evaluationSeed=args.evalSeed,
        updateMode=args.updateMode,
    )

    # name the curve file by update mode so runs are told apart in results/
    curveFileName = args.curveFile
    if curveFileName is None:
        curveFileName = "learningCurve"
        if args.updateMode != "online":
            curveFileName = "learningCurve" + args.updateMode[0].upper() + args.updateMode[1:] + "Update"

    print("training the learning agent by self play")
    print("settings:", settings)
    print("deckSeed", deckSeed, "playerSeed", playerSeed)

    curve = None
    if args.curve:
        curve = []

    trained = trainBySelfPlay(settings, deckSeed, playerSeed, args.progressEvery, curve)

    if curve is not None:
        print("")
        print("training curve, win rate against the rule-based agent:")
        print("dealsTrained", "ruleWins", "ruleWin%", "randomWins", "randomWin%")
        for row in curve:
            print(row["dealsTrained"], row["ruleWins"], round(row["ruleWinRate"], 1),
                  row["randomWins"], round(row["randomWinRate"], 1))
        curvePath = writeCurveFile(curve, settings, deckSeed, playerSeed, "results", curveFileName)
        print("wrote the curve to", curvePath)
    else:
        beforeWins = evaluateWeights(untrainedWeights(), args.evalGames)
        afterWins = evaluateWeights(trained, args.evalGames)
        print("untrained agent won", beforeWins, "of", args.evalGames, "games against random")
        print("trained agent won", afterWins, "of", args.evalGames, "games against random")
        print("gain:", afterWins - beforeWins, "games")

    # the model is not experiment output, so it stays out of results/
    weightsPath = saveTrainedWeights(trained, settings, deckSeed, playerSeed, ".", args.weightsFile)
    print("saved trained weights to", weightsPath)
