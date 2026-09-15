# no rules of its own, it just builds players and drives WhistGame

import argparse
import os # to check a weights file is really there before seating a learning agent

from .engine.rules import GameConfig
from .engine.game import WhistGame, makeSeatRngs
from .players.randomPlayer import RandomPlayer
from .players.human import HumanPlayer
from .players.ruleBasedPlayer import RuleBasedPlayer
from .players.searchPlayer import SearchPlayer
from .players.learningPlayer import LearningPlayer


# ----- THE TRAINED LEARNING AGENT -----
# a seated learning agent must be trained, an untrained one plays at random
defaultLearningWeightsFile = "learningWeightsAveraged.txt"


# ----- SEARCH AGENT SETTINGS -----
def makeSearchSettings(sampleCount=None, useVoidInference=True, selectionRule="average", useRulePlayouts=False):
    # so they can be passed around as a single value
    settings = {}
    settings["sampleCount"] = sampleCount # or None to use the agent's own default
    settings["useVoidInference"] = useVoidInference
    settings["selectionRule"] = selectionRule
    settings["useRulePlayouts"] = useRulePlayouts
    return settings


# ----- BUILDING THE PLAYERS -----
def makePlayers(seatTypes: list, seatRngs: list, config: GameConfig, searchSettings: dict | None = None,
                learningWeightsFile: str | None = None) -> list:
    if searchSettings is None:
        searchSettings = makeSearchSettings()
    if learningWeightsFile is None:
        learningWeightsFile = defaultLearningWeightsFile
    players = []
    for seat in range(len(seatTypes)): # by index so each takes its own rng
        name = seatTypes[seat]
        if name == "human":
            players.append(HumanPlayer())
        elif name == "rule":
            players.append(RuleBasedPlayer())
        elif name == "search":
            searchPlayer = SearchPlayer(seatRngs[seat], config)
            if searchSettings["sampleCount"] is not None: # lets the search cost be tuned from the command line
                searchPlayer.numberOfSamples = searchSettings["sampleCount"]
            searchPlayer.useVoidInference = searchSettings["useVoidInference"] # always present, so set it directly
            searchPlayer.selectionRule = searchSettings["selectionRule"]
            searchPlayer.useRulePlayouts = searchSettings["useRulePlayouts"]
            players.append(searchPlayer)
        elif name == "learning":
            learningPlayer = LearningPlayer(seatRngs[seat], config)
            # fail loudly on a missing model rather than seat a random-playing agent
            if not os.path.exists(learningWeightsFile):
                raise ValueError("no learning weights file at " + str(learningWeightsFile) +
                                 "; train one with whist.trainLearning, or pass --learningWeights")
            learningPlayer.loadWeights(learningWeightsFile)
            # exploreRate is already 0.0 on a fresh player, so it stays greedy
            players.append(learningPlayer)
        else: # any other name is treated as a random player
            players.append(RandomPlayer(seatRngs[seat]))
    return players


# ----- RUNNING ONE GAME -----
def runOneGame(deckSeed: int, playerSeed: int, seatTypes: list, searchSettings: dict | None = None,
               learningWeightsFile: str | None = None) -> dict:
    config = GameConfig()
    seatRngs = makeSeatRngs(playerSeed, config.numberOfPlayers)
    players = makePlayers(seatTypes, seatRngs, config, searchSettings, learningWeightsFile)
    game = WhistGame(config, deckSeed=deckSeed, playerSeed=playerSeed, players=players) # keeping both seeds apart
    result = game.playGame()

    # adding two keys changes less than returning the players as well
    searchMoves = 0
    searchSeconds = 0.0
    searchSimulatedMoves = 0 # the machine-independent cost
    for player in players:
        if isinstance(player, SearchPlayer):
            searchMoves = searchMoves + player.movesMade
            searchSeconds = searchSeconds + player.totalMoveSeconds
            searchSimulatedMoves = searchSimulatedMoves + player.simulatedMovesPlayed
    result["searchMoves"] = searchMoves
    result["searchSeconds"] = searchSeconds
    result["searchSimulatedMoves"] = searchSimulatedMoves
    return result


# ----- RUNNING MANY GAMES -----
def runManyGames(numberOfGames: int, startDeckSeed: int, startPlayerSeed: int, seatTypes: list) -> None:
    config = GameConfig() # only needed to know how many sides there are
    sideWins = [0 for side in config.partnerships]
    for gameNumber in range(numberOfGames):
        deckSeed = startDeckSeed + gameNumber
        playerSeed = startPlayerSeed + gameNumber
        result = runOneGame(deckSeed, playerSeed, seatTypes)
        winningSide = result["winningSide"]
        # print both seeds so the exact game can be reproduced from this line
        print("game", gameNumber, "deckSeed", deckSeed, "playerSeed", playerSeed, "won by side", winningSide, "scores", result["scores"])
        sideWins[winningSide] += 1
    print("summary: wins per side", sideWins)


# ----- EVALUATING ONE SEAT TYPE AGAINST ANOTHER -----
def evaluateMatchup(numberOfGames: int, startDeckSeed: int, startPlayerSeed: int, typeA: str, typeB: str,
                    searchSettings: dict | None = None, learningWeightsFile: str | None = None) -> dict:
    # two seats of typeA against two seats of typeB
    aWins = 0
    bWins = 0
    totalMoves = 0
    totalSeconds = 0.0
    totalSimulatedMoves = 0
    for gameNumber in range(numberOfGames):
        # alternate seating so typeA is not always on the same side
        if gameNumber % 2 == 0: # typeA on seats 0 and 2, which is side 0
            seatTypes = [typeA, typeB, typeA, typeB]
            aSide = 0
        else: # typeA on seats 1 and 3, which is side 1
            seatTypes = [typeB, typeA, typeB, typeA]
            aSide = 1
        deckSeed = startDeckSeed + gameNumber
        playerSeed = startPlayerSeed + gameNumber
        result = runOneGame(deckSeed, playerSeed, seatTypes, searchSettings, learningWeightsFile)
        if result["winningSide"] == aSide:
            aWins += 1
        else:
            bWins += 1
        totalMoves = totalMoves + result["searchMoves"]
        totalSeconds = totalSeconds + result["searchSeconds"]
        totalSimulatedMoves = totalSimulatedMoves + result["searchSimulatedMoves"]

    total = numberOfGames # every game ends with one side reaching the target
    aPercent = 0.0
    if total > 0:
        aPercent = aWins / total * 100
    averageMoveSeconds = 0.0 # zero when neither side searches
    if totalMoves > 0:
        averageMoveSeconds = totalSeconds / totalMoves
    # counted in simulated cards so it reproduces on any machine
    averageSimulatedMoves = 0.0
    if totalMoves > 0:
        averageSimulatedMoves = totalSimulatedMoves / totalMoves
    print(typeA, "side won", aWins, "of", total, "games")
    print(typeB, "side won", bWins, "of", total, "games")
    print(typeA, "win percentage:", round(aPercent, 1))
    print("average seconds per search move:", round(averageMoveSeconds, 6))
    print("total simulated moves:", totalSimulatedMoves)
    print("average simulated moves per search move:", round(averageSimulatedMoves, 1))
    print("search settings:", searchSettings) # record which conditions produced this output
    return {"aWins": aWins, "bWins": bWins, "total": total, "averageMoveSeconds": averageMoveSeconds,
            "searchMoves": totalMoves, "simulatedMoves": totalSimulatedMoves,
            "averageSimulatedMoves": averageSimulatedMoves}


# ----- SWEEPING THE SEARCH SAMPLE COUNT -----
def sweepSampleCounts(numberOfGames: int, startDeckSeed: int, startPlayerSeed: int, typeA: str, typeB: str,
                      sampleCounts: list, searchSettings: dict | None = None, learningWeightsFile: str | None = None) -> list:
    # so cost can be plotted against strength and time
    if searchSettings is None:
        searchSettings = makeSearchSettings()
    rows = []
    for sampleCount in sampleCounts:
        # so any other settings carry through unchanged
        thisSettings = dict(searchSettings)
        thisSettings["sampleCount"] = sampleCount
        result = evaluateMatchup(numberOfGames, startDeckSeed, startPlayerSeed, typeA, typeB, thisSettings, learningWeightsFile)
        aPercent = 0.0
        if result["total"] > 0:
            aPercent = result["aWins"] / result["total"] * 100
        row = {
            "sampleCount": sampleCount,
            "aWins": result["aWins"],
            "bWins": result["bWins"],
            "aPercent": aPercent,
            "averageMoveSeconds": result["averageMoveSeconds"],
            "simulatedMoves": result["simulatedMoves"],
            "averageSimulatedMoves": result["averageSimulatedMoves"],
        }
        rows.append(row)

    # the last two columns are the machine-independent cost
    print("sweep summary:", typeA, "versus", typeB, "over", numberOfGames, "games each")
    print("samples", "aWins", "bWins", "aWin%", "secPerMove", "simMoves", "simPerMove")
    for row in rows:
        print(row["sampleCount"], row["aWins"], row["bWins"], round(row["aPercent"], 1),
              round(row["averageMoveSeconds"], 6), row["simulatedMoves"], round(row["averageSimulatedMoves"], 1))
    return rows


# ----- EVALUATING THE RULE-BASED AGENT -----
# kept as a thin wrapper because whist/tests/testRuleBasedPlayer.py calls it and reads result["ruleWins"]
def evaluateRuleVsRandom(numberOfGames: int, startDeckSeed: int, startPlayerSeed: int) -> dict:
    result = evaluateMatchup(numberOfGames, startDeckSeed, startPlayerSeed, "rule", "random", makeSearchSettings())
    return {"ruleWins": result["aWins"], "randomWins": result["bWins"], "total": result["total"]}


# ----- RUNNABLE ENTRY POINT -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deckSeed", type=int, default=None)
    parser.add_argument("--playerSeed", type=int, default=None)
    parser.add_argument("--sideA", type=str, default="rule")
    parser.add_argument("--sideB", type=str, default="random")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--noVoidInference", action="store_true")
    parser.add_argument("--selectionRule", type=str, default="average")
    parser.add_argument("--useRulePlayouts", action="store_true")
    parser.add_argument("--learningWeights", type=str, default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sampleCounts", type=str, default="5,10,20,40")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()

    # treat --seed as a master value: when a specific seed is not given it falls back to --seed
    # so one number can repeat a whole experiment, while the two flags can still be set apart
    deckSeed = args.deckSeed if args.deckSeed is not None else args.seed
    playerSeed = args.playerSeed if args.playerSeed is not None else args.seed

    searchSettings = makeSearchSettings(args.samples, not args.noVoidInference, args.selectionRule, args.useRulePlayouts) # the flag turns void inference off

    # pygame is imported only here so everything else runs without it
    if args.gui:
        from .ui.pygameView import runView
        runView(not args.watch) # watch mode fills the human seat with ai as well
    elif args.sweep:
        sampleCounts = []
        for piece in args.sampleCounts.split(","):
            sampleCounts.append(int(piece))
        sweepSampleCounts(args.games, deckSeed, playerSeed, args.sideA, args.sideB, sampleCounts, searchSettings, args.learningWeights)
    else:
        evaluateMatchup(args.games, deckSeed, playerSeed, args.sideA, args.sideB, searchSettings, args.learningWeights)
