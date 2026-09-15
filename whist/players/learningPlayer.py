# features come only from the view and the move, never another seat's real hand

import random

from .base import Player
from ..engine.playerView import PlayerView
from ..engine.rules import GameConfig
from ..engine.cards import Card
from .ruleBasedPlayer import RuleBasedPlayer # reused for the trick-winner rule, see the note in __init__


# ----- NUMBER OF FEATURES -----
numberOfFeatures = 13 # the length of the vector featuresFor returns
numberOfWeights = numberOfFeatures + 1 # one weight per feature, plus a single bias term
biasIndex = numberOfFeatures # the bias sits in the last slot, so a feature's index is also its weight's index


# ----- THE LEARNING PLAYER -----
class LearningPlayer(Player):

    def __init__(self, rng: random.Random, config: GameConfig) -> None:
        self.rng = rng
        self.config = config # so the rank range, table size and partnerships stay settings

        # all zero so an untrained agent carries no hand-written judgement
        self.weights = [0.0 for slot in range(numberOfWeights)]

        # what this agent remembers while a deal is being played, all cleared at the deal boundary
        self.seat = None # learned from the first view it is shown
        self.dealFeatures = [] # in play order
        self.dealTrickWinners = [] # as the engine reported it

        # each one feature vector paired with the return of the deal it was played in
        self.trainingExamples = []

        # small on purpose, one deal's return is a noisy target
        self.learningRate = 0.01

        # training only, so an evaluated agent stays fully greedy
        self.exploreRate = 0.0

        # online is the baseline every measured number used, averaged is the experiment
        self.updateMode = "online"

        # resolveTrick cannot judge a part-played trick, so the rule agent's version is reused
        self.trickRules = RuleBasedPlayer()

    # ----- SMALL HELPERS -----
    def flag(self, condition: bool) -> float:
        # the 1.0 or 0.0 a linear model can use
        if condition:
            return 1.0
        return 0.0

    def sideOfSeat(self, seat: int) -> int:
        # same rule as the engine's sideOf, which a player cannot call
        for side in range(len(self.config.partnerships)):
            if seat in self.config.partnerships[side]:
                return side
        return -1

    def cardsOfSuitInHand(self, suit: str, hand: list) -> list:
        held = []
        for card in hand:
            if card.suit == suit:
                held.append(card)
        return held

    def scaledRank(self, rank: int) -> float:
        # a zero to one scale across the ranks this deck actually holds
        lowest = self.config.deckRanks[0]
        highest = self.config.deckRanks[0]
        for value in self.config.deckRanks:
            if value < lowest:
                lowest = value
            if value > highest:
                highest = value
        if highest == lowest: # a one-rank deck would divide by zero
            return 0.0
        return (rank - lowest) / (highest - lowest)

    def scaledBySuitSize(self, count: int) -> float:
        # scaled by a full suit rather than hand size, so the basis never moves
        cardsPerSuit = len(self.config.deckRanks)
        if cardsPerSuit == 0:
            return 0.0
        return count / cardsPerSuit

    def winningSeatOfTrick(self, currentTrick: list, trumpSuit: str | None) -> int:
        # the seat is found by matching bestCardSoFar's card, a trick never repeats a card
        bestCard = self.trickRules.bestCardSoFar(currentTrick, trumpSuit)
        for play in currentTrick:
            if play[1] == bestCard:
                return play[0]
        return currentTrick[0][0] # unreachable in practice, kept so a seat is always returned

    # ----- READING VOIDS FROM THE HISTORY -----
    def tricksFromHistory(self, playHistory: list) -> list:
        # mirrors SearchPlayer.tricksFromHistory, duplicated so saved results stay untouched
        tricks = []
        trickSize = self.config.numberOfPlayers
        current = []
        for play in playHistory:
            current.append(play)
            if len(current) == trickSize:
                tricks.append(current)
                current = []
        if current:
            tricks.append(current)
        return tricks

    def anySeatShownVoidIn(self, suit: str, view: PlayerView) -> bool:
        # the same void inference the search agent makes, for one suit
        for trick in self.tricksFromHistory(view.cardsPlayedSoFar):
            ledSuit = trick[0][1].suit
            if ledSuit != suit: # this trick says nothing about the suit being asked about
                continue
            for play in trick:
                if play[1].suit != ledSuit: # this seat could not follow, so it is void in the led suit
                    return True
        return False

    # ----- READING THE SCORE -----
    def scoreStanding(self, view: PlayerView) -> float:
        # 1.0 when this seat's side leads the score, 0.5 when level with the best other side, 0.0 when behind
        mySide = self.sideOfSeat(view.seat)
        myScore = view.scores[mySide]
        bestOther = None
        for side in range(len(view.scores)):
            if side == mySide:
                continue
            if bestOther is None or view.scores[side] > bestOther:
                bestOther = view.scores[side]
        if bestOther is None: # a one-side game has nobody to be ahead of
            return 0.5
        if myScore > bestOther:
            return 1.0
        if myScore == bestOther:
            return 0.5
        return 0.0

    # ----- THE FEATURE VECTOR -----
    def featuresFor(self, view: PlayerView, move: Card) -> list[float]:
        # fixed order matching report.md section 3.6, reordering breaks a trained agent
        trumpSuit = view.trumpSuit
        hand = view.ownHand # all the view offers
        sameSuitInHand = self.cardsOfSuitInHand(move.suit, hand)
        features = []

        # 1. the move's rank, scaled over the ranks the deck holds
        features.append(self.scaledRank(move.rank))

        # 2. whether the move is a trump
        features.append(self.flag(self.trickRules.isTrump(move, trumpSuit)))

        # 3. whether the move is the highest card of its own suit held in hand
        isHighestOfSuit = False # false when the move is not held, which only happens on misuse
        if sameSuitInHand:
            isHighestOfSuit = move.rank == self.trickRules.highestOf(sameSuitInHand).rank
        features.append(self.flag(isHighestOfSuit))

        # 4. whether the move is the lowest card of its own suit held in hand
        isLowestOfSuit = False
        if sameSuitInHand:
            isLowestOfSuit = move.rank == self.trickRules.lowestOf(sameSuitInHand).rank
        features.append(self.flag(isLowestOfSuit))

        # 5. whether playing this move now would take the trick as it stands, 0.0 when leading
        winsTrickNow = False # a leading move wins nothing yet, so this stays false
        if view.currentTrick:
            bestCard = self.trickRules.bestCardSoFar(view.currentTrick, trumpSuit)
            winsTrickNow = self.trickRules.beats(move, bestCard, trumpSuit)
        features.append(self.flag(winsTrickNow))

        # 6. how full the trick in progress is, scaled over the seats at the table
        features.append(len(view.currentTrick) / self.config.numberOfPlayers)

        # 7. whether a trump has already been played into the trick in progress
        trumpAlreadyPlayed = False
        for play in view.currentTrick:
            if self.trickRules.isTrump(play[1], trumpSuit):
                trumpAlreadyPlayed = True
        features.append(self.flag(trumpAlreadyPlayed))

        # 8. whether this seat's partner is winning the trick as it stands, 0.0 when leading
        partnerWinning = False # nobody is winning an empty trick
        if view.currentTrick:
            winningSeat = self.winningSeatOfTrick(view.currentTrick, trumpSuit)
            # the winner must be a different seat on the same side, so this seat itself never counts
            if winningSeat != view.seat and self.sideOfSeat(winningSeat) == self.sideOfSeat(view.seat):
                partnerWinning = True
        features.append(self.flag(partnerWinning))

        # 9. how much of a full suit is held in the move's own suit
        features.append(self.scaledBySuitSize(len(sameSuitInHand)))

        # 10. how much of a full suit is held in trumps
        trumpsInHand = 0
        for card in hand:
            if self.trickRules.isTrump(card, trumpSuit):
                trumpsInHand = trumpsInHand + 1
        features.append(self.scaledBySuitSize(trumpsInHand))

        # 11. how much of the move's suit has already been played this deal
        playedOfSuit = 0
        for play in view.cardsPlayedSoFar:
            if play[1].suit == move.suit:
                playedOfSuit = playedOfSuit + 1
        features.append(self.scaledBySuitSize(playedOfSuit))

        # 12. whether any seat has shown void in the move's suit
        features.append(self.flag(self.anySeatShownVoidIn(move.suit, view)))

        # 13. whether this seat's side is ahead on the game score
        features.append(self.scoreStanding(view))

        return features

    # ----- SCORING A MOVE -----
    def scoreFromFeatures(self, features: list) -> float:
        # kept apart from scoreMove so getMove builds features only once
        total = self.weights[biasIndex] # counts once for every move
        for index in range(len(features)):
            total = total + self.weights[index] * features[index]
        return total

    def scoreMove(self, view: PlayerView, move: Card) -> float:
        return self.scoreFromFeatures(self.featuresFor(view, move))

    # ----- SAVING AND LOADING THE WEIGHTS -----
    def saveWeights(self, filePath: str, details: dict | None = None) -> None:
        # repr round-trips a float exactly, so a reload is identical
        handle = open(filePath, "w")
        if details is not None:
            handle.write("# whist learning agent weights\n")
            for name in details: # written in the order the caller built them
                handle.write("# " + str(name) + ": " + str(details[name]) + "\n")
        for weight in self.weights:
            handle.write(repr(weight) + "\n")
        handle.close()

    def loadWeights(self, filePath: str) -> None:
        # replacing the ones held now
        handle = open(filePath, "r")
        lines = handle.readlines()
        handle.close()

        loaded = []
        for line in lines:
            text = line.strip()
            if not text: # such as one left at the end of the file
                continue
            if text.startswith("#"): # a header line saying how the file was produced
                continue
            loaded.append(float(text))

        # a wrong-sized file must fail loudly rather than load a different agent
        if len(loaded) != numberOfWeights:
            raise ValueError("expected " + str(numberOfWeights) + " weights but the file holds " + str(len(loaded)))
        self.weights = loaded

    # ----- CHOOSING A CARD -----
    def getMove(self, view: PlayerView) -> Card:
        # greedy over the learned value
        featureRows = [] # in the same order as view.legalMoves
        scores = []
        for move in view.legalMoves:
            features = self.featuresFor(view, move) # built once and kept, since the chosen one is recorded below
            featureRows.append(features)
            scores.append(self.scoreFromFeatures(features))

        bestScore = scores[0]
        for score in scores:
            if score > bestScore:
                bestScore = score

        bestIndexes = []
        for index in range(len(view.legalMoves)):
            if scores[index] == bestScore: # an exact match, so genuinely tied rather than merely close
                bestIndexes.append(index)

        # ties break by seat rng so untrained play is not one fixed card
        chosenIndex = self.rng.choice(bestIndexes)

        # training only, a zero exploreRate skips the rng so evaluation is unchanged
        if self.exploreRate > 0.0 and self.rng.random() < self.exploreRate:
            chosenIndex = self.rng.randrange(len(view.legalMoves))

        # record only the chosen move, from the view before it changes
        self.seat = view.seat
        self.dealFeatures.append(featureRows[chosenIndex])
        return view.legalMoves[chosenIndex]

    # ----- WATCHING THE DEAL -----
    def observe(self, event) -> None:
        # the engine never calls this itself, so the training loop must pass events on
        if event["what"] != "movePlayed": # notReady and handOver carry nothing this agent tracks
            return

        if event["trickWinnerSeat"] is not None: # the engine named its winner
            self.dealTrickWinners.append(event["trickWinnerSeat"])

        if event["handFinished"]: # the deal is over and its return is known
            self.finishDeal()

    def onMySide(self, seat: int) -> bool:
        # this agent's own seat or its partner
        for side in self.config.partnerships:
            if self.seat in side:
                return seat in side
        return False

    def tricksWonThisDeal(self) -> int:
        # the plain trick count for this side, not odd tricks or game score
        if self.seat is None: # this agent never played in the deal, so it has no side to count for
            return 0
        won = 0
        for winnerSeat in self.dealTrickWinners:
            if self.onMySide(winnerSeat):
                won = won + 1
        return won

    def finishDeal(self) -> None:
        dealReturn = self.tricksWonThisDeal() # one number, shared by every move played in this deal
        for features in self.dealFeatures:
            example = {}
            example["features"] = features
            example["dealReturn"] = dealReturn
            self.trainingExamples.append(example)

        # cleared at the deal boundary so two deals never bleed into one another
        self.dealFeatures = []
        self.dealTrickWinners = []

    # ----- THE LEARNING UPDATE -----
    def updateFromExample(self, features: list, dealReturn: float) -> None:
        # one least mean squares step, a zero feature leaves its weight unmoved
        prediction = self.scoreFromFeatures(features) # worked out before any weight moves
        error = dealReturn - prediction # positive means underrated

        for index in range(len(features)):
            self.weights[index] = self.weights[index] + self.learningRate * error * features[index]

        # the bias moves by the rate times the error alone, which is the same rule with a feature of one
        self.weights[biasIndex] = self.weights[biasIndex] + self.learningRate * error

    def updateFromExamples(self, examples: list) -> None:
        # each step uses the weights the previous step left behind, so this is plain online learning
        for example in examples:
            self.updateFromExample(example["features"], example["dealReturn"])

    def updateFromExamplesAveraged(self, examples: list) -> None:
        # every error uses the same starting weights, which makes this a genuine average
        if not examples: # nothing to average, so nothing moves
            return

        errors = [] # all measured against the weights as they stand now
        for example in examples:
            errors.append(example["dealReturn"] - self.scoreFromFeatures(example["features"]))

        # add up the step each example asks for, before any of them is applied
        steps = [0.0 for slot in range(numberOfWeights)]
        for index in range(len(examples)):
            features = examples[index]["features"]
            error = errors[index]
            for slot in range(len(features)):
                steps[slot] = steps[slot] + error * features[slot]
            steps[biasIndex] = steps[biasIndex] + error

        count = len(examples)
        for slot in range(numberOfWeights):
            self.weights[slot] = self.weights[slot] + self.learningRate * (steps[slot] / count)

    def applyUpdate(self, examples: list) -> None:
        # the routing lives here so a training loop never has to know which rule is in force
        if self.updateMode == "online": # the baseline: one step per example, in order
            self.updateFromExamples(examples)
            return
        if self.updateMode == "averaged":
            self.updateFromExamplesAveraged(examples)
            return
        # a bad mode must fail loudly, not quietly train by the wrong rule and produce a misleading curve
        raise ValueError("unknown update mode: " + str(self.updateMode))

    # ----- READING THE TRAINING EXAMPLES BACK -----
    def getTrainingExamples(self) -> list:
        return list(self.trainingExamples) # a copy of the list, so a caller cannot disturb the store

    def clearTrainingExamples(self) -> None:
        self.trainingExamples = []
