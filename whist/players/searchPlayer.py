# it chooses cards by simple pimc which is guess the hidden hands, play them out, and pick the best move.

import random
import time

from .base import Player
from ..engine.playerView import PlayerView
from ..engine.rules import GameConfig
from ..engine.cards import Card
from ..engine.game import gameFromPosition
from .randomPlayer import RandomPlayer
from .ruleBasedPlayer import RuleBasedPlayer


# ----- THE SEARCH-BASED PLAYER -----
class SearchPlayer(Player):

    def __init__(self, rng: random.Random, config: GameConfig) -> None:
        self.rng = rng # the seat's own seeded source
        self.config = config # so the sampler knows the full deck and seat count
        self.maxDealAttempts = 50 # how many shuffles to try before falling back to the plain deal
        self.numberOfSamples = 20 # guessed deals per legal move, kept as a setting so cost can be tuned
        self.useRulePlayouts = False # whether simulated seats play by heuristics; random is cheaper
        self.useVoidInference = True # whether guessed deals respect known voids
        self.selectionRule = "average" # average, bestCase or worstCase
        self.movesMade = 0 # for the time-per-move metric
        self.totalMoveSeconds = 0.0
        self.simulatedMovesPlayed = 0 # the machine-independent cost measure

    # ----- BUILDING A GUESSED DEAL -----
    def fullDeck(self) -> list[Card]:
        deck = []
        for suit in self.config.deckSuits:
            for rank in self.config.deckRanks:
                deck.append(Card(suit, rank))
        return deck

    def unseenCards(self, view: PlayerView) -> list[Card]:
        # not in its own hand and not yet played this hand
        seen = []
        for card in view.ownHand:
            seen.append(card)
        for play in view.cardsPlayedSoFar: # each play is a (seat, card) pair
            seen.append(play[1])

        unseen = []
        for card in self.fullDeck():
            if card not in seen:
                unseen.append(card)
        return unseen

    def cardsLeftPerSeat(self, view: PlayerView) -> list[int]:
        left = []
        for seat in range(self.config.numberOfPlayers):
            left.append(self.config.cardsPerHand)
        # cardsPlayedSoFar already includes the trick in progress
        for play in view.cardsPlayedSoFar:
            seat = play[0]
            left[seat] = left[seat] - 1
        return left

    def plainSampleDeal(self, view: PlayerView) -> list[list[Card]]:
        # the unconstrained version, kept for comparison: guesses a deal ignoring any known voids
        left = self.cardsLeftPerSeat(view)

        # shuffle a copy of the unseen cards so we never disturb the view or a shared list
        pool = list(self.unseenCards(view))
        self.rng.shuffle(pool)

        hands = []
        for seat in range(self.config.numberOfPlayers):
            hands.append([])

        nextCard = 0
        for seat in range(self.config.numberOfPlayers):
            if seat == view.seat: # my own hand is known, so it is copied rather than guessed
                hands[seat] = list(view.ownHand)
            else:
                for count in range(left[seat]):
                    hands[seat].append(pool[nextCard])
                    nextCard = nextCard + 1
        return hands

    def sampleDeal(self, view: PlayerView) -> list[list[Card]]:
        # no seat is dealt a suit it has shown it cannot hold
        if not self.useVoidInference: # the ablation: no inference at all
            return self.plainSampleDeal(view)
        left = self.cardsLeftPerSeat(view)
        voids = self.voidSuitsPerSeat(view)

        attempt = 0
        while attempt < self.maxDealAttempts:
            attempt = attempt + 1

            pool = list(self.unseenCards(view))
            self.rng.shuffle(pool)

            hands = []
            for seat in range(self.config.numberOfPlayers):
                hands.append([])

            deadEnd = False # true once a card has nowhere legal left to go
            for card in pool:
                placed = False
                for seat in range(self.config.numberOfPlayers):
                    if seat == view.seat: # my own hand is fixed, never dealt into from the pool
                        continue
                    if len(hands[seat]) >= left[seat]:
                        continue
                    if card.suit in voids[seat]:
                        continue
                    hands[seat].append(card)
                    placed = True
                    break
                if not placed:
                    deadEnd = True
                    break

            if not deadEnd:
                hands[view.seat] = list(view.ownHand)
                return hands

        # a rare safety net when the voids are too tight for the shuffles
        return self.plainSampleDeal(view)

    # ----- READING VOIDS FROM THE HISTORY -----
    def tricksFromHistory(self, view: PlayerView) -> list:
        # every full trick holds numberOfPlayers cards
        tricks = []
        trickSize = self.config.numberOfPlayers
        current = []
        for play in view.cardsPlayedSoFar: # in play order
            current.append(play)
            if len(current) == trickSize:
                tricks.append(current)
                current = []
        if current: # the last chunk is the trick in progress and may be short
            tricks.append(current)
        return tricks

    def voidSuitsPerSeat(self, view: PlayerView) -> list:
        voids = []
        for seat in range(self.config.numberOfPlayers):
            voids.append([])

        for trick in self.tricksFromHistory(view):
            ledSuit = trick[0][1].suit
            for play in trick:
                seat = play[0]
                card = play[1]
                # a trump played instead of following counts too, since it still shows the seat had none of the led suit
                if card.suit != ledSuit:
                    if ledSuit not in voids[seat]:
                        voids[seat].append(ledSuit)
        return voids

    # ----- PLAYING OUT A GUESSED DEAL -----
    def completedTricks(self, view: PlayerView) -> list:
        # the rule-based playout reads the public history
        finished = []
        for trick in self.tricksFromHistory(view):
            if len(trick) == self.config.numberOfPlayers:
                finished.append(trick)
        return finished

    def playoutPlayers(self) -> list:
        # sharing the agent's own rng is fine as the whole simulation is this agent's and stays reproducible
        players = []
        for seat in range(self.config.numberOfPlayers):
            if self.useRulePlayouts:
                players.append(RuleBasedPlayer())
            else: # the cheaper default
                players.append(RandomPlayer(self.rng))
        return players

    def simulateOnce(self, view: PlayerView, move: Card) -> int:
        # the tricks this seat's side wins from here to the end of the hand
        hands = self.sampleDeal(view) # guess the hidden hands, never the real ones

        # counting from zero is safe since every candidate shares the already-won tricks
        game = gameFromPosition(
            self.config,
            hands,
            view.trumpSuit,
            view.seat,
            currentTrick=view.currentTrick,
            trickHistory=self.completedTricks(view),
            players=self.playoutPlayers(),
        )

        # counted before the candidate move so that move is included too
        cardsToPlay = 0
        for hand in game.hands:
            cardsToPlay = cardsToPlay + len(hand)
        self.simulatedMovesPlayed = self.simulatedMovesPlayed + cardsToPlay

        game.applyMove(view.seat, move)
        game.playOutHand()
        return game.tricksWon[game.sideOf(view.seat)] # using the engine's partnership rule

    def scoreMove(self, view: PlayerView, move: Card) -> float:
        scores = []
        for sample in range(self.numberOfSamples):
            scores.append(self.simulateOnce(view, move))
        return self.aggregateScores(scores)

    def aggregateScores(self, scores: list) -> float:
        if self.selectionRule == "average": # the statistical rule
            total = 0
            for score in scores:
                total = total + score
            return total / len(scores)
        if self.selectionRule == "bestCase": # the optimistic rule
            best = scores[0]
            for score in scores:
                if score > best:
                    best = score
            return best
        if self.selectionRule == "worstCase": # the pessimistic rule
            worst = scores[0]
            for score in scores:
                if score < worst:
                    worst = score
            return worst
        # a bad rule must fail loudly, not quietly produce a wrong row in an experiment table
        raise ValueError("unknown selection rule: " + str(self.selectionRule))

    def getMove(self, view: PlayerView) -> Card:
        startTime = time.perf_counter()

        if len(view.legalMoves) == 1: # following suit often forces a single card, so searching it would be wasted work
            chosen = view.legalMoves[0]
        else:
            # on a tie keep the move found first so the choice stays deterministic
            bestCard = None
            bestAverage = 0.0
            for move in view.legalMoves:
                average = self.scoreMove(view, move)
                if bestCard is None or average > bestAverage:
                    bestCard = move
                    bestAverage = average
            chosen = bestCard

        # the single-move shortcut is timed too, or the average would be flattered
        self.totalMoveSeconds = self.totalMoveSeconds + (time.perf_counter() - startTime)
        self.movesMade = self.movesMade + 1
        return chosen

    def averageMoveSeconds(self) -> float:
        if self.movesMade == 0: # avoid dividing by zero
            return 0.0
        return self.totalMoveSeconds / self.movesMade

    def simulatedMovesPerRealMove(self) -> float:
        # unlike the time above this is fixed by the seed, so it reproduces anywhere
        if self.movesMade == 0:
            return 0.0
        return self.simulatedMovesPlayed / self.movesMade
