# this file contains no GUI code and no AI, meaning it can run a whole game with nothing on screen.

import random

from .cards import Card
from .rules import GameConfig, trumpNoTrump
from .deck import deal
from .playerView import PlayerView
from ..players.randomPlayer import RandomPlayer


# ----- TRICK HELPERS -----
def rankOfPlay(play: tuple[int, Card]) -> int:
    return play[1].rank # the rank of the card in a (seat, card) play


# ----- PER-SEAT RANDOM SOURCES -----
def makeSeatRngs(playerSeed: int, numberOfSeats: int) -> list:
    # gives every seat its own independent random source, so one seat drawing a number never shifts another's stream
    master = random.Random(playerSeed) # one master source that a whole experiment can be repeated from
    seatRngs = []
    for seat in range(numberOfSeats):
        # draw each seat's seed from the master rather than playerSeed + seat, so different experiments never overlap
        seatSeed = master.randrange(1000000000)
        seatRngs.append(random.Random(seatSeed))
    return seatRngs


# ----- GAME EVENTS -----
def makeStepEvent(what: str, seat: int | None = None, card: Card | None = None,
                  trickCards: list | None = None, trickWinnerSeat: int | None = None,
                  handFinished: bool = False) -> dict:
    # the single definition of a game event; the observe hook is meant to carry this same shape
    event = {}
    event["what"] = what # one of "movePlayed", "notReady", "handOver"
    event["seat"] = seat
    event["card"] = card
    event["trickCards"] = trickCards
    event["trickWinnerSeat"] = trickWinnerSeat
    event["handFinished"] = handFinished # true when that move emptied the hands
    return event


# ----- BUILDING A GAME FROM A POSITION -----
def gameFromPosition(
    config: GameConfig,
    hands: list,
    trumpSuit: str | None,
    whoseTurn: int,
    currentTrick: list | None = None,
    trickHistory: list | None = None,
    tricksWon: list | None = None,
    scores: list | None = None,
    players: list | None = None,
) -> "WhistGame":
    # never deals, so neither random source is used and players come from outside
    game = WhistGame(config, deckSeed=0, playerSeed=0, players=players)

    # copy every seat's hand before storing it, so playing the game out never changes the caller's lists
    copiedHands = []
    for hand in hands:
        copiedHands.append(list(hand))
    game.hands = copiedHands

    game.trumpSuit = trumpSuit # only the trump suit matters for playing a position out
    game.trumpCard = None # the turned-up card is not part of the view, so it is not needed here

    if currentTrick is None:
        game.currentTrick = []
    else:
        game.currentTrick = list(currentTrick)

    if trickHistory is None:
        game.trickHistory = []
    else:
        # copy each completed trick as well, so the stored history is fully independent of the caller's
        copiedHistory = []
        for trick in trickHistory:
            copiedHistory.append(list(trick))
        game.trickHistory = copiedHistory

    if tricksWon is None:
        game.tricksWon = [0 for side in config.partnerships]
    else:
        game.tricksWon = list(tricksWon)

    if scores is None:
        game.scores = [0 for side in config.partnerships]
    else:
        game.scores = list(scores)

    game.whoseTurn = whoseTurn

    # work out the leader rather than taking it as a parameter
    if game.currentTrick: # cards are already in the trick, so the leader played the first of them
        game.leader = game.currentTrick[0][0]
    else:
        game.leader = whoseTurn

    return game


# ----- THE WHIST GAME -----
class WhistGame:

    def __init__(self, config: GameConfig, deckSeed: int | None = None, playerSeed: int = 0, players: list | None = None) -> None:
        self.config = config
        self.deckRng = random.Random(deckSeed) # a source used only for dealing, so the deal is reproducible on its own
        self.seatRngs = makeSeatRngs(playerSeed, config.numberOfPlayers) # kept apart from the deal

        if players is None:
            self.players = [RandomPlayer(self.seatRngs[seat]) for seat in range(config.numberOfPlayers)]
        else:
            self.players = players

        # cumulative across the whole game, one entry per partnership
        self.scores = [0 for side in config.partnerships]

        self.dealerSeat = 0 # rotates clockwise each hand
        self.handsPlayed = 0

        # the per-hand state below is filled in by dealHand at the start of each hand
        self.hands = []
        self.trumpCard = None
        self.trumpSuit = None
        self.currentTrick = []
        self.trickHistory = []
        self.tricksWon = [] # one entry per partnership
        self.leader = 0
        self.whoseTurn = 0

    # ----- PARTNERSHIPS -----
    def sideOf(self, seat: int) -> int:
        # the partnership index into scores and tricksWon
        for side in range(len(self.config.partnerships)):
            if seat in self.config.partnerships[side]:
                return side
        return -1 # should never happen if the config lists every seat

    # ----- STARTING A HAND -----
    def dealHand(self) -> None:
        handSeed = self.deckRng.randrange(1000000000) # drawn only from the deck source
        hands, trumpCard = deal(self.config, seed=handSeed, dealerSeat=self.dealerSeat)

        self.hands = hands
        self.trumpCard = trumpCard
        self.trumpSuit = None
        if trumpCard is not None:
            self.trumpSuit = trumpCard.suit # trump is the suit of the dealer's last card

        self.currentTrick = []
        self.trickHistory = []
        self.tricksWon = [0 for side in self.config.partnerships]

        # the player to the dealer's left leads the first trick
        self.leader = (self.dealerSeat + 1) % self.config.numberOfPlayers
        self.whoseTurn = self.leader

    # ----- LEGAL MOVES -----
    def legalMoves(self, seat: int) -> list[Card]:
        # must follow the led suit if able, else any card
        hand = self.hands[seat]
        if not self.currentTrick:
            return list(hand)

        ledSuit = self.currentTrick[0][1].suit # the suit of the first card played this trick
        sameSuit = [card for card in hand if card.suit == ledSuit]
        if sameSuit:
            return sameSuit
        return list(hand)

    # ----- APPLYING A MOVE -----
    def applyMove(self, seat: int, card: Card) -> None:
        if seat != self.whoseTurn:
            raise ValueError("it is not this seat's turn to play")
        if card not in self.legalMoves(seat):
            raise ValueError("that card is not a legal move")

        self.hands[seat].remove(card)
        self.currentTrick.append((seat, card))

        if len(self.currentTrick) == self.config.numberOfPlayers:
            self.resolveTrick()
        else:
            self.whoseTurn = (self.whoseTurn + 1) % self.config.numberOfPlayers

    # ----- RESOLVING A TRICK -----
    def resolveTrick(self) -> None:
        ledSuit = self.currentTrick[0][1].suit # the suit that was led this trick

        # any trumps played beat everything else, so look for them first
        trumpPlays = []
        if self.trumpSuit is not None:
            trumpPlays = [play for play in self.currentTrick if play[1].suit == self.trumpSuit]

        if trumpPlays: # highest trump wins the trick
            winningPlay = max(trumpPlays, key=rankOfPlay)
        else: # no trumps, so highest card of the led suit wins
            ledPlays = [play for play in self.currentTrick if play[1].suit == ledSuit]
            winningPlay = max(ledPlays, key=rankOfPlay)

        winnerSeat = winningPlay[0]
        self.tricksWon[self.sideOf(winnerSeat)] += 1

        self.trickHistory.append(self.currentTrick)
        self.currentTrick = []
        self.leader = winnerSeat # the winner leads next
        self.whoseTurn = winnerSeat

    # ----- SCORING A HAND -----
    def scoreHand(self) -> None:
        # each side scores one point per trick above the book (6 in the standard 13-trick game)
        book = self.config.cardsPerHand // 2
        for side in range(len(self.config.partnerships)):
            if self.tricksWon[side] > book:
                self.scores[side] += self.tricksWon[side] - book
        # honours scoring is left off by default (config.countHonours) and added later

    def handOver(self) -> bool:
        return all(len(hand) == 0 for hand in self.hands)

    def gameOver(self) -> bool:
        return any(score >= self.config.targetScore for score in self.scores)

    # ----- BUILDING A PLAYER VIEW -----
    def cardsPlayedSoFar(self) -> list[tuple[int, Card]]:
        # the public history: every completed trick plus the trick in progress
        played = []
        for trick in self.trickHistory:
            played.extend(trick)
        played.extend(self.currentTrick)
        return played

    def viewFor(self, seat: int) -> PlayerView:
        # builds the restricted snapshot that this seat is allowed to see
        legal = [] # only fill in legal moves when it is actually this seat's turn
        if seat == self.whoseTurn:
            legal = self.legalMoves(seat)
        return PlayerView(
            seat=seat,
            ownHand=list(self.hands[seat]), # a copy of only this seat's hand
            trumpSuit=self.trumpSuit,
            currentTrick=list(self.currentTrick),
            cardsPlayedSoFar=self.cardsPlayedSoFar(),
            scores=list(self.scores),
            whoseTurn=self.whoseTurn,
            legalMoves=legal,
        )

    # ----- PLAYING ONE MOVE -----
    def playOneMove(self) -> dict:
        # hands control straight back, so an outside loop can drive the game a step at a time
        if self.handOver(): # a finished hand is a normal state that lasts many frames, so report rather than raise
            return makeStepEvent("handOver")

        seat = self.whoseTurn
        if not self.players[seat].isReady(): # a human with no move handed in yet is not ready
            return makeStepEvent("notReady", seat) # change no game state at all

        view = self.viewFor(seat)
        card = self.players[seat].getMove(view)
        tricksBefore = len(self.trickHistory)
        self.applyMove(seat, card) # so trick resolution stays in one place

        trickCards = None # filled in only when this move completed a trick
        trickWinnerSeat = None
        if len(self.trickHistory) > tricksBefore: # a trick finished, so it is the last stored trick
            # copied so a display reordering it cannot corrupt trickHistory
            trickCards = list(self.trickHistory[-1])
            trickWinnerSeat = self.leader # resolveTrick makes the trick winner the next leader

        return makeStepEvent("movePlayed", seat, card, trickCards, trickWinnerSeat, self.handOver())

    # ----- PLAYING ONE HAND -----
    def playOutHand(self) -> None:
        # does not deal, score or rotate
        while not self.handOver():
            event = self.playOneMove()
            if event["what"] == "notReady": # a seat with no move ready cannot be driven headless
                # fail loudly rather than spin forever on a seat that never advances
                raise ValueError("a seat was not ready; an outside loop is needed to drive that seat")

    def finishHand(self) -> None:
        # kept separate from playHand so an outside loop can call it when a hand finishes
        self.scoreHand()
        self.handsPlayed += 1
        self.dealerSeat = (self.dealerSeat + 1) % self.config.numberOfPlayers

    def playHand(self) -> None:
        self.dealHand()
        self.playOutHand()
        self.finishHand()

    # ----- PLAYING A FULL GAME -----
    def playGame(self) -> dict:
        while not self.gameOver():
            self.playHand()

        bestScore = max(self.scores)
        winningSide = self.scores.index(bestScore)
        return {
            "winningSide": winningSide,
            "scores": list(self.scores),
            "handsPlayed": self.handsPlayed,
        }


# ----- RUNNABLE ENTRY POINT -----
if __name__ == "__main__":
    config = GameConfig()
    game = WhistGame(config, deckSeed=1) # a fixed deck seed makes the run reproducible
    result = game.playGame()

    print("hands played:", result["handsPlayed"])
    print("final scores by side:", result["scores"])
    print("winning side:", result["winningSide"], "with seats", config.partnerships[result["winningSide"]])
