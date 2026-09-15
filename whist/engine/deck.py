# this file only deals cards, so no trick play or scoring logic lives here.

import random

from .cards import Card
from .rules import GameConfig, trumpNoTrump


# ----- BUILDING THE DECK -----
def buildDeck(config: GameConfig) -> list[Card]:
    deck = []
    for suit in config.deckSuits:
        for rank in config.deckRanks:
            deck.append(Card(suit, rank))
    return deck


# ----- SHUFFLING THE DECK -----
def shuffleDeck(deck: list[Card], seed: int | None = None) -> list[Card]:
    rng = random.Random(seed) # a private random source seeded for reproducible deals
    shuffled = list(deck) # copy so the original deck order is left untouched
    rng.shuffle(shuffled)
    return shuffled


# ----- DEALING THE CARDS -----
def deal(config: GameConfig, seed: int | None = None, dealerSeat: int = 0) -> tuple[list[list[Card]], Card | None]:
    deck = buildDeck(config)
    deck = shuffleDeck(deck, seed)

    hands = [[] for seat in range(config.numberOfPlayers)]

    # start left of the dealer so the trump card lands on the dealer last
    leftOfDealer = (dealerSeat + 1) % config.numberOfPlayers
    nextCard = 0
    for cardNumber in range(config.cardsPerHand):
        for step in range(config.numberOfPlayers):
            seat = (leftOfDealer + step) % config.numberOfPlayers
            hands[seat].append(deck[nextCard])
            nextCard = nextCard + 1

    # the trump card stays in the dealer's hand, None in a no-trump game
    trumpCard = None
    if config.trumpChoice != trumpNoTrump and hands[dealerSeat]:
        trumpCard = hands[dealerSeat][-1]

    return hands, trumpCard
