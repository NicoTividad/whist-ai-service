import random

from .base import Player
from ..engine.playerView import PlayerView
from ..engine.cards import Card


# ----- THE RANDOM PLAYER -----
class RandomPlayer(Player):
    # uses a seeded source for reproducible games

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def getMove(self, view: PlayerView) -> Card:
        return self.rng.choice(view.legalMoves)
