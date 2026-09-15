from .base import Player
from ..engine.playerView import PlayerView
from ..engine.cards import Card


# ----- THE HUMAN PLAYER -----
class HumanPlayer(Player):
    # a person plays this seat, so the move is supplied from outside rather than chosen here

    def __init__(self) -> None:
        self.pendingMove: Card | None = None

    def setMove(self, card: Card) -> None:
        self.pendingMove = card

    def getMove(self, view: PlayerView) -> Card:
        if self.pendingMove is None:
            raise ValueError("no move has been supplied yet")
        card = self.pendingMove
        self.pendingMove = None # clear it so it is not reused by accident
        return card

    def isReady(self) -> bool:
        # the human is only ready once outside code has handed a card in through setMove
        return self.pendingMove is not None
