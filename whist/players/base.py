# the shared shape so any player plugs into the engine loop unchanged

from ..engine.playerView import PlayerView
from ..engine.cards import Card


# ----- THE PLAYER INTERFACE -----
class Player:

    def getMove(self, view: PlayerView) -> Card:
        # chosen only from what the view allows
        raise NotImplementedError("a real player must provide getMove")

    def observe(self, event) -> None:
        # optional hook telling the player what just happened, such as a card played or a trick won
        pass

    def isReady(self) -> bool:
        # true by default so an ai player is always ready
        return True
