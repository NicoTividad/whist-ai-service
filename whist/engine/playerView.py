# this keeps hidden information hidden: a view never contains another player's hand.

from .cards import Card


# ----- PLAYER VIEW -----
class PlayerView:

    def __init__(
        self,
        seat: int,
        ownHand: list[Card],
        trumpSuit: str | None, # None in a no-trump game
        currentTrick: list[tuple[int, Card]],
        cardsPlayedSoFar: list[tuple[int, Card]],
        scores: list[int], # one entry per partnership
        whoseTurn: int,
        legalMoves: list[Card],
    ) -> None:
        self.seat = seat
        self.ownHand = ownHand
        self.trumpSuit = trumpSuit
        self.currentTrick = currentTrick
        self.cardsPlayedSoFar = cardsPlayedSoFar
        self.scores = scores
        self.whoseTurn = whoseTurn
        self.legalMoves = legalMoves
