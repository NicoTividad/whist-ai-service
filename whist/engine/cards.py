# no deck, dealing, trump, or game logic lives here.

# ----- SUITS AND RANKS -----

allSuits = ["clubs", "diamonds", "hearts", "spades"]

# 2..10 are themselves, jack = 11, queen = 12, king = 13, ace = 14
allRanks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


# ----- RANK LABELS -----
def rankLabel(rank: int) -> str:
    if rank == 11:
        return "jack"
    if rank == 12:
        return "queen"
    if rank == 13:
        return "king"
    if rank == 14:
        return "ace"
    return str(rank)


# ----- CARDS -----
class Card:

    def __init__(self, suit: str, rank: int) -> None:
        self.suit = suit
        self.rank = rank

    # lets two cards be compared with == (equal when suit and rank match)
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.suit == other.suit and self.rank == other.rank

    # lets cards be used in lists, sets and as dict keys (must agree with __eq__)
    def __hash__(self) -> int:
        return hash((self.suit, self.rank))

    # gives a readable form so a card prints like "ace of spades"
    def __repr__(self) -> str:
        return rankLabel(self.rank) + " of " + self.suit
