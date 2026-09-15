# data only, so a simplified variant is a settings change not a rewrite

# ----- TRUMP CHOICE OPTIONS -----

trumpTurnUpDealerLast = "turnUpDealerLast"
trumpNoTrump = "noTrump"


# ----- GAME CONFIG -----
class GameConfig:

    def __init__(
        self,
        numberOfPlayers: int = 4,
        fixedPartnerships: bool = True,
        partnerships: list[list[int]] | None = None,
        deckRanks: list[int] | None = None,
        deckSuits: list[str] | None = None,
        cardsPerHand: int = 13,
        targetScore: int = 5,
        countHonours: bool = False,
        trumpChoice: str = trumpTurnUpDealerLast,
    ) -> None:
        self.numberOfPlayers = numberOfPlayers
        self.fixedPartnerships = fixedPartnerships

        # seats sit clockwise as 0, 1, 2, 3 (north, east, south, west), partners opposite
        if partnerships is None:
            partnerships = [[0, 2], [1, 3]]
        self.partnerships = partnerships

        if deckRanks is None:
            deckRanks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        self.deckRanks = deckRanks

        if deckSuits is None:
            deckSuits = ["clubs", "diamonds", "hearts", "spades"]
        self.deckSuits = deckSuits

        self.cardsPerHand = cardsPerHand
        self.targetScore = targetScore
        self.countHonours = countHonours
        self.trumpChoice = trumpChoice
