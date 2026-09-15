# works out which cards are unseen from the view, never who holds them

from .base import Player
from ..engine.playerView import PlayerView
from ..engine.cards import Card, allRanks


# ----- THE RULE-BASED PLAYER -----
class RuleBasedPlayer(Player):

    # ----- SMALL CARD HELPERS -----
    def isTrump(self, card: Card, trumpSuit: str | None) -> bool:
        # never true in a no-trump game
        return trumpSuit is not None and card.suit == trumpSuit

    def highestOf(self, cards: list[Card]) -> Card:
        # a plain loop rather than max
        best = cards[0]
        for card in cards:
            if card.rank > best.rank:
                best = card
        return best

    def lowestOf(self, cards: list[Card]) -> Card:
        worst = cards[0]
        for card in cards:
            if card.rank < worst.rank:
                worst = card
        return worst

    # ----- READING THE TRICK -----
    def bestCardSoFar(self, currentTrick: list[tuple[int, Card]], trumpSuit: str | None) -> Card:
        # the card winning the trick right now: highest trump, or highest of the led suit if no trump played
        ledSuit = currentTrick[0][1].suit

        trumpsPlayed = []
        for play in currentTrick: # each play is a (seat, card) pair
            if self.isTrump(play[1], trumpSuit):
                trumpsPlayed.append(play[1])
        if trumpsPlayed:
            return self.highestOf(trumpsPlayed)

        ledCards = []
        for play in currentTrick:
            if play[1].suit == ledSuit:
                ledCards.append(play[1])
        return self.highestOf(ledCards)

    def beats(self, card: Card, bestCard: Card, trumpSuit: str | None) -> bool:
        if self.isTrump(bestCard, trumpSuit):
            return self.isTrump(card, trumpSuit) and card.rank > bestCard.rank # only a higher trump wins
        # the trick is currently won by a non-trump (the led suit)
        if self.isTrump(card, trumpSuit): # any trump beats a non-trump leader
            return True
        return card.suit == bestCard.suit and card.rank > bestCard.rank # or a higher card of the same led suit

    # ----- REASONING ABOUT HIDDEN CARDS -----
    def seenRanksOfSuit(self, suit: str, view: PlayerView) -> list[int]:
        # those played this hand plus those in my own hand
        seen = []
        for play in view.cardsPlayedSoFar: # already includes the current trick
            if play[1].suit == suit:
                seen.append(play[1].rank)
        for card in view.ownHand: # my own cards are known to me too
            if card.suit == suit:
                seen.append(card.rank)
        return seen

    def isBossCard(self, card: Card, view: PlayerView) -> bool:
        # true when no higher card of this suit is still unseen, so this card is the top one left
        seenRanks = self.seenRanksOfSuit(card.suit, view)
        for rank in allRanks:
            if rank > card.rank and rank not in seenRanks:
                return False # so an opponent could hold it and beat my card
        return True

    # ----- CHOOSING A CARD -----
    def getMove(self, view: PlayerView) -> Card:
        if not view.currentTrick:
            return self.leadCard(view)
        return self.followCard(view)

    def leadCard(self, view: PlayerView) -> Card:
        # prefer to cash boss cards (the top card left of a suit), otherwise fall back to the simple lead
        nonTrumpBosses = []
        trumpBosses = []
        for card in view.legalMoves:
            if self.isBossCard(card, view):
                if self.isTrump(card, view.trumpSuit):
                    trumpBosses.append(card)
                else:
                    nonTrumpBosses.append(card)

        # a side-suit boss can still be trumped, so cash these early
        if nonTrumpBosses:
            return self.highestOf(nonTrumpBosses)
        if trumpBosses: # a top trump cannot be beaten by anything, so it is always safe to cash
            return self.highestOf(trumpBosses)

        # no boss card held, so fall back to the simple lead
        nonTrumps = []
        for card in view.legalMoves:
            if not self.isTrump(card, view.trumpSuit):
                nonTrumps.append(card)
        if nonTrumps: # prefer to lead a non-trump so trumps are kept back
            return self.highestOf(nonTrumps)
        return self.highestOf(view.legalMoves)

    def followCard(self, view: PlayerView) -> Card:
        # win as cheaply as possible, or throw the lowest card away
        bestCard = self.bestCardSoFar(view.currentTrick, view.trumpSuit)

        winners = []
        for card in view.legalMoves:
            if self.beats(card, bestCard, view.trumpSuit):
                winners.append(card)

        if winners:
            nonTrumpWinners = []
            for card in winners:
                if not self.isTrump(card, view.trumpSuit):
                    nonTrumpWinners.append(card)
            if nonTrumpWinners: # prefer winning without spending a trump
                return self.lowestOf(nonTrumpWinners)
            return self.lowestOf(winners)

        # we cannot win, so throw away the lowest card and keep the good ones
        nonTrumps = []
        for card in view.legalMoves:
            if not self.isTrump(card, view.trumpSuit):
                nonTrumps.append(card)
        if nonTrumps: # prefer to throw a non-trump and hold on to trumps
            return self.lowestOf(nonTrumps)
        return self.lowestOf(view.legalMoves)
