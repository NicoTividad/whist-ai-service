# the shapes that cross the wire; nothing here knows how to play Whist

from pydantic import BaseModel, Field, field_validator
from whist.engine.cards import Card


allowedSeats = ["human", "rule", "search", "learning", "random"]


class CardModel(BaseModel):
    suit: str = Field(pattern="^(clubs|diamonds|hearts|spades)$")
    rank: int = Field(ge=2, le=14)


def toCard(model: CardModel) -> Card:
    return Card(model.suit, model.rank)


def fromCard(card: Card) -> dict:
    return {"suit": card.suit, "rank": card.rank}


class GameStateModel(BaseModel):
    deckSeed: int = Field(ge=0)
    playerSeed: int = Field(ge=0)
    seats: list[str] = Field(min_length=4, max_length=4)
    handsPlayed: int = Field(ge=0)
    scores: list[int]
    moves: list[CardModel] = Field(max_length=52)
    sampleCount: int = Field(default=20, ge=1, le=160)

    @field_validator("seats")
    @classmethod
    def seatsMustBeKnown(cls, value: list) -> list:
        for name in value:
            if name not in allowedSeats:
                raise ValueError("unknown seat type: " + str(name))
        return value