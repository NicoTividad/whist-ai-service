# rebuilds a game from the wire state; the service keeps nothing between requests

from whist.engine.rules import GameConfig
from whist.engine.game import WhistGame, makeSeatRngs
from whist.main import makePlayers, makeSearchSettings

def rebuildGame(deckSeed: int, playerSeed: int, seats: list, handsPlayed: int,
                scores: list, moves: list, sampleCount: int = 20) -> WhistGame:
    config = GameConfig()
    seatRngs = makeSeatRngs(playerSeed + len(moves), config.numberOfPlayers)
    searchSettings = makeSearchSettings(sampleCount)
    players = makePlayers(seats, seatRngs, config, searchSettings)
    game = WhistGame(config, deckSeed=deckSeed, playerSeed=playerSeed, players=players)
    for deal in range(handsPlayed):
        game.dealHand()
        game.finishHand()
    game.dealHand()
    game.scores = list(scores)
    for card in moves:
        game.applyMove(game.whoseTurn, card)
    return game
