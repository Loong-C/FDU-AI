from hero import GameState
from layout import getLayout


def main():
    gameState = GameState()
    gameState.initialize(getLayout('smallClassic'), numDragonAgents=2)

    print('Coin positions:', gameState.getCoins().asList())
    print('Second dragon position:', gameState.getDragonPosition(2))


if __name__ == '__main__':
    main()
