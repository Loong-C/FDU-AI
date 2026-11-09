"""Launch the Hero game with keyboard ticks or agent-driven scheduling."""
import sys
from Game.Application.game_setup import readCommand, runGames

def main(argv=None):
    return runGames(**readCommand(sys.argv[1:] if argv is None else argv))
