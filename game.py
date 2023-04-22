from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFile

from player import Player
# from panda3d.core import *


loadPrcFile("config/conf.prc")


class Base(ShowBase):
    def _init_(self):
        super().__init__()


class Game():
    '''The main game class.  This class encapsulates all non built-in game classes and their logic.'''
    def __init__(self, base):
        self.base = base        
        self.player = Player(base=self.base, charId=0)
        self.base.taskMgr.add(self.player.update, "move_task")
    

# Game Init
base = Base()
game = Game(base)

game.base.run()