from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFile, OrthographicLens
from player import Player


loadPrcFile("config/conf.prc")


class Base(ShowBase):
    def __init__(self):
        super().__init__()


class Game():
    '''The main game class.  This class encapsulates all non built-in game classes and their logic.'''
    def __init__(self, base):
        self.base = base
        self.player = Player(base=self.base, charId=0, player_model_file='output/ship/ship')
        self.base.taskMgr.add(self.player.move_ship, "move_task")
        self.base.taskMgr.add(self.player.update_animation, "update_animation")
        lens = OrthographicLens()
        lens.setFilmSize(640, 480)
        lens.setNearFar(-50, 50)
        self.base.cam.node().setLens(lens)

# Game Init
base = Base()
game = Game(base)
game.base.run()
