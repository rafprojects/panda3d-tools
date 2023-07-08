from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFile, OrthographicLens
from panda3d.core import CollisionTraverser, CollisionHandlerEvent
from src.player import Player
from src.entity import Enemy, EnemySpawner


loadPrcFile("config/conf.prc")


class Base(ShowBase):
    def __init__(self):
        super().__init__()
        
        
    


class Game():
    '''The main game class.  This class encapsulates all non built-in game classes and their logic.'''
    def __init__(self, base):
        self.base = base
        self.base.set_background_color(0.1, 0.1, 0.1, 1)
        # self.base.win.setAspectRatio(16.0/9.0)
        self.base.accept('window-event', self.adjust_aspect_ratio)
        
        self.enemies = []
        self.player = Player(base=self.base, charId=0, player_model_file='assets/sprites/ship/ship')
        self.base.taskMgr.add(self.player.move_ship, "move_task")
        self.base.taskMgr.add(self.player.update_animation, "update_animation")
        self.base.taskMgr.add(self.player.update_bullets, "update_bullets")
        
        # self.hfov = 75
        self.lens = OrthographicLens()
        self.lens.setFilmSize(640, 480)
        self.lens.setNearFar(-50, 50)
        # self.lens.setFov(self.hfov)
        self.base.cam.node().setLens(self.lens)
        
        self.enemy_spawner = EnemySpawner(
            base=self.base, 
            enemy_class=Enemy, 
            spawn_interval=300.0, 
            spawn_area=(-300.0, 300.0, 50.0, 200.0))
        
        # Collision Handling Block
        self.cTrav = CollisionTraverser()
        self.collHandler = CollisionHandlerEvent()
        self.collHandler.addInPattern('%fn-into-%in')
        self.base.accept('bullet-into-enemy', self.bullet_enemy_collision)
        
    def bullet_enemy_collision(self, entry):
        """Handles the collision between a bullet and an enemy."""
        bullet = entry.getFromNodePath().getPythonTag("owner")
        enemy = entry.getIntoNodePath().getPythonTag("owner")

        # Adjust HP or destroy the objects as needed
        bullet.HP -= 1
        enemy.HP -= 1

        if bullet.HP <= 0:
            bullet.destroy()
        if enemy.HP <= 0:
            enemy.destroy()
        
    def adjust_aspect_ratio(self, window=None):
        wp = window.getProperties()
        window_ratio = wp.getXSize() / wp.getYSize()
        
        # vfov = self.hfov / window_ratio
        # lens = self.base.cam.node().getLens()
        # lens = OrthographicLens()
        # lens.setAspectRatio(window_ratio)
        # lens.setFov(self.hfov, vfov)
        self.lens.setFilmSize(2 * window_ratio, 2)
        # self.base.cam.node().setLens(lens)

# Game Init
base = Base()
game = Game(base)
game.base.run()
