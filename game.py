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
        # SETUP
        self.base = base
        self.base.set_background_color(0.1, 0.1, 0.1, 1)
        lens = OrthographicLens()
        lens.setFilmSize(640, 480)
        lens.setNearFar(-50, 50)
        self.base.cam.node().setLens(lens)
        
        # Collision Handling Block
        self.cTrav = CollisionTraverser()
        self.collHandler = CollisionHandlerEvent()
        self.collHandler.addInPattern('%fn-into-%in')
        self.cTrav.showCollisions(self.base.render)
        self.base.accept('bullet-into-enemy', self.bullet_enemy_collision)
        
        # player block
        self.player = Player(base=self.base, 
                             charId=0, 
                             player_model_file='assets/sprites/ship/ship.egg', 
                             cTrav=self.cTrav, 
                             cHandler=self.collHandler
                             )
        self.base.taskMgr.add(self.player.move_ship, "move_task")
        self.base.taskMgr.add(self.player.update_animation, "update_animation")
        self.base.taskMgr.add(self.player.update_bullets, "update_bullets")
        self.base.taskMgr.add(self.traverse_collisions, "traverse_task")
        
        # enemy stuff
        self.enemies = []
        self.enemy_spawner = EnemySpawner(
            base=self.base, 
            enemy_class=Enemy, 
            spawn_interval=300.0, 
            spawn_area=(-300.0, 300.0, 50.0, 200.0),
            cTrav=self.cTrav,
            cHandler=self.collHandler
            )
        
    def traverse_collisions(self, task):
        self.cTrav.traverse(self.base.render)
        return task.cont
           
    def bullet_enemy_collision(self, entry):
        """Handles the collision between a bullet and an enemy."""
        bullet = entry.getFromNodePath().getPythonTag("bullet")
        enemy = entry.getIntoNodePath().getPythonTag("enemy")

        # Adjust HP or destroy the objects as needed
        bullet.HP -= 1
        enemy.HP -= 1

        if bullet.HP <= 0:
            bullet.destroy()
        if enemy.HP <= 0:
            enemy.destroy()

# Game Init
base = Base()
game = Game(base)
game.base.run()
