from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFile, OrthographicLens
from panda3d.core import CollisionTraverser, CollisionHandlerEvent
from src.player import Player
from src.entity import Enemy, EnemySpawner
from src.collision import handle_bullet, handle_enemy, get_bullet_and_enemy_from_entry


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
        self.enemies = []  # central enemy list
        self.enemy_spawner = EnemySpawner(
            base=self.base,
            game=self,
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
        print(f"ENEMIES: {self.enemies}")
        print(f"PLAYER_BULLETS: {self.player.bullets[0].collNodePath}")
        bullet_node, enemy_node = get_bullet_and_enemy_from_entry(entry)
        print(f"COLLIDE: B:{bullet_node} | E:{enemy_node}")
        print()
        bullet = next((b for b in self.player.bullets if b.collNodePath == bullet_node), None)
        if bullet:
            handle_bullet(bullet, self.player.bullets)
        enemy = next((e for e in self.enemies if e.collNodePath == enemy_node), None)
        if enemy:
            handle_enemy(enemy, self.enemies)


# Game Init
base = Base()
game = Game(base)
game.base.run()
