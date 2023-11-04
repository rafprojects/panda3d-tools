import time
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
        self.base.accept('player-into-enemy', self.player_enemy_collision)
        
        # player block
        self.player = Player(base=self.base,
                             charId=0,
                             entity_type='player',
                             model_file='assets/sprites/ship/ship.egg',
                             cTrav=self.cTrav,
                             cHandler=self.collHandler
                             )
        # Task Manager
        self.base.taskMgr.add(self.player.move_ship, "move_task")
        self.base.taskMgr.add(self.player.update_animation, "update_animation")
        self.base.taskMgr.add(self.player.update_bullets, "update_bullets")
        self.base.taskMgr.add(self.traverse_collisions, "traverse_task")
        
        # enemy stuff
        self.enemy_limit = 10
        self.enemies = []  # central enemy list
        self.enemy_ids_global = []
        self.enemy_spawner = EnemySpawner(
            base=self.base,
            enemy_class=Enemy,
            spawn_interval=300.0,
            spawn_area=(-300.0, 300.0, 50.0, 200.0),
            cTrav=self.cTrav,
            cHandler=self.collHandler,
            enemyL=self.enemies,
            global_enemy_idsL=self.enemy_ids_global,
            enemy_limit=self.enemy_limit
        )
        # print(self.enemy_spawner.enemies)
        # self.enemies.extend(self.enemy_spawner.enemies)
    
    def traverse_collisions(self, task):
        self.cTrav.traverse(self.base.render)
        return task.cont
    
    def bullet_enemy_collision(self, entry):
        """Handles the collision between a bullet and an enemy."""
        bullet_node = entry.getFromNodePath().getPythonTag("bullet")
        enemy_node = entry.getIntoNodePath().getPythonTag("enemy")

        bullet = next((b for b in self.player.bullets if b.collNode.getPythonTag('bullet') == bullet_node), None)
        enemy = next((e for e in self.enemies if e.collNode.getPythonTag('enemy') == enemy_node), None)
        if bullet and enemy:
            bullet.HP -= 1
            if bullet.HP <= 0:
                self.player.bullets.remove(bullet)
                bullet.removeNode()
            enemy.HP -= 2
            if enemy.HP <= 0:
                enemy.removeNode()
                self.enemies.remove(enemy)
                self.base.taskMgr.remove('update_enemy_{}'.format(enemy.id))

    def player_enemy_collision(self, entry):
        enemy_node = entry.getIntoNodePath().getPythonTag("enemy")
        enemy = next((e for e in self.enemies if e.collNode.getPythonTag('enemy') == enemy_node), None)
        player = self.player
        
        if enemy and player:
            current_time = time.time()
            
            if hasattr(enemy, 'last_collision') and current_time - enemy.last_collision < 0.3:
                return
            if hasattr(player, 'last_collision') and current_time - player.last_collision < 0.3:
                return
            
            enemy.last_collision = current_time
            player.last_collision = current_time
            
            enemy.HP -= 2
            player.HP -= 2
            print(player.HP)
            if enemy.HP <= 0:
                # print("Enemy destroyed")
                enemy.removeNode()
                self.enemies.remove(enemy)
                self.base.taskMgr.remove('update_enemy_{}'.format(enemy.id))
            if player.HP <= 0:
                print("Player died")

    # def update_enemies():
        

# Game Init
base = Base()
game = Game(base)
game.base.run()
