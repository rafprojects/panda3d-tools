import time
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFile, OrthographicLens
from panda3d.core import CollisionTraverser, CollisionHandlerEvent, CollisionNode, CollisionPlane, Plane, Vec3, Point3, BitMask32
from src.player import Player
from src.entity import Enemy, EnemySpawner
from src.collision import handle_bullet, handle_enemy, get_bullet_and_enemy_from_entry


loadPrcFile("config/conf.prc")


class Base(ShowBase):
    """The base class for the game, separated in case of future expansion."""
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
        
        # PLAYFIELD
        self.scale_factor = 1 # 1:1 pixel-unit scale
        self.resolution_x = 640 * self.scale_factor
        self.resolution_y = 480 * self.scale_factor
        self.half_width = self.resolution_x / 2
        self.half_height = self.resolution_y / 2
        self.playfield_bounds = {
            'left': -self.half_width,
            'right': self.half_width,
            'top': self.half_height,
            'bottom': -self.half_height
        }
        self.planepos = {
            'left': (1, 0, 0),
            'right': (-1, 0, 0),
            'top': (0, -1, 0),
            'bottom': (0, 1, 0)
        }
        
        # COLLISION HANDLING
        self.boundaries = {k: CollisionPlane(Plane(Vec3(self.planepos[k]), Point3(v, 0, 0))) for k, v in self.playfield_bounds.items()}
        self.cTrav = CollisionTraverser()
        self.collHandler = CollisionHandlerEvent()
        self.collHandler.addInPattern('%fn-into-%in')
        self.collHandler.addOutPattern('%fn-out-%in')
        
        self.cTrav.showCollisions(self.base.render) # Debugging
        
        self.base.accept('bullet-into-enemy', self.bullet_enemy_collision)
        # self.base.accept('bullet-out-enemy', self.end_bullet_enemy_collision) # TODO: Implement this
        self.base.accept('player-into-enemy', self.player_enemy_collision)
        self.base.accept('player-out-enemy', self.end_player_enemy_collision)
        for side, boundary in self.boundaries.items(): # Playfield boundary collision setup
            coll_node = CollisionNode('boundary_' + side)
            coll_node.addSolid(boundary)
            coll_node.setIntoCollideMask(BitMask32.bit(1))
            self.base.render.attachNewNode(coll_node)
        self.active_collisions = set() # Track active collisions
        
        # PLAYER
        self.player = Player(base=self.base,
                             charId=0,
                             entity_type='player',
                             model_file='assets/sprites/ship/ship.egg',
                             cTrav=self.cTrav,
                             cHandler=self.collHandler
                             )
        # TASK MANAGER
        self.base.taskMgr.add(self.player.move_ship, "move_task")
        self.base.taskMgr.add(self.player.update_animation, "update_animation")
        self.base.taskMgr.add(self.player.update_bullets, "update_bullets")
        self.base.taskMgr.add(self.traverse_collisions, "traverse_task")
        self.base.taskMgr.add(self.apply_collision_damage, "collision_damage_task")
        
        # ENEMIES
        self.enemy_limit = 10
        self.enemies = []  # central enemy list
        self.enemy_ids_global = [] # Possibly don't need this
        self.enemy_spawner = EnemySpawner(
            base=self.base,
            enemy_class=Enemy,
            spawn_interval=300.0,
            spawn_area=(-300.0, 300.0, 50.0, 200.0),
            cTrav=self.cTrav,
            cHandler=self.collHandler,
            enemyL=self.enemies,
            global_enemy_idsL=self.enemy_ids_global,
            enemy_limit=self.enemy_limit,
            player_ref=self.player
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
            # Add the player-enemy pair to the active collisions set
            self.active_collisions.add((player, enemy))
            # Initialize last collision time to now
            player.last_collision = time.time()
            enemy.last_collision = time.time()

    def end_player_enemy_collision(self, entry):
        enemy_node = entry.getIntoNodePath().getPythonTag("enemy")
        enemy = next((e for e in self.enemies if e.collNode.getPythonTag('enemy') == enemy_node), None)
        player = self.player

        if enemy and player:
            # Remove the player-enemy pair from the active collisions set
            self.active_collisions.discard((player, enemy))

    def apply_collision_damage(self, task):
        cooldown = 0.3  # Time in seconds before damage can be applied again
        current_time = time.time()

        # Create a copy of the active collisions to avoid runtime errors during iteration
        active_collisions_copy = list(self.active_collisions)

        for player, enemy in active_collisions_copy:
            if current_time - player.last_collision >= cooldown:
                num_enemies = len([e for (p, e) in self.active_collisions if p == player])
                if num_enemies > 0:
                    # Each enemy and player share the damage equally
                    damage_per_entity = 2 / (num_enemies + 1)  # +1 for the player

                    # Apply damage to each enemy
                    for (p, e) in active_collisions_copy:
                        if p == player:
                            e.HP -= damage_per_entity
                            e.last_collision = current_time  # Reset the collision timer for this enemy
                            print(f"Enemy HP: {e.HP}")

                            if e.HP <= 0:
                                self.enemies.remove(e)
                                e.removeNode()
                                self.base.taskMgr.remove('update_enemy_{}'.format(e.id))
                                print("Enemy destroyed")
                                self.active_collisions.discard((player, e))  # End collision if enemy is destroyed

                    # Apply damage to the player
                    player.HP -= damage_per_entity * num_enemies
                    player.last_collision = current_time  # Reset the collision timer for the player
                    print("PLAYER HP: ", player.HP)

                    if player.HP <= 0:
                        print("Player died")

        return task.cont


# Game Init
base = Base()
game = Game(base)
game.base.run()
