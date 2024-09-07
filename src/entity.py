import random
import math
from panda3d.core import CollisionBox, CollisionNode, Point3, GeomNode, BoundingBox, NodePath

from .eggmodel import Eggmodel
from .common import get_box_dimensions, make_bounding_box
from .movement import StraightDown, SinusoidalMovement, LinearMovement, CircularMovement, RandomMovement, FigureEightMovement, OutwardExpandingSpiralMovement, PendulumMovement, ZigZagMovement, RandomWalkMovement, ArcMovement, TeleportMovement, StraightEvadingMovement, EvadingMovement, WaitAndBeelineMovement


class Entity(Eggmodel):
    def __init__(self, HP, pos, scale, base, model_file, entity_type, cTrav, cHandler):
        super().__init__(pos, scale, base, model_file)
        self.HP = HP
        self.scale = scale
        self.entity_type = entity_type
    
        # Collision stuff
        pointA, pointB = get_box_dimensions(eggfile=model_file, scale_factor=self.scale, offsetT=None)
        self.collBox = CollisionBox(
            Point3(pointA[0], pointA[1], pointA[2]),
            Point3(pointB[0], pointB[1], pointB[2])
        )
        self.collNode = CollisionNode(self.entity_type)
        self.collNodePath = self.model.attachNewNode(self.collNode)
        self.collNode.setPythonTag(self.entity_type, self)
        self.collNodePath.node().addSolid(self.collBox)
        cTrav.addCollider(self.collNodePath, cHandler)
        self.collNodePath.show()  # temporary show for debugging
        self.alive = True  # tracking to aid destroying the object
        self.last_collision = 0
        
    
    def destroy(self):
        self.alive = False
        self.model.removeNode()


class Enemy(Entity):
    _id = 0
    def __init__(self, HP, pos, scale, base, model_file, entity_type, cTrav, cHandler, velocity, movement_behavior, player_ref=None):
        super().__init__(HP, pos, scale, base, model_file, entity_type, cTrav, cHandler)
        self.id = Enemy._id
        Enemy._id += 1
        self.player = player_ref # Reference to player object, used for tracking & targeting
        
        # Movement
        self.velocity = velocity
        self.direction = None
        self.max_speed = 50
        self.max_direction = 20
        self.movement_behavior = movement_behavior
        self.radius = 0.2 # for circ mvmt
        self.angle = 0 # for circ mvmt
        self.amplitude = 6 # for sinusoidal mvmt
        self.frequency = 10 # for sinusoidal mvmt
        
        self.change_interval = 3
        
        # DBG
        # make_bounding_box(self)
    def update_pos(self, dt):
        if hasattr(self.movement_behavior, 'move'):
            self.movement_behavior.move(self, dt, self.player)
        
        # Boundary Check
        min_x, max_x = -350, 350
        min_y, max_y = -50, 50
        min_z, max_z = -450, 450
        
        x, y, z = self.getPos()
        x = max(min(x, max_x), min_x)
        y = max(min(y, max_y), min_y)
        z = max(min(z, max_z), min_z)
        self.setPos(x, y, z)
    
    def fire(self):
        pass
    
    def update(self, task):
        dt = globalClock.getDt()
        
        self.update_pos(dt)

        return task.cont

class EnemySpawner():
    def __init__(self, base, enemy_class, spawn_interval, spawn_area, cTrav, cHandler, enemyL, global_enemy_idsL, enemy_limit, player_ref=None):
        self.base = base
        self.enemy_class = enemy_class
        self.spawn_interval = spawn_interval
        self.spawn_area = spawn_area
        self.spawn_timer = 0.0
        self.enemy_limit = enemy_limit
        self.enemies = enemyL
        self.enemy_ids_sub = global_enemy_idsL
        self.spawn_task = self.base.taskMgr.add(self.spawn_enemies, "spawn_enemies")
        self.cTrav = cTrav
        self.cHandler = cHandler
        self.player = player_ref
    
    # def generate_enemy():
    # def populate_id(self, limit):
    #     id = None
    #     while not id:
    #         i = random.choice(range(1,limit))
    #         if i not in self.enemy_ids_sub:
    #             print(i)
    #             self.enemy_ids_sub.append(i)
    #             id = i
    #             break
    #     return id
    
    def spawn_enemies(self, task):
        if len(self.enemies) < self.enemy_limit:
            
            x = random.uniform(self.spawn_area[0], self.spawn_area[1])
            y = random.uniform(self.spawn_area[2], self.spawn_area[3])
            # mvmt_f = StraightDown()
            # mvmt_f = FigureEightMovement(amplitude=60, frequency=0.5, speed=1)
            # mvmt_f = SpiralMovement(initial_radius=0, spiral_rate=3, rotation_speed=0.5, speed=1)
            # mvmt_f = PendulumMovement(amplitude=50, frequency=1, speed=2)
            # mvmt_f = ZigZagMovement(amplitude=50, frequency=1, speed=2)
            # mvmt_f = RandomWalkMovement(step_size=30, speed=10)
            # mvmt_f = ArcMovement(radius=200, angle_range=270, speed=2)
            # mvmt_f = TeleportMovement(teleport_interval=10, area_bounds=[(50, 400), (30, 400)], speed=1.0)
            # mvmt_f = EvadingMovement(evade_distance=50, evade_speed=75.0, wander_speed=20.0)
            # mvmt_f = StraightEvadingMovement(evade_distance=50, evade_speed=200.0, wander_speed=50.0)
            mvmt_f = WaitAndBeelineMovement(wait_time=3, beeline_speed=1000, max_beeline_distance=None, follow_through=True)
            enemy = self.enemy_class(
                HP=1,
                pos=(x, 0, y),
                scale=0.4,
                base=self.base,
                model_file='assets/sprites/enemies/asteroid/asteroid.egg',
                entity_type='enemy',
                velocity=30,
                cTrav=self.cTrav,
                cHandler=self.cHandler,
                movement_behavior=mvmt_f,
                player_ref=self.player
            )
            # print(enemy.id)
            self.enemy_ids_sub.append(enemy.id)
            self.enemies.append(enemy)
            enemy.reparentTo(self.base.render)
            self.base.taskMgr.add(enemy.update, 'update_enemy_{}'.format(enemy.id))
        else:
            # self.enemies = [bullet.removeNode() for bullet in self.enemies]
            pass
        return task.again
