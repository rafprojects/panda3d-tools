import random
import math
from panda3d.core import CollisionBox, CollisionNode, Point3, GeomNode, BoundingBox, NodePath

from .eggmodel import Eggmodel
from .common import get_box_dimensions, make_bounding_box
from .movement import StraightDown, SinusoidalMovement, LinearMovement


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
    def __init__(self, HP, pos, scale, base, model_file, entity_type, cTrav, cHandler, velocity, movement_behavior):
        super().__init__(HP, pos, scale, base, model_file, entity_type, cTrav, cHandler)
        self.id = Enemy._id
        Enemy._id += 1
        
        # Movement
        self.velocity = velocity
        self.direction = None
        self.max_speed = 30
        self.max_direction = 10
        
        self.movement_behavior = movement_behavior
        self.radius = 15 # for circ mvmt
        self.amplitude = 6
        self.frequency = 10
        
        # DBG
        # make_bounding_box(self)
    def update_pos(self, dt):
        self.movement_behavior.move(self, dt)
        # angle = (globalClock.getFrameTime() * self.velocity) % 360
        # x = math.cos(math.radians(angle)) * self.radius
        # y = math.sin(math.radians(angle)) * self.radius
        # self.setPos(x, y, self.getZ())
    
    def fire(self):
        pass
    
    def update(self, task):
        dt = globalClock.getDt()
        
        self.update_pos(dt)

        return task.cont

class EnemySpawner():
    def __init__(self, base, enemy_class, spawn_interval, spawn_area, cTrav, cHandler, enemyL, global_enemy_idsL, enemy_limit):
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
            mvmt_f = StraightDown()
            enemy = self.enemy_class(
                HP=10,
                pos=(x, 0, y),
                scale=0.4,
                base=self.base,
                model_file='assets/sprites/enemies/asteroid/asteroid.egg',
                entity_type='enemy',
                velocity=25.0,
                cTrav=self.cTrav,
                cHandler=self.cHandler,
                movement_behavior=mvmt_f
            )
            print(enemy.id)
            self.enemy_ids_sub.append(enemy.id)
            self.enemies.append(enemy)
            enemy.reparentTo(self.base.render)
            self.base.taskMgr.add(enemy.update, 'update_enemy_{}'.format(enemy.id))
        else:
            # self.enemies = [bullet.removeNode() for bullet in self.enemies]
            pass
        return task.again
