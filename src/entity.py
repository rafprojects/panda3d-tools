import random
from panda3d.core import CollisionBox, CollisionNode, Point3, GeomNode, BoundingBox, NodePath

from .eggmodel import Eggmodel
from .common import get_box_dimensions, make_bounding_box


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
        self.collNode.setPythonTag(self.entity_type, self.model)
        self.collNodePath.node().addSolid(self.collBox)
        cTrav.addCollider(self.collNodePath, cHandler)
        self.collNodePath.show()  # temporary show for debugging
        self.alive = True  # tracking to aid destroying the object
        
    def destroy(self):
        self.alive = False
        self.model.removeNode()
        
        
class Enemy(Entity):
    def __init__(self, HP, pos, scale, base, model_file, entity_type, cTrav, cHandler, velocity):
        super().__init__(HP, pos, scale, base, model_file, entity_type, cTrav, cHandler)
        self.velocity = velocity
        # DBG
        # make_bounding_box(self)
        
    def update_pos(self, dt):
        self.setZ(self.getZ() - self.velocity * dt)
    
    def fire(self):
        pass


class EnemySpawner():
    def __init__(self, base, enemy_class, spawn_interval, spawn_area, cTrav, cHandler):
        self.base = base
        self.enemy_class = enemy_class
        self.spawn_interval = spawn_interval
        self.spawn_area = spawn_area
        self.spawn_timer = 0.0
        self.enemies = []
        self.spawn_task = self.base.taskMgr.add(self.spawn_enemies, "spawn_enemies")
        self.cTrav = cTrav
        self.cHandler = cHandler
        
    def spawn_enemies(self, task):
        if len(self.enemies) < 20:
            x = random.uniform(self.spawn_area[0], self.spawn_area[1])
            y = random.uniform(self.spawn_area[2], self.spawn_area[3])
            enemy = self.enemy_class(
                HP=10,
                pos=(x, 0, y),
                scale=0.4,
                base=self.base,
                model_file='assets/sprites/enemies/asteroid/asteroid.egg',
                entity_type='enemy',
                velocity=1.0,
                cTrav=self.cTrav,
                cHandler=self.cHandler
            )
            self.enemies.append(enemy)
            enemy.reparentTo(self.base.render)
        else:
            # self.enemies = [bullet.removeNode() for bullet in self.enemies]
            pass
        return task.again
    