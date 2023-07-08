import random
import PIL
from panda3d.core import CollisionBox, CollisionNode, Point3

from .eggmodel import Eggmodel
from .common import get_dimensions_from_egg


class Entity(Eggmodel):
    def __init__(self, HP, pos, scale, base, model_file):
        super().__init__(pos, scale, base, model_file)
        self.HP = HP
        self.scale = scale
        

class Enemy(Entity):
    def __init__(self, HP, pos, scale, base, model_file, velocity):
        super().__init__(HP, pos, scale, base, model_file)
        self.velocity = velocity
        # Collision stuff
        size_x, size_y = get_dimensions_from_egg(model_file)
        self.collBox = CollisionBox(Point3(-size_x*self.scale, -0.5, -size_y*self.scale), Point3(size_x*self.scale, 0.5, size_y*self.scale))
        self.collNode = CollisionNode('enemy')
        self.collNode.addSolid(self.collBox)
        self.collNodePath = self.attachNewNode(self.collNode)
        self.collNodePath.show()  # temporary show for debugging
        
    def update_pos(self, dt):
        self.setZ(self.getZ() - self.velocity * dt)
    
    def fire(self):
        pass


class EnemySpawner():
    def __init__(self, base, enemy_class, spawn_interval, spawn_area):
        self.base = base
        self.enemy_class = enemy_class
        self.spawn_interval = spawn_interval
        self.spawn_area = spawn_area
        self.spawn_timer = 0.0
        self.enemies = []
        self.spawn_task = self.base.taskMgr.add(self.spawn_enemies, "spawn_enemies")
        
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
                velocity=1.0)
            self.enemies.append(enemy)
            # enemy.reparentTo(self.base.render)
            # print(self.enemies)
        else:
            # self.enemies = [bullet.removeNode() for bullet in self.enemies]
            pass
        return task.again
    