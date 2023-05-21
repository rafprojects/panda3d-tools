import random
from panda3d.core import NodePath
from eggmodel import Eggmodel


class Entity(Eggmodel):
    def __init__(self, HP, pos, scale, parent, base, model_file):
        super().__init__(pos, scale, parent, base, model_file)
        self.HP = HP
        

class Enemy(Entity):
    def __init__(self, HP, pos, scale, parent, base, model_file, velocity):
        super().__init__(HP, pos, scale, parent, base, model_file)
        self.velocity = velocity
        
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
        x = random.uniform(self.spawn_area[0], self.spawn_area[1])
        y = random.uniform(self.spawn_area[2], self.spawn_area[3])
        enemy = self.enemy_class(
            HP=10, 
            pos=(x, 0, y), 
            scale=3.0, 
            parent=NodePath("enemy"), 
            base=self.base, 
            model_file='output/enemies/asteroid/asteroid', 
            velocity=1.0)
        self.enemies.append(enemy)
        self.base.render.attachNewNode(enemy)
        return task.again