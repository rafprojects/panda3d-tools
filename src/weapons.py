from panda3d.core import NodePath
from panda3d.core import CollisionSphere, CollisionBox, CollisionNode, Point3

from .common import get_dimensions_from_egg

class Bullet(NodePath):
    def __init__(self, base, scale, shipPos, vel, bullet_model):
        super().__init__("bullet")
        self.model = base.loader.loadModel(bullet_model)
        self.model.reparentTo(self)
        self.scale = scale
        self.setPos(shipPos)
        self.setScale(self.scale)
        self.velocity = vel
        self.setTwoSided(True)
        size_x, size_y = get_dimensions_from_egg(bullet_model)
        # print(size_x, size_y)
        
        self.HP = 1
        # Collision stuff
        self.collBox = CollisionBox(Point3(-size_x*self.scale, -0.5, -size_y*self.scale), Point3(size_x*self.scale, 0.5, size_y*self.scale))
        self.collNode = CollisionNode('bullet')
        self.collNode.addSolid(self.collBox)
        self.collNodePath = self.attachNewNode(self.collNode)
        self.collNodePath.show()   # temporary show for debugging
        
    def update_pos(self, dt):
        # print(f"Bullet pos: {self.getPos()}")
        self.setZ(self.getZ() + self.velocity * dt)
        