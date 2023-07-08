from panda3d.core import NodePath
from panda3d.core import CollisionSphere, CollisionBox, CollisionNode, Point3

from .common import get_dimensions_from_egg, get_box_dimensions

class Bullet(NodePath):
    def __init__(self, base, scale, shipPos, vel, bullet_offset, bullet_model):
        super().__init__("bullet")
        self.model = base.loader.loadModel(bullet_model)
        self.model.reparentTo(self)
        bounds = self.model.getBounds()
        print(f"CENTER: {bounds.getCenter()}")
        self.scale = scale
        self.setScale(self.scale)
        self.setPos(shipPos)
        self.velocity = vel
        self.setTwoSided(True)
        # size_x, size_y = get_dimensions_from_egg(bullet_model, half=True)
        pointA, pointB = get_box_dimensions(eggfile=bullet_model, scale_factor=self.scale, offsetT=None)
        print(pointA, pointB)
        
        self.HP = 1
        # Collision stuff
        self.collBox = CollisionBox(
            Point3(pointA[0], pointA[1], pointA[2]), 
            Point3(pointB[0], pointB[1], pointB[2])
        )
        self.collNode = CollisionNode('bullet')
        self.collNode.addSolid(self.collBox)
        self.collNodePath = self.attachNewNode(self.collNode)
        self.collNodePath.show()   # temporary show for debugging
        
    def update_pos(self, dt):
        # print(f"Bullet pos: {self.getPos()}")
        self.setZ(self.getZ() + self.velocity * dt)
        