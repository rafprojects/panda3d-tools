from panda3d.core import NodePath


class Bullet(NodePath):
    def __init__(self, base, scale, shipPos, vel, bullet_model):
        super().__init__("bullet")
        self.model = base.loader.loadModel(bullet_model)
        self.model.reparentTo(self)
        self.setPos(shipPos)
        self.setScale(scale)
        self.velocity = vel
        self.setTwoSided(True)
        
    def update_pos(self, dt):
        # print(f"Bullet pos: {self.getPos()}")
        self.setZ(self.getZ() + self.velocity * dt)