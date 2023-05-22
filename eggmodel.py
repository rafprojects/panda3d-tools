from panda3d.core import NodePath


class Eggmodel(NodePath):
    """Basic model class that loads an egg model and attaches it to the scene graph."""
    def __init__(self, pos, scale, base, model_file):
        super().__init__("eggmodel")
        self.base = base
        self.model = self.base.loader.loadModel(model_file)
        self.model.setPos(pos)
        self.model.setScale(scale)
        self.model.reparentTo(self)
