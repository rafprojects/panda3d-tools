from panda3d.core import NodePath


class Eggmodel(NodePath):
    """Basic model class that loads an egg model and attaches it to the scene graph."""
    def __init__(self, pos, scale, base, model_file):
        super().__init__("eggmodel")
        self.base = base
        try:
            self.model = self.base.loader.loadModel(model_file)
        except Exception as e:
            print(f"Error loading egg model: {model_file}\n{e}")
        self.model.setScale(scale)
        self.model.setPos(pos)
        self.model.reparentTo(self)
        