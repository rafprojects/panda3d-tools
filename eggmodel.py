class Eggmodel():
    """Basic model class that loads an egg model and attaches it to the scene graph."""
    def __init__(self, base, model_file, pos, scale, parent):
        self.base = base
        self.model = self.base.loader.loadModel(model_file)
        if pos:
            self.model.setPos(pos)
        self.model.setScale(scale)
        
        self.model.reparentTo(self.base.render2d)
