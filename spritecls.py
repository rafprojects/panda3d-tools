from panda3d.core import Texture, CardMaker


class SpriteLoad():
    def __init__(self, base, texture_path, pos=(0, 0, 0), scale=1.0, parent=None, pixel_art_scaling=True):
        # self.idle_textures = textureGetter(base_path, f"{type}/idle")
        self.base = base
        self.texture = base.loader.loadTexture(texture_path)
        if pixel_art_scaling:
            self.texture.setMagfilter(Texture.FT_nearest)
            self.texture.setMinfilter(Texture.FT_nearest)

        card_maker = CardMaker("sprite")
        tex_w = self.texture.getXSize()
        tex_h = self.texture.getYSize()
        card_maker.setFrame(-tex_w / 2, tex_w / 2, -tex_h / 2, tex_h / 2)

        self.sprite = base.aspect2d.attachNewNode(card_maker.generate())
        if parent:
            self.sprite.reparentTo(parent)
        self.sprite.setTexture(self.texture)
        self.sprite.setTransparency(True)
        self.sprite.setPos(pos)
        self.sprite.setScale(scale * 0.03)
