from panda3d.core import Texture, CardMaker


class SpriteLoad():
    '''SpriteLoad is a class that loads a sprite from a texture and attaches it to the scene graph.\n The sprite optionally can enable pixel art scaling to avoid blurring.\nSprite start position and scale can be set, and the scale is multiplied by 0.03 to adjust to the field size.\n '''
    def __init__(self, base, texture_path, pos=(0, 0, 0), scale=1.0, parent=None, pixel_art_scaling=True):

        # self.idle_textures = textureGetter(base_path, f"{type}/idle")
        self.base = base # pull in the base object to be able to act on it from within a separate class
        self.texture = base.loader.loadTexture(texture_path)
        if pixel_art_scaling: # if we want to use nearest neighbor scaling to avoid blurring
            self.texture.setMagfilter(Texture.FT_nearest)
            self.texture.setMinfilter(Texture.FT_nearest)

        # this block of code creates a sprite from the texture and sets it to the correct size matching the texture
        card_maker = CardMaker("sprite")
        tex_w = self.texture.getXSize()
        tex_h = self.texture.getYSize()
        card_maker.setFrame(-tex_w / 2, tex_w / 2, -tex_h / 2, tex_h / 2)

        # this block attaches the card to the scene graph and sets it to the correct position with the correct scale factor for the visual
        self.sprite = base.aspect2d.attachNewNode(card_maker.generate())
        if parent:
            self.sprite.reparentTo(parent)
        self.sprite.setTexture(self.texture)
        self.sprite.setTransparency(True)
        self.sprite.setPos(pos)
        self.sprite.setScale(scale * 0.03)
