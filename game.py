from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import DirectButton, DirectFrame
from panda3d.core import CardMaker, Texture  #, SpriteNode
from panda3d.core import loadPrcFile
from spritecls import SpriteLoad
from helpers import textureGetter
# from panda3d.core import *

loadPrcFile("config/conf.prc")

# from panda3d.core import ConfigVariableManager
# ConfigVariableManager.getGlobalPtr().listVariables()

# loadPrcFileData("", "notify-level-glgsg debug")
# class Game(ShowBase):
#     def __init__(self):
#         super().__init__()
#         # self.set_background_color(0.1, 0.1, 0.1, 1)
        
#         # Load the texture image for the player character
#         sprite_texture = self.loader.loadTexture("output/ship/TGA_14.png")
        
#         # Create a rectangular card for the player character
#         cardmaker = CardMaker("card")
#         cardmaker.setFrame(-0.5, 0.5, -0.5, 0.5)  # x, y, z, w
#         # setFrame() sets the size of the card
#         # x and y are the left and right edges
#         # z and w are the bottom and top edges
#         # the units are in pixels
#         card = self.render2d.attachNewNode(cardmaker.generate())
#         card.setTexture(sprite_texture)
        
#         # window_size = self.win.getSize()
#         # sprite_size = sprite_texture.getOrigFileXSize(), sprite_texture.getOrigFileYSize()
        
        # card.setPos((window_size[0] - sprite_size[0]) / 2.0, 0, (window_size[1] - sprite_size[1]) / 2.0)
        
        
        # card.setPos(0, 0, 0)
        
        # card.reparentTo(self.render)
     
        
# class Game(ShowBase):
#     def __init__(self):
#         super().__init__()

#         # Load the texture image
#         tex = self.loader.loadTexture('output/ship/TGA_14.png')

#         # Create a card that will display the texture
#         cm = CardMaker('card')
#         cm.setFrame(-1, 1, -1, 1)
#         card = self.render2d.attachNewNode(cm.generate())
#         card.setTexture(tex)

#         # Center the card on the screen
#         card.setPos(0, 0, 0)

class Base(ShowBase):
    def _init_(self):
        super().__init__()


class Game():
    def __init__(self, base):
        self.base = base        
        # self.player_texture = textureGetter("output", "ship/idle")[0]
        self.player_texture = "output/ship/idle/0.png"
        self.player_sprite = SpriteLoad(texture_path=self.player_texture, 
                                        base=self.base, 
                                        pos=(0, 0, 0), 
                                        scale=0.5, )
                                        # parent=self.base.aspect2d)
        
        self.key_map = {
            "left": False,
            "right": False,
            "up": False,
            "down": False
        }
        
        self.base.accept("arrow_left", self.update_key_map, ["left", True])
        self.base.accept("arrow_left-up", self.update_key_map, ["left", False])
        self.base.accept("arrow_right", self.update_key_map, ["right", True])
        self.base.accept("arrow_right-up", self.update_key_map, ["right", False])
        self.base.accept("arrow_up", self.update_key_map, ["up", True])
        self.base.accept("arrow_up-up", self.update_key_map, ["up", False])
        self.base.accept("arrow_down", self.update_key_map, ["down", True])
        self.base.accept("arrow_down-up", self.update_key_map, ["down", False])
        
        self.base.taskMgr.add(self.move_task, "move_task")
        
    def update_key_map(self, key, value):
        self.key_map[key] = value
        
    def move_task(self, task):
        dt = globalClock.getDt()
        # getDt() returns the time in seconds since the last frame
        speed = 10.0
        
        if self.key_map["left"]:
            self.player_sprite.move(int(-speed * dt), 0)
            print("left")
        if self.key_map["right"]:
            self.player_sprite.move(int(speed * dt), 0)
            print("right")
        if self.key_map["up"]:
            self.player_sprite.move(0, int(speed * dt))
            print("up")
        if self.key_map["down"]:
            self.player_sprite.move(0, int(-speed * dt))
            print("down")
            
        return task.cont  # task.cont tells Panda to continue calling this task
       
        
base = Base()
game = Game(base)
# textures = [game.loader.loadTexture(f"objects/idle/{i}.png") for i in range(0, 2)]


game.base.run()