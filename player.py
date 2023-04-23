from panda3d.core import Vec3
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence

from spritecls import SpriteLoad
from helpers import textureLoader
from sequences import PLAYER_SEQUENCES, ANIM_FRAME_RATE


class Player():
    '''Player handles the player character. This includes the sprite, movement, animations, and other player related functions.\nThe sprite texture, movement speed, scale and position are hardcoded (currently)'''
    def __init__(self, base, charId):
        self.base = base
        self.charId = charId
        self.player_anim_states = ["idle", "left", "right"]
        self.player_textures = textureLoader(self.base, self.player_anim_states, "output", "ship")
        self.player_sprite = SpriteLoad(texture_obj=self.player_textures[("idle", 0)], 
                                        base=base, 
                                        pos=(0, 0, 0), 
                                        scale=0.2,
                                        pixel_art_scaling=True)
        self.move_speed = 2
       
        self.moving = {"left": False, "right": False, "up": False, "down": False}
        
        # Keypress event handlers
        self.base.accept("arrow_left", self.set_move_direction, ["left", True])
        self.base.accept("arrow_left-up", self.set_move_direction, ["left", False])
        self.base.accept("arrow_right", self.set_move_direction, ["right", True])
        self.base.accept("arrow_right-up", self.set_move_direction, ["right", False])
        self.base.accept("arrow_up", self.set_move_direction, ["up", True])
        self.base.accept("arrow_up-up", self.set_move_direction, ["up", False])
        self.base.accept("arrow_down", self.set_move_direction, ["down", True])
        self.base.accept("arrow_down-up", self.set_move_direction, ["down", False])
        
        # self.player_sequences = Sequence(
        #     self.player_textures[("idle", 0)],
        #     self.player_textures[("idle", 1)],
        #     name="idle",
        # )
        # self.spliced_sequences.loop("idle")
        
    def set_move_direction(self, direction, state):
        '''Sets the direction of movement to the state (True or False)'''
        self.moving[direction] = state

    def update(self, task):
        '''Updates the player sprite position based on the movement vector'''
        move_vec = Vec3(0, 0, 0) # create a vector to store the movement in

        if self.moving['left']:
            move_vec.x -= self.move_speed * globalClock.getDt()
        if self.moving['right']:
            move_vec.x += self.move_speed * globalClock.getDt()
        if self.moving['up']:
            move_vec.z += self.move_speed * globalClock.getDt()
        if self.moving['down']:
            move_vec.z -= self.move_speed * globalClock.getDt()

        self.player_sprite.sprite.setPos(self.player_sprite.sprite.getPos() + move_vec)
        
        return task.cont # return task.cont to keep the task running
