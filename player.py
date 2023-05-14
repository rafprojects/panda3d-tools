from panda3d.core import Vec3
from sequences import PLAYER_SEQUENCES, ANIM_FRAME_RATE
from eggmodel import Eggmodel


class Player():
    '''Player handles the player character. This includes the sprite, movement, 
    animations, and other player related functions.\n
    The movement speed, scale and position are hardcoded (currently)'''
    def __init__(self, base, charId, player_model_file):
        self.base = base
        self.charId = charId
        self.player_anim_states = ["idle", "left", "right"]
        self.player_sprite = Eggmodel(base=base, 
                                      model_file=player_model_file, 
                                      pos=(0, 0, 0), 
                                      scale=0.8, 
                                      parent=self.base.render)
        self.move_speed = 2
        self.moving = {"left": False, "right": False, "up": False, "down": False}
        self.animation_time = 0.0
        self.anim_frame_rate = ANIM_FRAME_RATE
        self.current_frame = 0
        self.player_sequences = PLAYER_SEQUENCES
        # Keypress event handlers
        self.base.accept("arrow_left", self.set_move_direction, ["left", True])
        self.base.accept("arrow_left-up", self.set_move_direction, ["left", False])
        self.base.accept("arrow_right", self.set_move_direction, ["right", True])
        self.base.accept("arrow_right-up", self.set_move_direction, ["right", False])
        self.base.accept("arrow_up", self.set_move_direction, ["up", True])
        self.base.accept("arrow_up-up", self.set_move_direction, ["up", False])
        self.base.accept("arrow_down", self.set_move_direction, ["down", True])
        self.base.accept("arrow_down-up", self.set_move_direction, ["down", False])
 
    def set_move_direction(self, direction, state):
        '''Sets the direction of movement to the state (True or False)'''
        self.moving[direction] = state
    
    def update(self, task):
        '''Updates the player sprite position based on the movement vector'''
        move_vec = Vec3(0, 0, 0)  # create a vector to store the movement in

        if self.moving['left']:
            move_vec.x -= self.move_speed * globalClock.getDt()
        if self.moving['right']:
            move_vec.x += self.move_speed * globalClock.getDt()
        if self.moving['up']:
            move_vec.z += self.move_speed * globalClock.getDt()
        if self.moving['down']:
            move_vec.z -= self.move_speed * globalClock.getDt()

        self.player_sprite.model.setPos(self.player_sprite.model.getPos() + move_vec)

        return task.cont  # return task.cont to keep the task running

    # def update_animation(self, dt):
    #     """Updates the player sprite animation based on the movement vector"""
    #     self.animation_time += dt
    #     if self.animation_time >= self.anim_frame_rate:
    #         self.animation_time -= self.anim_frame_rate
    #         if self.moving['left']:
    #             # print(f"LEFT_CURRENT_FRAME_BEFORE_CALC: {self.current_frame}")
    #             if self.current_frame != self.frame_counts["left"] -1:
    #                 self.current_frame = (self.current_frame + 1) % self.frame_counts["left"]
    #             self.player_sprite.sprite.setTexture(self.player_textures[("left", self.current_frame)])
    #         elif self.moving['right']:
    #             # print(f"RIGHT_CURRENT_FRAME_BEFORE_CALC: {self.current_frame}")
    #             if self.current_frame != self.frame_counts["right"] -1:
    #                 self.current_frame = (self.current_frame + 1) % self.frame_counts["right"]
    #             self.player_sprite.sprite.setTexture(self.player_textures[("right", self.current_frame)])
    #         else:
    #             self.current_frame = (self.current_frame + 1) % self.frame_counts["idle"]
    #             self.player_sprite.sprite.setTexture(self.player_textures[("idle", self.current_frame)])

    # def update_animation_task(self, task):
    #     dt = globalClock.getDt()
    #     self.update_animation(dt)
    #     return task.cont
