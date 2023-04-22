from panda3d.core import Vec3

from spritecls import SpriteLoad


class Player():
    def __init__(self, base, charId):
        self.base = base
        self.charId = charId
        self.player_texture = "output/ship/idle/0.png"
        self.player_sprite = SpriteLoad(texture_path=self.player_texture, 
                                        base=base, 
                                        pos=(0, 0, 0), 
                                        scale=0.2,
                                        pixel_art_scaling=True)
        self.move_speed = 2
        self.moving = {"left": False, "right": False, "up": False, "down": False}
        
        base.accept("arrow_left", self.set_move_direction, ["left", True])
        base.accept("arrow_left-up", self.set_move_direction, ["left", False])
        base.accept("arrow_right", self.set_move_direction, ["right", True])
        base.accept("arrow_right-up", self.set_move_direction, ["right", False])
        base.accept("arrow_up", self.set_move_direction, ["up", True])
        base.accept("arrow_up-up", self.set_move_direction, ["up", False])
        base.accept("arrow_down", self.set_move_direction, ["down", True])
        base.accept("arrow_down-up", self.set_move_direction, ["down", False])
        
    def set_move_direction(self, direction, state):
        self.moving[direction] = state

    def update(self, task):
        move_vec = Vec3(0, 0, 0)

        if self.moving['left']:
            move_vec.x -= self.move_speed * globalClock.getDt()
            print("left")
        if self.moving['right']:
            move_vec.x += self.move_speed * globalClock.getDt()
            print("right")
        if self.moving['up']:
            move_vec.z += self.move_speed * globalClock.getDt()
            print("up")
        if self.moving['down']:
            move_vec.z -= self.move_speed * globalClock.getDt()
            print("down")

        self.player_sprite.sprite.setPos(self.player_sprite.sprite.getPos() + move_vec)
        
        return task.cont
