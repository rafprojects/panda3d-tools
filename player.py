from panda3d.core import NodePath
from eggmodel import Eggmodel


class Player():
    '''Player handles the player character. This includes the sprite, movement, 
    animations, and other player related functions.\n
    The movement speed, scale and position are hardcoded (currently)'''
    def __init__(self, base, charId, player_model_file):
        self.base = base
        self.charId = charId
        self.player_sprite = Eggmodel(base=base,  
                                      pos=(0, 0, 0), 
                                      scale=0.4, 
                                      parent=self.base.render,
                                      model_file=player_model_file)
        # Movement & animation variables
        self.move_speed = 200
        self.moving_keymap = {"left": False, "right": False, "up": False, "down": False}
        self.state = "idle"
        self.animation_time = 0.0
        # Track the player's position
        self.x = 0
        self.y = 0
        # Track weapons
        self.bullets = []
        self.bullet_vel = 400
        # Keypress event handlers
        self.base.accept("arrow_left", self.update_moving_keymap, ["left", True])
        self.base.accept("arrow_left-up", self.update_moving_keymap, ["left", False])
        self.base.accept("arrow_right", self.update_moving_keymap, ["right", True])
        self.base.accept("arrow_right-up", self.update_moving_keymap, ["right", False])
        self.base.accept("arrow_up", self.update_moving_keymap, ["up", True])
        self.base.accept("arrow_up-up", self.update_moving_keymap, ["up", False])
        self.base.accept("arrow_down", self.update_moving_keymap, ["down", True])
        self.base.accept("arrow_down-up", self.update_moving_keymap, ["down", False])
        self.base.accept("space", self.fire_bullet, [self.bullet_vel])
        
 
    def update_moving_keymap(self, direction, state):
        '''Sets the direction of movement to the state (True or False)'''
        self.moving_keymap[direction] = state
        if not state and direction in ["left", "right"]:
            # this allows smooth animation transitions when going quickly between left and right
            self.animation_time = 0.0

    def move_ship(self, task):
        '''Updates the player sprite position based on the movement vector'''
        dt = globalClock.getDt()
        if self.moving_keymap['left']:
            self.x -= self.move_speed * dt
        if self.moving_keymap['right']:
            self.x += self.move_speed * dt
        if self.moving_keymap['up']:
            self.y += self.move_speed * dt
        if self.moving_keymap['down']:
            self.y -= self.move_speed * dt
        self.player_sprite.model.setPos(self.x, 0, self.y)
        return task.cont  # return task.cont to keep the task running

    def fire_bullet(self, vel):
        bullet = Bullet(self.base, 1.0, self.player_sprite.model.getPos(), vel)
        print(f"SHOT BULLET AT {bullet.getPos()}")
        bullet.reparentTo(self.base.render)
        self.bullets.append(bullet) 
        print(f"NUM BULLETS: {len(self.bullets)}")

    def update_animation(self, task):
        dt = globalClock.getDt()
        self.animation_time += dt
        seq_node = self.player_sprite.model.find('**/+SequenceNode').node()
        # TODO: This is a bit of a mess, but it works for now.  Clean it up later.
        if self.moving_keymap['left']:
            if self.state != "left":
                seq_node.play(2, 3)
                self.state = "left"
            if self.animation_time >= 1.0 / seq_node.getFrameRate():  # Ensure that it stays on the last frame
                seq_node.pose(3)
        elif self.moving_keymap['right']:
            if self.state != "right":
                seq_node.play(4, 5)
                self.state = "right"
            if self.animation_time >= 1.0 / seq_node.getFrameRate():  # Ensure that it stays on the last frame
                seq_node.pose(5)
        else:
            if self.state != "idle":
                seq_node.loop(True, 0, 1)
                self.state = "idle"
            self.animation_time = 0.0

        return task.cont
    
    def update_bullets(self, task):
        dt = globalClock.getDt()
        for bullet in self.bullets:
            bullet.update_pos(dt)
            if bullet.getZ() > 205:
                bullet.removeNode()
                self.bullets.remove(bullet)
            if len(self.bullets) > 6:
                self.bullets[0].removeNode()
                self.bullets.remove(self.bullets[0])
        return task.cont

class Bullet(NodePath):
    def __init__(self, base, scale, shipPos, vel):
        super().__init__("bullet")
        self.model = base.loader.loadModel('output/weapons/bullet')
        self.model.reparentTo(self)
        self.setPos(shipPos)
        self.setScale(scale)
        self.velocity = vel
        self.setTwoSided(True)
        
    def update_pos(self, dt):
        # print(f"Bullet pos: {self.getPos()}")
        self.setZ(self.getZ() + self.velocity * dt)