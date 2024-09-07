from .eggmodel import Eggmodel
from .entity import Entity
from .weapons import Bullet
from .common import make_bounding_box, get_bounding_box_dimensions


class Player(Entity):
    '''Player handles the player character. This includes the sprite, movement, 
    animations, and other player related functions.\n
    The movement speed, scale and position are hardcoded (currently)'''
    def __init__(self, base, charId, entity_type, model_file, cTrav, cHandler, HP=100, pos=(0, 0, 0), scale=0.4, playfield_bounds=None):
        super().__init__(HP, pos, scale, base, model_file, entity_type, cTrav, cHandler)
        self.base = base
        self.charId = charId
        # self.scale_factor = 0.4
        
        # bounds = self.getBounds()
        # print(f"PLAYER BOUNDS: {bounds}")
        # print(f"PLAYER CENTER: {bounds.getCenter()}")
        self.reparentTo(self.base.render)
        # Movement & animation variables
        self.move_speed = 200
        self.moving_keymap = {"left": False, "right": False, "up": False, "down": False}
        self.state = "idle"
        self.animation_time = 0.0
        # Track player position
        self.x = 0
        self.y = 0
        self.playfield_bounds = playfield_bounds
        # Track weapons
        self.bullets = []
        self.bullet_vel = 400
        self.bullet_offset = (0, 0)  # align bullet to ship
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
        
        self.cTrav = cTrav
        self.cHandler = cHandler
        self.last_collision = 0
        # print(get_bounding_box_dimensions(self.player_entity))
        # make_bounding_box(self.player_entity)

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
            
        # Clamp the player position within the playfield boundaries
        self.x = max(self.playfield_bounds['left'], min(self.x, self.playfield_bounds['right']))
        self.y = max(self.playfield_bounds['bottom'], min(self.y, self.playfield_bounds['top']))
        
        self.model.setPos(self.x, 0, self.y)
        return task.cont

    def fire_bullet(self, vel):
        bullet_coords = (self.model.getX() + self.bullet_offset[0],
                         0,
                         self.model.getZ() + self.bullet_offset[1]
                        )
        bullet = Bullet(self.base,
                        0.5,
                        bullet_coords,
                        vel,
                        self.bullet_offset,
                        'assets/sprites/weapons/bullet.egg',
                        cTrav=self.cTrav,
                        cHandler=self.cHandler
                        )
        bullet.reparentTo(self.base.render)
        self.bullets.append(bullet)

    def update_animation(self, task):
        dt = globalClock.getDt()
        self.animation_time += dt
        seq_node = self.model.find('**/+SequenceNode').node()
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
            # print(get_bounding_box_dimensions(self.player_entity))
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
