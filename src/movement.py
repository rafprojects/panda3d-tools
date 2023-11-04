import math
import random

class MovementBehavior:
    def move(self, enemy, dt):
        raise NotImplementedError

class StraightDown(MovementBehavior):
    def move(self, enemy, dt):
        enemy.setZ(enemy.getZ() - enemy.velocity * dt)

class LinearMovement(MovementBehavior):
    def move(self, enemy, dt):
        enemy.setY(enemy.getY() - enemy.velocity * dt)

class SinusoidalMovement(MovementBehavior):
    def move(self, enemy, dt):
        x = enemy.getX() + math.sin(dt * enemy.frequency) * enemy.amplitude
        y = enemy.getY() - enemy.velocity * dt
        enemy.setPos(x, y, enemy.getZ())

class CircularMovement(MovementBehavior):
    def move(self, enemy, dt):
        angle = (dt * enemy.velocity) % 360
        x = math.cos(math.radians(angle)) * enemy.radius
        y = math.sin(math.radians(angle)) * enemy.radius
        enemy.setPos(x, y, enemy.getZ())

class RandomMovement(MovementBehavior):
    def move(self, enemy, dt):
        if dt % enemy.change_interval == 0:
            enemy.speed = random.uniform(-enemy.max_speed, enemy.max_speed)
            enemy.direction = random.uniform(-enemy.max_direction, enemy.max_direction)

        x = enemy.getX() + enemy.direction * dt
        y = enemy.getY() + enemy.velocity * dt
        enemy.setPos(x, y, enemy.getZ())
