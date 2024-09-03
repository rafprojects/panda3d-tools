import math
import random
from .movement_eases import quadratic_ease_io, cubic_ease_in_out, exponential_ease_in_out, back_ease_in_out, bounce_inverted_ease_in_out, hermite_ease, exponential_back_ease_in_out, linear_ease


class MovementBehavior:
    def move(self, enemy, dt):
        raise NotImplementedError

class StraightDown(MovementBehavior):
    def move(self, enemy, dt):
        enemy.setZ(enemy.getZ() - enemy.velocity * dt)

class LinearMovement(MovementBehavior):
    def __init__(self, peak_x, ease_in_power=2, ease_out_power=2, speed=1.0, start_x=None):
        self.total_distance = abs(peak_x - start_x) if start_x is not None else 100.0
        self.elapsed_time = 0.0
        self.direction = 1  # 1 for forward, -1 for reverse
        self.peak_x = peak_x
        self.start_x = start_x if start_x is not None else 0.0
        self.current_x = self.start_x
        self.ease_in_power = ease_in_power
        self.ease_out_power = ease_out_power
        self.speed = speed  # Speed factor to control movement speed
    
    def move(self, enemy, dt):
        # Update elapsed time based on speed and direction
        self.elapsed_time += dt * self.speed
        t = self.elapsed_time / self.total_distance
        
        # Clamp t between 0 and 1
        t = max(0, min(t, 1.0))

        # Calculate eased position
        if self.direction == 1:  # Moving towards the peak
            # eased_speed = quadratic_ease_io(t, self.ease_in_power, self.ease_out_power)
            eased_speed = exponential_back_ease_in_out(t)
        else:  # Moving back towards the start
            # eased_speed = quadratic_ease_io(1 - t, self.ease_in_power, self.ease_out_power)
            eased_speed = exponential_back_ease_in_out(1 - t)

        # Update position
        new_x = self.start_x + eased_speed * (self.peak_x - self.start_x)
        enemy.setX(new_x)
        
        # Debugging Output
        print(f"new_x: {new_x}, t: {t}, direction: {self.direction}")

        # Check if the enemy has reached the end of the current movement and needs to reverse
        if t >= 1.0:
            self.direction *= -1  # Reverse direction
            self.elapsed_time = 0.0  # Reset elapsed time for the next movement cycle


class SinusoidalMovement(MovementBehavior):
    def __init__(self, amplitude, frequency, ease_in_power=2, ease_out_power=2, speed=1.0, y_speed=7.0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.elapsed_time = 0.0
        self.ease_in_power = ease_in_power
        self.ease_out_power = ease_out_power
        self.speed = speed  # Speed factor to control movement speed
        self.y_speed = y_speed  # Speed factor to control vertical movement speed
        self.base_x = None  # To store the initial X position of the enemy
    
    def move(self, enemy, dt):
        # Store the base X position once
        if self.base_x is None:
            self.base_x = enemy.getX()
        
        # Update elapsed time based on speed
        self.elapsed_time += dt * self.speed
        
        # Normalize time based on frequency and use easing
        t = (self.elapsed_time * self.frequency) % 1.0  # Normalized time for one full cycle (0 to 1)
        eased_t = quadratic_ease_io(t, self.ease_in_power, self.ease_out_power)
        
        # Calculate new X position using eased time
        x_offset = math.sin(eased_t * 2 * math.pi) * self.amplitude
        new_x = self.base_x + x_offset
        
        # Smoothly update the enemy's X position
        enemy.setX(new_x)
        
        # Update Y position for downward movement
        new_y = enemy.getZ() - self.y_speed * dt
        enemy.setZ(new_y)
        
        # Debugging Output
        # print(f"x_offset: {x_offset}, t: {t}, eased_t: {eased_t}, new_x: {new_x}")


class CircularMovement(MovementBehavior):
    def __init__(self, radius, frequency, descent_speed=0.0, ease_in_power=2, ease_out_power=2, speed=1.0):
        self.radius = radius
        self.frequency = frequency  # Number of full circles per second
        self.descent_speed = descent_speed  # Units per second
        self.speed = speed  # Speed multiplier
        self.elapsed_time = 0.0
        self.ease_in_power = ease_in_power
        self.ease_out_power = ease_out_power
        self.base_x = None  # Center X position of the circular path
        self.base_z = None  # Center Z position of the circular path

    def move(self, enemy, dt):
        """
        Updates the enemy's position to follow a circular path with optional vertical descent.

        :param enemy: The enemy object to move.
        :param dt: Delta time since the last frame (in seconds).
        """
        # Initialize base positions on the first frame
        if self.base_x is None or self.base_z is None:
            self.base_x = enemy.getX()
            self.base_z = enemy.getZ()

        # Update elapsed time with speed multiplier
        self.elapsed_time += dt * self.speed

        # Calculate normalized time `t` based on frequency (cycles per second)
        t = (self.elapsed_time * self.frequency) % 1.0  # Range: [0.0, 1.0)

        # Apply easing to the normalized time
        eased_t = linear_ease(t)

        # Calculate the current angle in radians for the circular path
        angle = eased_t * 2 * math.pi  # Full circle: 0 to 2π radians

        # Compute the new X and Z positions based on the circular path
        new_x = self.base_x + self.radius * math.cos(angle)
        new_z = self.base_z + self.radius * math.sin(angle)

        # Apply vertical descent by updating the base Z position
        if self.descent_speed != 0.0:
            self.base_z -= self.descent_speed * dt  # Descend downward over time

            # Update the new Z position relative to the descending base_z
            new_z = self.base_z + self.radius * math.sin(angle)

        # Update the enemy's position in the X-Z plane
        enemy.setX(new_x)
        enemy.setZ(new_z)

        # Ensure Y-axis remains constant (no movement along Y)
        # If necessary, explicitly set Y to its initial value
        # For example:
        # enemy.setY(initial_y)

        # Debugging Output
        # print(f"Angle: {math.degrees(angle):.2f}°, t: {t:.2f}, eased_t: {eased_t:.2f}, "
        #       f"New X: {new_x:.2f}, New Z: {new_z:.2f}, Base Z: {self.base_z:.2f}")



class RandomMovement(MovementBehavior):
    def move(self, enemy, dt):
        print(dt)
        if dt % enemy.change_interval != 0:
            enemy.speed = random.uniform(-enemy.max_speed, enemy.max_speed)
            enemy.direction = random.uniform(-enemy.max_direction, enemy.max_direction)

        x = enemy.getX() + enemy.direction * dt
        y = enemy.getY() + enemy.velocity * dt
        enemy.setPos(x, y, enemy.getZ())
