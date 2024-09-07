import math
import random
from .movement_eases import quadratic_ease_io, cubic_ease_in_out, exponential_ease_in_out, back_ease_in_out, bounce_inverted_ease_in_out, hermite_ease, exponential_back_ease_in_out, linear_ease


class MovementBehavior:
    def move(self, enemy, dt, player):
        raise NotImplementedError

class StraightDown(MovementBehavior):
    def move(self, enemy, dt, player):
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
    
    def move(self, enemy, dt, player):
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
    
    def move(self, enemy, dt, player):
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

    def move(self, enemy, dt, player):
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

class FigureEightMovement(MovementBehavior):
    def __init__(self, amplitude, frequency, speed=1.0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.elapsed_time = 0.0
        self.speed = speed

    def move(self, enemy, dt, player):
        self.elapsed_time += dt * self.speed
        t = self.elapsed_time * self.frequency
        
        # Figure-eight movement based on Lissajous curves
        new_x = self.amplitude * math.sin(t)
        new_z = self.amplitude * math.sin(2 * t)
        
        # Update enemy position
        enemy.setX(new_x)
        enemy.setZ(new_z)
        
        # Debugging output
        print(f"Figure-Eight Movement: X = {new_x}, Z = {new_z}")

class OutwardExpandingSpiralMovement(MovementBehavior):
    def __init__(self, initial_radius, spiral_rate, rotation_speed, speed=1.0):
        self.radius = initial_radius
        self.spiral_rate = spiral_rate
        self.rotation_speed = rotation_speed
        self.elapsed_time = 0.0
        self.speed = speed

    def move(self, enemy, dt, player):
        self.elapsed_time += dt * self.speed
        self.radius += self.spiral_rate * dt
        angle = self.elapsed_time * self.rotation_speed
        
        new_x = self.radius * math.cos(angle)
        new_z = self.radius * math.sin(angle)
        
        # Update enemy position
        enemy.setX(new_x)
        enemy.setZ(new_z)
        
        # Debugging output
        print(f"Spiral Movement: X = {new_x}, Z = {new_z}, Radius = {self.radius}")

class PendulumMovement(MovementBehavior):
    def __init__(self, amplitude, frequency, speed=1.0, damping=0.0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.elapsed_time = 0.0
        self.speed = speed
        self.damping = damping  # Damping to simulate energy loss over time

    def move(self, enemy, dt, player):
        self.elapsed_time += dt * self.speed
        t = self.elapsed_time * self.frequency
        amplitude = self.amplitude * math.exp(-self.damping * self.elapsed_time)  # Apply damping
        
        new_x = amplitude * math.sin(t)
        new_z = enemy.getZ()  # Keep Z constant or allow for slight vertical oscillation
        
        # Update enemy position
        enemy.setX(new_x)
        enemy.setZ(new_z)
        
        # Debugging output
        print(f"Pendulum Movement: X = {new_x}, Z = {new_z}, Amplitude = {amplitude}")

class ZigZagMovement(MovementBehavior):
    def __init__(self, amplitude, frequency, speed=1.0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.elapsed_time = 0.0
        self.speed = speed

    def move(self, enemy, dt, player):
        self.elapsed_time += dt * self.speed
        t = self.elapsed_time * self.frequency
        
        # Zig-zag pattern
        new_x = self.amplitude * math.sin(t)
        new_z = enemy.getZ() - dt * self.speed  # Vertical descent
        
        # Update enemy position
        enemy.setX(new_x)
        enemy.setZ(new_z)
        
        # Debugging output
        print(f"Zig-Zag Movement: X = {new_x}, Z = {new_z}")

class RandomWalkMovement(MovementBehavior):
    def __init__(self, step_size, speed=1.0):
        self.step_size = step_size
        self.speed = speed

    def move(self, enemy, dt, player):
        step_x = (random.random() - 0.5) * self.step_size
        step_z = (random.random() - 0.5) * self.step_size
        
        new_x = enemy.getX() + step_x * dt * self.speed
        new_z = enemy.getZ() + step_z * dt * self.speed
        
        # Update enemy position
        enemy.setX(new_x)
        enemy.setZ(new_z)
        
        # Debugging output
        print(f"Random Walk Movement: X = {new_x}, Z = {new_z}")

class ArcMovement(MovementBehavior):
    def __init__(self, radius, angle_range, speed=1.0):
        self.radius = radius
        self.angle_range = angle_range  # Full angle of the arc (e.g., 180°)
        self.elapsed_time = 0.0
        self.speed = speed

    def move(self, enemy, dt, player):
        self.elapsed_time += dt * self.speed
        t = self.elapsed_time % 1.0
        
        # Calculate angle for the arc movement
        angle = t * math.radians(self.angle_range)
        new_x = self.radius * math.cos(angle)
        new_z = self.radius * math.sin(angle)
        
        # Update enemy position
        enemy.setX(new_x)
        enemy.setZ(new_z)
        
        # Debugging output
        print(f"Arc Movement: X = {new_x}, Z = {new_z}")

class TeleportMovement(MovementBehavior):
    def __init__(self, teleport_interval, area_bounds, speed=1.0):
        self.teleport_interval = teleport_interval
        self.area_bounds = area_bounds  # [(xmin, xmax), (zmin, zmax)]
        self.elapsed_time = 0.0
        self.speed = speed

    def move(self, enemy, dt, player):
        self.elapsed_time += dt
        
        if self.elapsed_time >= self.teleport_interval:
            new_x = random.uniform(self.area_bounds[0][0], self.area_bounds[0][1])
            new_z = random.uniform(self.area_bounds[1][0], self.area_bounds[1][1])
            
            # Update enemy position
            enemy.setX(new_x)
            enemy.setZ(new_z)
            
            # Reset timer
            self.elapsed_time = 0.0
            
            # Debugging output
            print(f"Teleport Movement: X = {new_x}, Z = {new_z}")

class RandomMovement(MovementBehavior):
    def move(self, enemy, dt, player):
        print(dt)
        if dt % enemy.change_interval != 0:
            enemy.speed = random.uniform(-enemy.max_speed, enemy.max_speed)
            enemy.direction = random.uniform(-enemy.max_direction, enemy.max_direction)

        x = enemy.getX() + enemy.direction * dt
        y = enemy.getY() + enemy.velocity * dt
        enemy.setPos(x, y, enemy.getZ())

class EvadingMovement(MovementBehavior):
    def __init__(self, evade_distance, evade_speed, wander_speed=0.5):
        self.evade_distance = evade_distance  # Minimum distance to maintain from the player
        self.evade_speed = evade_speed        # Speed for evading the player
        self.wander_speed = wander_speed      # Speed for random wandering

    def move(self, enemy, dt, player):
        # Calculate the direction vector towards the player
        direction_x = player.model.getX() - enemy.model.getX()
        direction_z = player.model.getZ() - enemy.model.getZ()
        distance = math.sqrt(direction_x**2 + direction_z**2)
        print("DISTANCE: ", distance)
        
        # Handle the case where distance is zero to avoid division by zero
        if distance == 0:
            # Apply a small random movement to avoid staying in place
            direction_x = random.uniform(-1, 1)
            direction_z = random.uniform(-1, 1)
            distance = 1  # Set to 1 to normalize and prevent zero division
        
        # If the player is too close, move away quickly
        if distance < self.evade_distance:
            print("CLOSE to enemy ", enemy.id)
            # Normalize the direction vector
            direction_x /= distance
            direction_z /= distance
            
            # Move away from the player (use evade_speed)
            new_x = enemy.getX() - direction_x * self.evade_speed * dt
            new_z = enemy.getZ() - direction_z * self.evade_speed * dt
            
            # Update enemy position
            enemy.setX(new_x)
            enemy.setZ(new_z)
            
            # Debugging output for evading
            # print(f"Evading: Moving away from player at X = {player.getX()}, Z = {player.getZ()} "
                #   f"to new X = {new_x}, Z = {new_z}")
        else:
            # Random wandering behavior when not evading
            random_wander_x = random.uniform(-1, 1)
            random_wander_z = random.uniform(-1, 1)
            
            # Move at the wandering speed
            new_x = enemy.getX() + random_wander_x * self.wander_speed * dt
            new_z = enemy.getZ() + random_wander_z * self.wander_speed * dt
            
            # Update enemy position
            enemy.setX(new_x)
            enemy.setZ(new_z)
            
            # Debugging output for wandering
            # print(f"Wandering: Enemy wandering at X = {new_x}, Z = {new_z}")


class StraightEvadingMovement(MovementBehavior):
    def __init__(self, evade_distance, evade_speed, wander_speed=0.5, evade_duration=2.0, max_evade_distance=50.0):
        self.evade_distance = evade_distance      # Minimum distance to maintain from the player
        self.evade_speed = evade_speed            # Speed for evading the player
        self.wander_speed = wander_speed          # Speed for random wandering
        self.evade_duration = evade_duration      # Duration to stay in evade mode
        self.max_evade_distance = max_evade_distance  # Maximum distance the enemy can move while evading
        self.evading = False                      # Track if currently evading
        self.evade_timer = 0.0                    # Timer to track evading time
        self.evade_direction_x = 0.0              # Store the direction to evade
        self.evade_direction_z = 0.0              # Store the direction to evade
        self.start_x = 0.0                        # Store the initial X position when evading starts
        self.start_z = 0.0                        # Store the initial Z position when evading starts

    def move(self, enemy, dt, player):
        if self.evading:
            # Continue evading for a certain duration or until max distance is reached
            self.evade_timer += dt

            # Calculate the current distance from the starting evade point
            current_x = enemy.model.getX()
            current_z = enemy.model.getZ()
            distance_moved = math.sqrt((current_x - self.start_x) ** 2 + (current_z - self.start_z) ** 2)

            if self.evade_timer >= self.evade_duration or distance_moved >= self.max_evade_distance:
                # Stop evading and go back to wandering mode
                self.evading = False
                self.evade_timer = 0.0
                print(f"Enemy {enemy.id} finished evading, moved {distance_moved:.2f} units, returning to wander mode.")
            else:
                # Move in the stored evade direction
                new_x = enemy.model.getX() + self.evade_direction_x * self.evade_speed * dt
                new_z = enemy.model.getZ() + self.evade_direction_z * self.evade_speed * dt
                enemy.model.setX(new_x)
                enemy.model.setZ(new_z)
                print(f"EVADING: Enemy {enemy.id} continuing evasion to X = {new_x}, Z = {new_z}")
            return  # Skip the rest of the logic while evading

        # Calculate the direction vector towards the player
        direction_x = player.model.getX() - enemy.model.getX()
        direction_z = player.model.getZ() - enemy.model.getZ()
        distance = math.sqrt(direction_x**2 + direction_z**2)
        
        # Handle the case where distance is zero to avoid division by zero
        if distance == 0:
            direction_x = random.uniform(-1, 1)
            direction_z = random.uniform(-1, 1)
            distance = 1  # Set to 1 to normalize and prevent zero division
        
        print(f"Distance from player to enemy {enemy.id}: {distance:.2f}")

        if distance < self.evade_distance:
            # If player is too close, start evading
            self.evading = True
            self.evade_timer = 0.0  # Reset the evade timer
            self.start_x = enemy.model.getX()  # Store the initial position for max distance tracking
            self.start_z = enemy.model.getZ()

            # Normalize the direction vector and store it
            direction_x /= distance
            direction_z /= distance
            self.evade_direction_x = -direction_x  # Move in the opposite direction of the player
            self.evade_direction_z = -direction_z

            # Move away from the player immediately
            new_x = enemy.model.getX() + self.evade_direction_x * self.evade_speed * dt
            new_z = enemy.model.getZ() + self.evade_direction_z * self.evade_speed * dt
            enemy.model.setX(new_x)
            enemy.model.setZ(new_z)
            print(f"EVADING: Enemy {enemy.id} starts evading to X = {new_x}, Z = {new_z}")
        else:
            # Apply random wandering behavior when not evading
            random_wander_x = random.uniform(-1, 1)
            random_wander_z = random.uniform(-1, 1)
            
            new_x = enemy.model.getX() + random_wander_x * self.wander_speed * dt
            new_z = enemy.model.getZ() + random_wander_z * self.wander_speed * dt
            enemy.model.setX(new_x)
            enemy.model.setZ(new_z)
            print(f"WANDERING: Enemy {enemy.id} wandering at X = {new_x}, Z = {new_z}")


class WaitAndBeelineMovement(MovementBehavior):
    def __init__(self, wait_time, beeline_speed, max_beeline_distance=None, follow_through=False):
        """
        :param wait_time: Time the enemy waits before starting the beeline.
        :param beeline_speed: Speed at which the enemy moves towards the player.
        :param max_beeline_distance: Optional. The distance from the player to stop beelining.
                                     If None, no distance-based stop is enforced.
        :param follow_through: If True, the enemy beelines towards the player's initial position
                               and continues in that direction indefinitely without stopping.
                               If False, the enemy follows the player and stops based on distance.
        """
        self.wait_time = wait_time           # Time to wait before starting the beeline movement
        self.beeline_speed = beeline_speed   # Speed when moving towards the player
        self.max_beeline_distance = max_beeline_distance  # Optional: Stop if enemy gets too close to player
        self.follow_through = follow_through  # If True, lock the initial player position and continue
        self.waiting = True                  # Whether the enemy is in "wait mode"
        self.wait_timer = 0.0                # Timer to track waiting time
        self.beelining = False               # Track if the enemy is beelining
        self.direction_x = None              # Store the movement vector X
        self.direction_z = None              # Store the movement vector Z

    def move(self, enemy, dt, player):
        if self.waiting:
            # Increment the wait timer
            self.wait_timer += dt
            # print(f"WAITING: Enemy {enemy.id} is waiting... Time: {self.wait_timer:.2f}/{self.wait_time}")
            
            # Check if the waiting period is over
            if self.wait_timer >= self.wait_time:
                # Stop waiting and start beelining
                self.waiting = False
                self.beelining = True

                # Calculate the vector from the enemy to the player (at the moment of beelining)
                player_x = player.model.getX()
                player_z = player.model.getZ()

                direction_x = player_x - enemy.model.getX()
                direction_z = player_z - enemy.model.getZ()
                distance = math.sqrt(direction_x**2 + direction_z**2)

                # Handle ZeroDivisionError when distance is zero (enemy is at the player's position)
                if distance == 0:
                    direction_x = random.uniform(-1, 1)
                    direction_z = random.uniform(-1, 1)
                    distance = 1  # Set distance to a non-zero value to avoid division by zero

                # Normalize the direction vector
                self.direction_x = direction_x / distance
                self.direction_z = direction_z / distance

                # print(f"Enemy {enemy.id} locks onto player position, calculating vector: X = {self.direction_x}, Z = {self.direction_z}")
                # print(f"Enemy {enemy.id} finished waiting, starting beeline!")
            else:
                return  # Stay in wait mode until the time elapses

        if self.beelining:
            # If follow_through is True, use the stored vector and continue past the player
            if self.follow_through:
                # Continue in the same direction indefinitely using the stored vector
                new_x = enemy.model.getX() + self.direction_x * self.beeline_speed * dt
                new_z = enemy.model.getZ() + self.direction_z * self.beeline_speed * dt
                enemy.model.setX(new_x)
                enemy.model.setZ(new_z)
                # print(f"FOLLOW THROUGH: Enemy {enemy.id} continues moving past target in vector direction to X = {new_x}, Z = {new_z}")
            else:
                # Follow the player (recalculate the direction to the player continuously)
                player_x = player.model.getX()
                player_z = player.model.getZ()

                direction_x = player_x - enemy.model.getX()
                direction_z = player_z - enemy.model.getZ()
                distance = math.sqrt(direction_x**2 + direction_z**2)

                # Handle ZeroDivisionError when distance is zero (enemy is at the player's position)
                if distance == 0:
                    direction_x = random.uniform(-1, 1)
                    direction_z = random.uniform(-1, 1)
                    distance = 1  # Set distance to a non-zero value to avoid division by zero

                # Normalize the direction vector
                direction_x /= distance
                direction_z /= distance

                if self.max_beeline_distance and distance <= self.max_beeline_distance:
                    # print(f"Enemy {enemy.id} reached max beeline distance. Stopping.")
                    self.beelining = False  # Stop beelining
                    return

                # Move towards the player's current position
                new_x = enemy.model.getX() + direction_x * self.beeline_speed * dt
                new_z = enemy.model.getZ() + direction_z * self.beeline_speed * dt
                enemy.model.setX(new_x)
                enemy.model.setZ(new_z)
                # print(f"BEELINING: Enemy {enemy.id} moving towards player at X = {player_x}, Z = {player_z} to new X = {new_x}, Z = {new_z}")
