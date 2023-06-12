# Code Details

### General Stuff
This is a place tracking/reasoning the development choices and methodology used in this project.

The Panda3d ShowBase is loaded in a class separate from the main Game class that runs the game.
The ShowBase "base" object is shared by other class objects by passing it through them as a parameter.
I'm not sure if this has any particular benefit yet, but was a good personal exercise of class inheritance

The "sprites" for player (and eventually all other active sprite characters and such) are actually sequenced textures on panda3d models (constructed by the Eggmodel class), which gives it a cleaner look than the texture swapping technique previously used.  
This is apparently better compatable with panda3d, since it's technically a "3d" model,
and the sprite animation is doing the same thing automatically "on top" of the model.

### BULLETS

The bullet class is a child of the Panda3d NodePath. The "sprite" is a bullet egg model that is reparented to the class object itself, so that the entire NodePath/sprite can be parented to render as one.  This is the only way to get the bullet to visually move across the screen.

> NOTE: We might be able to make this much more flexible by setting the bullet model as a parameter rather than hardcode, and the scale/vel in a reference table for weapons.

The class has flexible scale and velocity, the latter of which is a Player class variable that is read by the spacekey listener as the parameter for the Player fire_bullet function.
This flexibility is intended for future powerup implementations.

The Bullet class' update_pos method is called within the Player update_bullets method in order to update each bullet object's position.  The update_bullets method is set in the taskMgr.