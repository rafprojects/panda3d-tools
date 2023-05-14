# Code Details
This is a place tracking/reasoning the development choices and methodology used in this project.

The Panda3d ShowBase is loaded in a class separate from the main Game class that runs the game.
The ShowBase "base" object is shared by other class objects by passing it through them as a parameter.

The "sprites" for player (and eventually all other active sprite characters and such) are actually 
sequenced textures on panda3d models, which gives it a cleaner look than the texture swapping technique
previously used.  This is apparently better compatable with panda3d, since it's technically a "3d" model,
and the sprite animation is doing the same thing automatically "on top" of the model.