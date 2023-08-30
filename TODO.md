### Player
- implement second weapon
  - depend on shot
  - reference file for weapon types/levels/power/cost
  - must fire by second button
- add chip/shot stat
- add inventory (weapons, weap levels, gun, shield)

### Enemy

- ~~random spawn point func~~
- fix periodic spawning
- how to implement enemy movement patterns, and make them unique per enemy
  - modular movement patterns via functions that can be assigned to enemies
- modular spawn pattern functions that can be pulled into spawn generator function
- enemy stats reference file (HP, immunities, body power, shot power, movement patterns)
- enemy shot ability that is optional (the general circle enemy shot)
  - this is separate from enemy special abilities unique to enemy
  - this should be able to be applied to any given enemy
  - shot should have several options to modify speed, reload, power, multi

### Collision

- ~~create collision shape for player~~
- ~~create collision shapes for bullet~~
- ~~create collision shapes for enemy~~
- ~~player VS enemy body~~
- ~~enemy VS player bullet~~
- player VS enemy shot
  - depends on enemy shot ability implementation
- player VS obstacle / environment
- screen edge collision
- enemy health reduce on bullet hit
- enemy destruction on 0 HP
- bullet destruction on hit enemy

### Playfield
- procedural moving starry background
-  

### Brainstorm these

- storing enemy data in yaml to be read into the game for spawning
  - can use the yaml import methods made elsewhere

 

## NEXT 

- bullet collides and there's a reaction, but it crashes the game.
ERROR:
File "/home/raf/projects/py_projects/pandas_learn/game.py", line 81, in bullet_enemy_collision
    if enemy.HP <= 0:
AttributeError: 'NoneType' object has no attribute 'HP'
- Gotta fix this first