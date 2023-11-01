

def handle_bullet(bullet_node, bulletsL):
    print(f"FOUND BULLET: {bullet_node}")
    # Adjust HP or destroy the objects as needed
    bullet_node.HP -= 1
    if bullet_node.HP <= 0:
        bulletsL.remove(bullet_node)
        bullet_node.destroy()
        bullet_node.collNodePath.removeNode()


def handle_enemy(enemy_node, enemiesL):
    print(f"COLLIDE ENEMY: {enemy_node}")
    enemy_node.HP -= 10
    if enemy_node.HP <= 0:
        enemy_node.collNodePath.removeNode()
        enemy_node.destroy()
        enemiesL.remove(enemy_node)


def get_bullet_and_enemy_from_entry(entry):
    bullet_node = entry.getFromNodePath().getPythonTag("bullet")
    enemy_node = entry.getIntoNodePath().getPythonTag("enemy")
    return bullet_node, enemy_node
