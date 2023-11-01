from panda3d.core import LineSegs, NodePath


def grab_text_file_line(textfile, line_num):
    if not isinstance(line_num, int):
        raise Exception('This needs to be an int')
    files = []
    with open(textfile, 'r') as file:
        files = file.readlines()
    return files[line_num]


def get_dimensions_from_egg(eggfile, half=False):
    line = grab_text_file_line(eggfile, 1)
    chunk = line.split()[6].split(',')
    if not half:
        return int(chunk[1]), int(chunk[3])
    else:
        return int(chunk[1]) / 2, int(chunk[3]) / 2


def get_box_dimensions(eggfile, scale_factor, offsetT=None):
    x, y = get_dimensions_from_egg(eggfile)
    if not offsetT:
        return ((-x*scale_factor, -0.5, -y*scale_factor), (x*scale_factor, 0.5, y*scale_factor))
    else: 
        return (
                (
                    (-x+offsetT[0])*scale_factor, 
                    -0.5, 
                    (-y+offsetT[1])*scale_factor
                ), 
                (
                    (x+offsetT[0])*scale_factor, 
                    0.5, 
                    (y+offsetT[1])*scale_factor
                )
                )
       
        
def get_bounding_box_dimensions(obj):
    min_point, max_point = obj.getTightBounds()
    return ((min_point.getX(), min_point.getY(), min_point.getZ()), 
            (max_point.getX(), max_point.getY(), max_point.getZ())
        )
    
    
def make_bounding_box(obj):
    min_point, max_point = obj.getTightBounds()
    lines = LineSegs()

    # Draw a box with lines
    lines.moveTo(min_point)
    lines.drawTo(max_point.getX(), min_point.getY(), min_point.getZ())
    lines.drawTo(max_point.getX(), min_point.getY(), max_point.getZ())
    lines.drawTo(min_point.getX(), min_point.getY(), max_point.getZ())
    lines.drawTo(min_point)
    lines.moveTo(max_point)
    lines.drawTo(min_point.getX(), max_point.getY(), max_point.getZ())
    lines.drawTo(min_point.getX(), max_point.getY(), min_point.getZ())
    lines.drawTo(max_point.getX(), max_point.getY(), min_point.getZ())
    lines.drawTo(max_point)

    lines.moveTo(min_point.getX(), min_point.getY(), max_point.getZ())
    lines.drawTo(min_point.getX(), max_point.getY(), max_point.getZ())
    lines.moveTo(max_point.getX(), min_point.getY(), max_point.getZ())
    lines.drawTo(max_point.getX(), max_point.getY(), max_point.getZ())
    lines.moveTo(min_point.getX(), min_point.getY(), min_point.getZ())
    lines.drawTo(min_point.getX(), max_point.getY(), min_point.getZ())
    lines.moveTo(max_point.getX(), min_point.getY(), min_point.getZ())
    lines.drawTo(max_point.getX(), max_point.getY(), min_point.getZ())

    # Create a node and add it to the object
    node = lines.create()
    node_path = NodePath(node)
    node_path.reparentTo(obj.model)
    
    
    
# print(grab_text_file_line('assets/sprites/weapons/bullet.egg', 1))
# print(get_dimensions_from_egg('assets/sprites/weapons/bullet.egg', half=True))