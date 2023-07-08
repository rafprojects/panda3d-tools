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
    return ((-x*scale_factor, -0.5, -y*scale_factor), (x*scale_factor, 0.5, y*scale_factor))

# print(grab_text_file_line('assets/sprites/weapons/bullet.egg', 1))
# print(get_dimensions_from_egg('assets/sprites/weapons/bullet.egg', half=True))