def grab_text_file_line(textfile, line_num):
    if not isinstance(line_num, int):
        raise Exception('This needs to be an int')
    files = []
    with open(textfile, 'r') as file:
        files = file.readlines()
    return files[line_num]

def get_dimensions_from_egg(eggfile):
    line = grab_text_file_line(eggfile, 1)
    chunk = line.split()[6].split(',')
    return int(chunk[1]), int(chunk[3])
    

print(grab_text_file_line('assets/sprites/weapons/bullet.egg', 1))
print(get_dimensions_from_egg('assets/sprites/weapons/bullet.egg'))