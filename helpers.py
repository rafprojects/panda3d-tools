import os


def textureGetter(base_path, subdir=None):
    '''Get a list of texture filepaths from a directory'''
    if subdir:
        filepath = f"{base_path}/{subdir}"
    else:
        filepath = f"{base_path}"
    return [f"{filepath}/{i}.png" for i in range(len(os.listdir(filepath)))]      
    
def textureLoader(anim_states, base_path, subdir=None):
    '''Prepare a dictionary of textures for use in an animation'''
    texture_dict = {}
    if subdir:
        filepath = f"{base_path}/{subdir}"
    else:
        filepath = f"{base_path}"
    for state in anim_states:
        for i in range(len(os.listdir(f"{filepath}/{state}"))):
            filename = f"{filepath}/{state}/{i}.png"
            texture_dict[(state, i)] = filename
    return texture_dict
    
# print(textureGetter("output", "ship/idle"))

# print(textureLoader(["idle", "left", "right"], "output", "ship"))