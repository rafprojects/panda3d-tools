import os


def textureGetter(base_path, subdir=None):
    '''Get a list of texture filepaths from a directory'''
    if subdir:
        filepath = f"{base_path}/{subdir}"
    else:
        filepath = f"{base_path}"
    return [f"{filepath}/{i}.png" for i in range(len(os.listdir(filepath)))]      
    
    
def textureLoader(base, anim_states, base_path, subdir=None):
    textureD = {}
    if subdir:
        filepath = f"{base_path}/{subdir}"
    else:
        filepath = f"{base_path}"
    for state in anim_states:
        for i in range(len(os.listdir(f"{filepath}/{state}"))):
            filename = f"{filepath}/{state}/{i}.png"
            texture = base.loader.loadTexture(filename)
            textureD[(state, i)] = texture
    return textureD
    
    
def get_frame_counts(texture_dict):
    frame_countsD = {}
    for key in texture_dict:
        if key[0] not in frame_countsD:
            frame_countsD[key[0]] = 1
        else:
            frame_countsD[key[0]] += 1
    return frame_countsD
    
    
# print(textureGetter("output", "ship/idle"))

# print(textureLoader(base=None, anim_states=["idle", "left", "right"], base_path="output", subdir="ship"))