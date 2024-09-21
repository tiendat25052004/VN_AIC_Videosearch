import glob
import json
scene_map_dict = dict()
for part in glob.glob('dict/SceneJSON/*'):
    for video_path in glob.glob(f'{part}/*'):
        with open(video_path,'r') as file:
            scene_map_dict[f'{part[-3:]}_{video_path[-9:-5]}'] = json.loads(''.join(file.readlines()))

lst = scene_map_dict['L01_V001']
for i in lst:
    print(i[0])
    print(type(i[0]))
    break