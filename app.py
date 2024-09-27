import copy
import time
import json
import requests
import numpy as np
from flask_cors import CORS
from flask import Flask, request, jsonify
from utils.parse_frontend import parse_data
from utils.faiss_processing import MyFaiss
from utils.context_encoding import VisualEncoding
from utils.semantic_embed.tag_retrieval import tag_retrieval
from utils.combine_utils import merge_searching_results_by_addition
from utils.search_utils import group_result_by_video, search_by_filter, group_result_by_video_old
from gevent.pywsgi import WSGIServer
import glob
print("Starting server")

json_path = 'dict/id2img_fps.json'
audio_json_path = 'dict/audio_id2img_id.json'
scene_path = 'dict/scene_id2info.json'
bin_clip_file = 'dict/v17/faiss_clip.bin'
bin_blip_file = 'dict/v17/faiss_blip2.bin'
video_division_path = 'dict/video_division_batch.json'
img2audio_json_path = 'dict/img_id2audio_id.json'

VisualEncoder = VisualEncoding()
print("ok1")
CosineFaiss = MyFaiss(bin_clip_file, bin_blip_file,
                      json_path, audio_json_path, img2audio_json_path)
print("ok2")
TagRecommendation = tag_retrieval()
print("ok3")
DictImagePath = CosineFaiss.id2img_fps
TotalIndexList = np.array(list(range(len(DictImagePath)))).astype('int64')
print("Run 1")
with open(scene_path, 'r') as f:
    Sceneid2info = json.load(f)

with open('dict/map_keyframes.json', 'r') as f:
    KeyframesMapper = json.load(f)

with open(video_division_path, 'r') as f:
    VideoDivision = json.load(f)

with open('dict/video_id2img_id.json', 'r') as f:
    Videoid2imgid = json.load(f)

# Đọc scene
scene_map_dict = dict()
for part in glob.glob('dict/SceneJSON/*'):
    for video_path in glob.glob(f'{part}/*'):
        with open(video_path,'r') as file:
            scene_map_dict[f'{part[-3:]}_{video_path[-9:-5]}'] = json.loads(''.join(file.readlines()))

def find_split(part,video_id,frame):
    lst = scene_map_dict[f'{part}_{video_id}']
    for i in lst:
        if int(frame) >= int(i[0]) and int(frame) <= int(i[1]):
            frame_id = str(i[0])
            frame_id = '0'*(6-len(frame_id)) + frame_id
            return f'{part}_{video_id}_{frame_id}'

print("Run 2")
def get_search_space(id):
  # id starting from 1 to 4
  search_space = []
  video_space = VideoDivision[f'list_{id}']
  for video_id in video_space:
    search_space.extend(Videoid2imgid[video_id])
  return search_space
# def get_search_space(id):
#     # id starting from 1 to 4
#     search_space = []
#     # video_space = VideoDivision[f'list_{id}']
#     # for video_id in video_space:
#     #     search_space.extend(Videoid2imgid[video_id])
#     for i in range(1, 12):
#         if i < 10:
#             l = '0'+str(i)
#         else:
#             l = str(i)
#         name = f"L{l}_V"
#         map1 = [1, 2, 5, 6, 7]
#         map2 = [3, 4, 8, 11, 12]
#         map3 = [9, 10]
#         if i in map1:
#             for i in range(1, 32):
#                 id = str(i)
#                 while len(id) != 3:
#                     id = '0' + id
#                 search_space.extend(Videoid2imgid[name+id])
#         elif id in map2:
#             for i in range(1, 31):
#                 id = str(i)
#                 while len(id) != 3:
#                     id = '0' + id
#                 search_space.extend(Videoid2imgid[name+id])
#         else:
#             for i in range(1, 30):
#                 id = str(i)
#                 while len(id) != 3:
#                     id = '0' + id
#                 search_space.extend(Videoid2imgid[name+id])
#     print(len(search_space))
#     with open('temp.txt', 'w') as f:
#         f.write(str(search_space))
#     return search_space


SearchSpace = dict()
for i in range(1, 3):
    SearchSpace[i] = np.array(get_search_space(i)).astype('int64')
SearchSpace[0] = TotalIndexList

print("Run 3")
def get_near_frame(idx):
    image_info = DictImagePath[idx]
    scene_idx = image_info['scene_idx'].split('/')
    near_keyframes_idx = copy.deepcopy(
        Sceneid2info[scene_idx[0]][scene_idx[1]][scene_idx[2]][scene_idx[3]]['lst_keyframe_idxs'])
    return near_keyframes_idx


def get_related_ignore(ignore_index):
    total_ignore_index = []
    for idx in ignore_index:
        total_ignore_index.extend(get_near_frame(idx))
    return total_ignore_index


# Run Flask app
app = Flask(__name__, template_folder='templates')
CORS(app)

print("Run 4")
@app.route('/data')
def index():
    pagefile = []
    for id, value in DictImagePath.items():
        if int(id) > 500:
            break
        pagefile.append({'imgpath': value['image_path'], 'id': id})
    data = {'pagefile': pagefile}
    return jsonify(data)


@app.route('/imgsearch')
def image_search():
    print("image search")
    k = int(request.args.get('k'))
    id_query = int(request.args.get('imgid'))
    lst_scores, list_ids, _, list_image_paths = CosineFaiss.image_search(
        id_query, k=k)
    
    data = group_result_by_video_old(
        lst_scores, list_ids, list_image_paths, KeyframesMapper)

    return jsonify(data)


@app.route('/textsearch', methods=['POST'], strict_slashes=False)
def text_search():
    print("text search")
    data = request.json

    search_space_index = int(data['search_space'])
    k = int(data['k'])
    clip = data['clip']
    blip = data['blip']
    blip = True
    query = data['textquery']
    
    range_filter = int(data['range_filter'])
    index = None
    if data['filter']:
        index = np.array(data['id']).astype('int64')
        k = min(k, len(index))
        print("using index")

    keep_index = None
    ignore_index = None
    if data['ignore']:
        ignore_index = get_related_ignore(
            np.array(data['ignore_idxs']).astype('int64'))
        keep_index = np.delete(TotalIndexList, ignore_index)
        print("using ignore")

    if keep_index is not None:
        if index is not None:
            index = np.intersect1d(index, keep_index)
        else:
            index = keep_index

    if index is None:
        index = SearchSpace[0]
    else:
        index = np.intersect1d(index, SearchSpace[0])
    k = min(k, len(index))

    if clip and blip:
        model_type = 'both'
    elif blip:
        model_type = 'blip'
    elif clip:
        model_type = 'clip'
        
    scores_map = dict()
    list_ids_dict = dict()
    # for query in queries:
        # print(query)
        # with open('temp.txt','a') as file:
        #     file.write(str(query))
    if data['filtervideo'] != 0:
        print('filter video')
        mode = data['filtervideo']
        prev_result = data['videos']
        data = search_by_filter(prev_result, query, k, mode, model_type, range_filter,
                                ignore_index, keep_index, Sceneid2info, DictImagePath, CosineFaiss, KeyframesMapper)
    else:
        if model_type == 'both':
            scores_clip, list_clip_ids, _, _ = CosineFaiss.text_search(
                query, index=None, k=k, model_type='clip')
            scores_blip, list_blip_ids, _, _ = CosineFaiss.text_search(
                query, index=None, k=k, model_type='blip')
            lst_scores, list_ids = merge_searching_results_by_addition([scores_clip, scores_blip],
                                                                    [list_clip_ids, list_blip_ids])
            infos_query = list(map(CosineFaiss.id2img_fps.get, list(list_ids)))
            list_image_paths = [info['image_path'] for info in infos_query]
        else:
            lst_scores, list_ids, _, list_image_paths = CosineFaiss.text_search(query, index=None, k=k, model_type=model_type)
                
        #tạo score map cho từng query
    #     score_map_dict = dict()
    #     distinct_frame_posittion = set()
        
        
    #     for i in range(k):
    #         part = list_image_paths[i].split('/')[3].replace('_extra','')
    #         video_id = list_image_paths[i].split('/')[4]
    #         frame_id = list_image_paths[i].split('/')[5][:6]
            
    #         frame_posittion = find_split(part,video_id,frame_id)
            
    #         distinct_frame_posittion.add(frame_posittion)
    #         score_map_dict[frame_posittion] = max(score_map_dict.get(frame_posittion,0),lst_scores[i])
    #         list_ids_dict[frame_posittion] = list_ids[i]
        
        
    #     for x in distinct_frame_posittion:
    #         scores_map[x] = scores_map.get(x,0) + score_map_dict[x]
            
    # data = group_result_by_video(
    #     lst_scores, list_ids, list_image_paths, 
    #     KeyframesMapper,
    #     scores_map,
    #     list_ids_dict,
    #     scene_map_dict)
    
    data = group_result_by_video_old(
        lst_scores, list_ids, list_image_paths, KeyframesMapper)
    return jsonify(data)


@app.route('/panel', methods=['POST'], strict_slashes=False)
def panel():
    print("panel search")
    search_items = request.json
    k = int(search_items['k'])
    search_space_index = int(search_items['search_space'])

    index = None
    if search_items['useid']:
        index = np.array(search_items['id']).astype('int64')
        k = min(k, len(index))

    keep_index = None
    if search_items['ignore']:
        ignore_index = get_related_ignore(
            np.array(search_items['ignore_idxs']).astype('int64'))
        keep_index = np.delete(TotalIndexList, ignore_index)
        print("using ignore")

    if keep_index is not None:
        if index is not None:
            index = np.intersect1d(index, keep_index)
        else:
            index = keep_index

    if index is None:
        index = SearchSpace[search_space_index]
    else:
        index = np.intersect1d(index, SearchSpace[search_space_index])
    k = min(k, len(index))

    # Parse json input
    object_input = parse_data(search_items, VisualEncoder)
    if search_items['ocr'] == "":
        ocr_input = None
    else:
        ocr_input = search_items['ocr']

    if search_items['asr'] == "":
        asr_input = None
    else:
        asr_input = search_items['asr']

    semantic = False
    keyword = True
    lst_scores, list_ids, _, list_image_paths = CosineFaiss.context_search(object_input=object_input, ocr_input=ocr_input, asr_input=asr_input,
                                                                           k=k, semantic=semantic, keyword=keyword, index=index, useid=search_items['useid'])

    data = group_result_by_video_old(
        lst_scores, list_ids, list_image_paths, KeyframesMapper)
    return jsonify(data)


@app.route('/getrec', methods=['POST'], strict_slashes=False)
def getrec():
    print("get tag recommendation")
    k = 50
    text_query = request.json
    tag_outputs = TagRecommendation(text_query, k)
    return jsonify(tag_outputs)


@app.route('/relatedimg')
def related_img():
    print("related image")
    id_query = int(request.args.get('imgid'))
    image_info = DictImagePath[id_query]
    image_path = image_info['image_path']
    scene_idx = image_info['scene_idx'].split('/')

    video_info = copy.deepcopy(Sceneid2info[scene_idx[0]][scene_idx[1]])
    video_url = video_info['video_metadata']['watch_url']
    video_range = video_info[scene_idx[2]][scene_idx[3]]['shot_time']

    near_keyframes = video_info[scene_idx[2]
                                ][scene_idx[3]]['lst_keyframe_paths']
    near_keyframes.remove(image_path)

    data = {'video_url': video_url, 'video_range': video_range,
            'near_keyframes': near_keyframes}
    return jsonify(data)


@app.route('/getvideoshot')
def get_video_shot():
    print("get video shot")

    if request.args.get('imgid') == 'undefined':
        return jsonify(dict())

    id_query = int(request.args.get('imgid'))
    image_info = DictImagePath[id_query]
    scene_idx = image_info['scene_idx'].split('/')
    shots = copy.deepcopy(
        Sceneid2info[scene_idx[0]][scene_idx[1]][scene_idx[2]])

    selected_shot = int(scene_idx[3])
    total_n_shots = len(shots)
    new_shots = dict()
    for select_id in range(max(0, selected_shot-5), min(selected_shot+6, total_n_shots)):
        new_shots[str(select_id)] = shots[str(select_id)]
    shots = new_shots

    for shot_key in shots.keys():
        lst_keyframe_idxs = []
        for img_path in shots[shot_key]['lst_keyframe_paths']:
            # print(img_path)
            data_part, video_id, frame_id = img_path.replace(
                '/data/KeyFrames/', '').replace('.webp', '').split('/')[-3:]
            key = f'{data_part}_{video_id}'.replace('_extra', '')
            if 'extra' not in data_part:
                if len(key.split('_')) >= 3:
                    key = video_id.replace('_extra', '')
                frame_id = KeyframesMapper[key][str(int(frame_id.split('.')[0]))]

            frame_id = int(str(frame_id).split('.')[0])
            lst_keyframe_idxs.append(frame_id)
        shots[shot_key]['lst_idxs'] = shots[shot_key]['lst_keyframe_idxs']
        shots[shot_key]['lst_keyframe_idxs'] = lst_keyframe_idxs

    data = {
        'collection': scene_idx[0],
        'video_id': scene_idx[1],
        'shots': shots,
        'selected_shot': scene_idx[3]
    }
    return jsonify(data)


@app.route('/feedback', methods=['POST'], strict_slashes=False)
def feed_back():
    data = request.json
    k = int(data['k'])
    prev_result = data['videos']
    lst_pos_vote_idxs = data['lst_pos_idxs']
    lst_neg_vote_idxs = data['lst_neg_idxs']
    lst_scores, list_ids, _, list_image_paths = CosineFaiss.reranking(
        prev_result, lst_pos_vote_idxs, lst_neg_vote_idxs, k)
    data = group_result_by_video_old(
        lst_scores, list_ids, list_image_paths, KeyframesMapper)
    return jsonify(data)


@app.route('/translate', methods=['POST'], strict_slashes=False)
def translate():
    data = request.json
    text_query = data['textquery']
    text_query_translated = CosineFaiss.translater(text_query)
    return jsonify(text_query_translated)

print("Starting server2")
# Debug/Development
# app.run(host="0.0.0.0", port="8080")
# Production
http_server = WSGIServer(('', 8080), app)
http_server.serve_forever()
