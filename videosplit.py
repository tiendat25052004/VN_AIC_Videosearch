import random
import numpy as np
from glob import glob
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import json

class VideoSplit:
    def __init__(self, data_path='dict/context_encoded/tags_encoded/*'):
        list_vidieo_folder = glob(data_path)
        list_vidieo_folder.sort()
        list_vidieo_folder = np.array(list_vidieo_folder)[::2].tolist()
        self.list_vidieo_path = []
        for folder in list_vidieo_folder:
            self.list_vidieo_path += glob(folder+'/*.txt')
        self.all_vidieo_name = list(map(lambda x: '_'.join([x[34:37], x[-8:-4]]), self.list_vidieo_path))
    
    def generate_random_video(self, n=4):
        result = self.generate_random(len(self.all_vidieo_name), n=n)
        final_result = {}
        for key in result.keys():
            final_result[key] = np.array(self.all_vidieo_name)[np.array(result[key])].tolist()
        return final_result
    
    def generate_tag_based(self, n=4):
        vectorizer = TfidfVectorizer(input='filename', ngram_range = (1, 1), token_pattern=r"(?u)\b[\w\d]+\b")
        X = vectorizer.fit_transform(self.list_vidieo_path)
        sparse_vector = X.toarray()
        kmean = KMeans(n_clusters=4, random_state=42, n_init='auto').fit(sparse_vector)
        cluster_list = {}
        result = {}
        for i in range(n):
            result[f'list_{i+1}'] = []
        for i in range(0, 4):
            cluster_list[f'cluster_{i}'] = np.array(self.all_vidieo_name)[kmean.labels_==i].tolist()
        
        for key in cluster_list.keys():
            random_video_path_id = self.generate_random(len(cluster_list[key]), n=n)
            for list_key in random_video_path_id.keys():
                result[list_key] += np.array(cluster_list[key])[random_video_path_id[list_key]].tolist()
        return result
    
    def generate_batch_based(self, n=2):
        result = {}
        for i in range(n):
            result[f'list_{i+1}'] = []
        for video in self.all_vidieo_name:
            print(video)
            if (int(video.split('_')[0][1:])-1) / 12 < 1:
                result["list_1"].append(video)
                if int(video.split('_')[0][1:]) == 12:
                    print("ok")
            elif (int(video.split('_')[0][1:])-1) / 12 < 2:
                result["list_2"].append(video)
            else:
                print(int(video.split('_')[0][1:])-1)
        return result
    
    @staticmethod
    def generate_random(num_vidieos, n=4):
        result = {}
        first_half = np.arange(num_vidieos).tolist()
        second_half = np.arange(num_vidieos).tolist()
        random.shuffle(first_half)
        random.shuffle(second_half)
        final_shuffle = first_half + second_half
        for i in range(n):
            result[f'list_{i+1}'] = final_shuffle[math.ceil(len(final_shuffle)/n)*i : math.ceil(len(final_shuffle)/n)*(i+1)]
        return result
    
    def test(self, dict_test):
        test_dict = []
        for key in dict_test.keys():
            test_dict += dict_test[key]
        if len(set(test_dict)) == len(self.all_vidieo_name):
            return 'right !!!'
        else:
            return 'wrong !!!!'
        
if __name__ == '__main__':
    split = VideoSplit()
    result = split.generate_batch_based()
    print(split.test(result))
    with open("dict/video_division_batch.json", "w") as f:
        f.write(json.dumps(result))
    