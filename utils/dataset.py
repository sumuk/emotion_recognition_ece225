import os
import glob
import numpy as np
import random
from multiprocessing.pool import ThreadPool

class Emotion():
    '''
    class to sample files in each emotion
    '''
    def __init__(self,folder_path,label) -> None:
        self.size = {}
        self.label = label
        for file_no,file in enumerate(glob.glob(os.path.join(folder_path,"*.npy"))):
            data = np.load(file)
            self.size[file_no]= (file,data.shape[1])
        self.total_size = len(self.size)
        print("Done loading the {0} class".format(os.path.split(folder_path)[-1]))

    def sample(self,emotion_samples)->tuple:
        out = random.choices(self.size,k=emotion_samples)
        length=[]
        array = []
        for i in out:
            array.append(np.load(i[0]))
            length.append(i[1])
        return (array,[self.label]*emotion_samples,length)

class Audiodataset():
    '''
    class for reading the subfolder of each emotion
    '''
    def __init__(self,folder_path) -> None:
        self.emotions = os.listdir(folder_path)
        self.emotions.sort()
        labels = {j:i for i,j in enumerate(self.emotions)}
        self.emotions = [Emotion(os.path.join(folder_path,i),labels[i]) for i in self.emotions]

    def sample(self,sample_per_emotion) -> tuple:
        '''
        return the samples after appending the zero to make
        all samples equal size and it provides the length array
        '''
        def get_samples(obj):
            return obj.sample(sample_per_emotion)
        with ThreadPool(5) as p:
            t=p.map(get_samples,self.emotions)
        max_len = max([max(i[2]) for i in t])
        length=[]
        labels = [] 
        for i in t:
            length.extend(i[2])
            labels.extend(i[1])
        data = np.vstack([np.pad(j,[(0,0),(0,max_len-le),(0,0)],'constant',constant_values=(0,0)) for j,le in zip(i[0],i[2]) for i in t])
        return (data,labels,length)  



        
