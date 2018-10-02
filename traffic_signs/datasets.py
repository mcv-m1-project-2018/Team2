from typing import List, Tuple
import numpy as np
import fnmatch
import os
import cv2
from _ast import List
import random

class Size:
    height: float
    width: float

    def max(self,other):
        if self.height > other.height and self.width > other.width:
            return self
        elif self.height < other.height and self.width < other.width:
            return other
        else:
            NewSize = Size()
            NewSize.height = max(self.height,other.height)
            NewSize.width = max(self.width, other.width)
            return NewSize

    def min(self, other):
        if self.height < other.height and self.width < other.width:
            return self
        elif self.height > other.height and self.width > other.width:
            return other
        else:
            NewSize = Size
            NewSize.height = min(self.height, other.height)
            NewSize.width = min(self.width, other.width)
            return NewSize

def get_mask_area(gt,mask):
    self.mask1=cv2.imread(mask)
    
    return mask_area


def get_filling_factor(gt, mask):

    # compute the area of bboxes
    bbox_area = (gt.bottom_right[0] - gt.top_left[0] + 1) * (gt.bottom_right[1] - gt.top_left[1] + 1)
    mask_area = get_mask_area(gt,mask)

    # return the filling ratio
    return mask_area/float(bbox_area)

class SignalType:
    max_size: Tuple[float]
    min_size: Tuple[float]
    form_factor: List[float]
    filling_ratio: List[float]
    form_factor_avg: float
    filling_ratio_avg: float
    appearance_frequency: float

    def add_signal(self, gt , mask):
        self.max_size = Size.max(gt.size)
        self.min_size = Size.min(gt.size)
        self.form_factor.append(float(gt.size.width/gt.size.height))
        self.filling_ratio.append(get_filling_factor(gt,mask))

    def get_avg(self,data_length):
        self.form_factor_avg = np.mean(self.form_factor)
        self.filling_ratio_avg = np.mean(self.filling_ratio)
        self.appearance_frequency = len(self.form_factor)/data_length



class GroundTruth:
    top_left: Tuple[float]
    bottom_right: Tuple[float]
    size: Size
    type: str

class Data:
    """Stores the content of a data element."""

    name: str
    gt: List[GroundTruth] = []
    img: str
    mask: str

    def __init__(self, directory: str, name: str):
        self.name = name
        self.img = '{}/{}'.format(directory, name)
        self.mask = '{}/mask/mask.{}.png'.format(directory, name)
        with open('{}/gt/gt.{}.txt'.format(directory, name)) as f:
            for line in f.readlines():
                parts = line.split(' ')
                gt = GroundTruth()
                gt.top_left = (parts[0], parts[1])
                gt.bottom_right = (parts[2], parts[3])
                gt.type = parts[4]
                gt.size.width = float(parts[3]) - float(parts[1])
                gt.size.height = float(parts[2]) - float(parts[0])
                self.gt.append(gt)


class DatasetManager:
    """We will use k-fold validation. More info at https://www.openml.org/a/estimation-procedures/1"""

    data: List[Data]
    _dir: str

    def __init__(self, directory: str):
        self._dir = directory

    def load_data(self):
        file_names = sorted(fnmatch.filter(os.listdir(self._dir), '*.jpg'))
        for file_name in file_names:
            self.data.append(Data(self._dir, file_name))
        print('Loading data')

    
    def trainsplit(self):
       Atype=List[Data]
       Btype=List[Data]
       Ctype=List[Data]
       Dtype=List[Data]
       Etype=List[Data]
       Ftype=List[Data]
       types= [] 
       for sample in data:
                if (sample.gt.type()=='A'):
                    Atype.append(sample)
                elif (sample.gt.type()=='B'):
                    Btype.append(sample)
                elif (sample.gt.type()=='C' ):
                    Ctype.append(sample)  
                elif (sample.gt.type()=='D'):
                    Dtype.append(sample)
                elif (sample.gt.type()=='E'):
                    Etype.append(sample)
                else:
                    Ftype.append(sample)   
       types.append(Atype)
       types.append(Btype)
       types.append(Ctype)
       types.append(Dtype)
       types.append(Etype)
       types.append(Ftype)
     
       return types
   

    def get_sets(self):
        training=List[Data]
        test=List[Data]
        [types]=trainsplit()
        for i in types:
            random.shuffle(types[i][:])
            for j in types[i]:
                if j<= 0.7*len(types[i]):
                    training.append(types[i][j])
                else:
                    test.append(types[i][j])
                 
        
        "Get the validation and training sets of data from the original training dataset 30% to 70% from each class"
        pass

