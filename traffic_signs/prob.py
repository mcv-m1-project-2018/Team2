

from dataset_manager import DatasetManager
from typing import List
import numpy as np
from matplotlib import pyplot as plt
import cv2
from functional import seq

from model import GroundTruth
from methods.operations import histogram_equalization

from methods.method2 import instance as method2


dataManager = DatasetManager("../datasets/train")
print('Loading data...')
dataManager.load_data()
data = dataManager.data
sign_type_stats = {}
print('Running...')
total = 0




for sample in data:
    img = sample.get_img()
    mask = sample.get_mask_img()
    (result_mask,result)=method2.get_mask(img)
    
    plt.subplot(2, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.subplot(2, 2, 2)
    plt.imshow(result_mask, cmap="gray")
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 2, 4)
    plt.imshow(result)
   
    print("mask")
    print(np.sum(result_mask))
    plt.show()
    
    
    
    
    