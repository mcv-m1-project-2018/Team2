import cv2
import numpy as np
from matplotlib import pyplot as plt

from dataset_manager import DatasetManager
from methods import method1, method2, method3, method4

dataManager = DatasetManager("../datasets/train")
print('Loading data...')
dataManager.load_data()
data = dataManager.data
sign_type_stats = {}
print('Running...')
total = 0
method1.train(data)
method2.train(data)
method3.train(data)
method4.train(data)

for sample in data:
    img = sample.get_img()
    mask_1, im1 = method1.get_mask(img)
    mask_2, im2 = method2.get_mask(img, True)
    mask_3, im3 = method3.get_mask(img)
    mask_4, im4 = method4.get_mask(img)
