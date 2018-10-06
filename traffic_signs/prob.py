import cv2
import numpy as np
from matplotlib import pyplot as plt

from dataset_manager import TestDatasetManager
from methods import method3

dataManager = TestDatasetManager("../datasets/train")
print('Loading data...')
dataManager.load_data()
data = dataManager.data
sign_type_stats = {}
print('Running...')
total = 0
method3.train(data)

for sample in data:
    img = sample.get_img()
    mask = sample.get_mask_img()
    (result_mask, result) = method3.get_mask(img)

    plt.subplot(2, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.subplot(2, 2, 2)
    plt.imshow(result_mask, cmap="gray")
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 2, 4)
    plt.imshow(result)

    plt.show()
