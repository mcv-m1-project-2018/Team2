import numpy as np
from enum import Enum
import cv2
from matplotlib import image as mpimg
from builtins import int

class Blocks():
    numberofblocks:int
    blocks:List[np.array]
    masks:List[np.array]


def getBlocks(img:np.array):
    size_x = img.shape[0]
    size_y = img.shape[1]
    numblock=(self.numberofblocks/2)
    block_x = int(size_x / numblock)
    block_y = int(size_y / numblock)
    block=0
    
    if size_x % columns == 0 & size_y % rows == 0:

        for i in range(numblock):
            for j in range(numblock):
                while( j != numblock and i != numblock):
                    block = img[i * block_y:(i + 1) * block_x, j * block_y: (j + 1) * block_y]
                    self.blocks.append(block)

    else:
        #Padding
        while img.shape[0] % columns !=0:
            padding = np.array([img[:, -1]])
            img = np.hstack((img, padding))
         
        while img.shape[1] % rows !=0:
            padding = np.array([img[-1,:]])
            img = np.vstack((img,padding))
    
        for i in range(numblock):
            for j in range(numblock):
                while( j != numblock and i != numblock):
                    block = img[i * block_y:(i + 1) * block_x, j * block_y: (j + 1) * block_y]   
                    self.blocks.append(block)
    
def getMasks(img:np.array,):
    #in progress
    
    
    
    
    
    

  