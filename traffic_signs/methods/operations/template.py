from typing import List
import numpy as np
from model import GroundTruth
from model import rectangle
from model import Data
import cv2

class Template: 
    
    signs: List[rectangle]
    masks: List[np.array] 
       
    def __init__(self):
        self.signs= []
        self.masks=[]
    
    def add_sign(self,a:rectangle):
        self.signs.append(a)
 
    def get_sizes(self,data:Data):  
        sign_type= {}
        total = 0
        for sample in data:
            for gt in sample.gt:
                if gt.type not in sign_type.keys():
                    sign_type[gt.type] = Template()

            sign_type[gt.type].add_sign(a=rectangle(gt.rectangle.top_left,gt.rectangle.width,gt.rectangle.height))
            total += 1

        return sign_type, total  

    def get_max_areas(self,data:Data):    
        sign_type=self.get_sizes(data)
        sum=0
        types=['A','B','C','D','E','F']
        max={}
        max1=0
        for i in enumerate(types):  
            if i not in max.keys():
                max[i] = Template()     
            for j in range(len(sign_type[i])):        
                if sign_type[i].signs[j].get_area()>max1: 
                    max1 = sign_type_stats[i].signs[j]; 
    
                max[i].add_sign(sign_type[i].signs[j])
                
        return max    
    
    def draw_mask_circle(self, width:int,height:int):
        image = np.zeros((int(width)+10, int(height)+10, 3), np.uint8)
        image[:]=0
        center=int(width)/2
        cv2.circle(image, center, width/2, (255,255,255),-1)
        return image 
    
    def draw_mask_triangle(self,width:int,height:int):
        image = np.zeros((int(width)+10, int(height)+10, 3), np.uint8)
        image[:]=0
        
        point1= (4, int(width/2)+4)
        point2= ((height+4), 4)
        point3= ((height+4), 4+width)
        
      
        cv2.circle(image, point1, 2, (255,255,255), -1)
        cv2.circle(image, point2, 2, (255,255,255), -1)
        cv2.circle(image, point3, 2, (255,255,255), -1)
        
        triangle= np.array( [point1, point2, point3] )

        cv2.drawContours(image, [triangle], 0, (255,255,255), -1)
        
        return image
    def draw_mask_triangle_inv(self,width:int,height:int):
        image = np.zeros((int(width)+10, int(height)+10, 3), np.uint8)
        image[:]=0
        
        point1= (4, 4)
        point2= (4, 4+width)
        point3= (height+4, int(width/2)+2)
      
        cv2.circle(image, point1, 2, (255,255,255), -1)
        cv2.circle(image, point2, 2, (255,255,255), -1)
        cv2.circle(image, point3, 2, (255,255,255), -1)
        
        triangle= np.array( [point1, point2, point2] )

        cv2.drawContours(image, [triangle], 0, (255,255,255), -1)
        
        return image    
    def draw_rectangles(self,width:int,height:int):
        image = np.zeros((int(width)+10, int(height)+10, 3), np.uint8)
        image[:]=0
        top_rigth=(4,4)
        bottom_right=(4+width,4+height) 
        cv2.rectangle(image,top_rigth,bottom_right,(255,255,255),-1)
        return image
    
    def draw_by_type(self, type:str, width:int, height:int):
        switcher = {
            'A': self.draw_mask_triangle,
            'B': self.raw_mask_triangle_inv,
            'C': self.draw_mask_circle,
            'D': self.draw_mask_circle,
            'E': self.draw_mask_circle,
            'F': self.draw_rectangles
        }
         
        func = switcher.get(type, lambda: "Invalid Type")

        # Execute the function
        image=func(width,height)
        
        return image
        
    def train_masks(self,data:Data):   
        images={}
        maxes=self.get_max_areas(data)
        types=['A','B','C','D','E','F']
        
        for i in enumerate(types):  
            average_width=0;
            average_height=0;    
            for j in len(maxes[i]):
                average_width=average_width+maxes[i].signs[j].rectangle.width
                average_height=average_height+ maxes[i].signs[j].rectangle.height  
            self.masks[i]=self.draw_by_type(i,int(average_width/len(maxes[i])), int(average_height/len(maxes[i])))
                                  
    
    def template_matching(self, img: np.array):
        res={}
        types=['A','B','C','D','E','F']
        final=0
        for i in enumerate(types): 
            res[i]= cv2.matchTemplate(img,self.masks[i],cv2.TM_CCOEFF_NORMED)
            if res[i]>final:
                final=res[i]
                type=i;
         
        threshold = 0.5
        locations = np.where( final >= threshold)
        
        return locations,type

instance= Template()        
        
        
        
        
       
         
         
            
        
        
        