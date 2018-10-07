import numpy as np
from evaluation.bbox_iou import bbox_iou


def performance_accumulation_pixel(pixel_candidates, pixel_annotation):
    """ 
    performance_accumulation_pixel()

    Function to compute different performance indicators 
    (True Positive, False Positive, False Negative, True Negative) 
    at the pixel level
       
    [pixelTP, pixelFP, pixelFN, pixelTN] = performance_accumulation_pixel(pixel_candidates, pixel_annotation)
       
    Parameter name      Value
    --------------      -----
    'pixel_candidates'   Binary image marking the detected areas
    'pixel_annotation'   Binary image containing ground truth
       
    The function returns the number of True Positive (pixelTP), False Positive (pixelFP), 
    False Negative (pixelFN) and True Negative (pixelTN) pixels in the image pixel_candidates
    """

    pixel_candidates = np.uint64(pixel_candidates > 0)
    pixel_annotation = np.uint64(pixel_annotation > 0)

    pixelTP = np.sum(pixel_candidates & pixel_annotation)
    pixelFP = np.sum(pixel_candidates & (pixel_annotation == 0))
    pixelFN = np.sum((pixel_candidates == 0) & pixel_annotation)
    pixelTN = np.sum((pixel_candidates == 0) & (pixel_annotation == 0))

    return [pixelTP, pixelFP, pixelFN, pixelTN]


def performance_accumulation_window(detections, annotations):
    """ 
    performance_accumulation_window()

    Function to compute different performance indicators (True Positive, 
    False Positive, False Negative) at the object level.
    
    Objects are defined by means of rectangular windows circumscribing them.
    Window format is [ struct(x,y,w,h)  struct(x,y,w,h)  ... ] in both
    detections and annotations.
    
    An object is considered to be detected correctly if detection and annotation 
    windows overlap by more of 50%
    
       function [TP,FN,FP] = PerformanceAccumulationWindow(detections, annotations)
    
       Parameter name      Value
       --------------      -----
       'detections'        List of windows marking the candidate detections
       'annotations'       List of windows with the ground truth positions of the objects
    
    The function returns the number of True Positive (TP), False Positive (FP), 
    False Negative (FN) objects
    """

    detections_used = np.zeros(len(detections))
    annotations_used = np.zeros(len(annotations))
    TP = 0
    for ii in range(len(annotations)):
        for jj in range(len(detections)):
            if (detections_used[jj] == 0) & (bbox_iou(annotations[ii], detections[jj]) > 0.5):
                TP = TP + 1
                detections_used[jj] = 1
                annotations_used[ii] = 1

    FN = np.sum(annotations_used == 0)
    FP = np.sum(detections_used == 0)

    return [TP, FN, FP]
