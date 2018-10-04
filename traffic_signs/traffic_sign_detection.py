#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import fnmatch
import os
import pickle

import imageio

import evaluation.evaluation_funcs as evalf
from candidate_generation_pixel import candidate_generation_pixel
from candidate_generation_window import candidate_generation_window
from dataset_manager import DatasetManager
from model import Result


def traffic_sign_detection(directory, output_dir, pixel_method, window_method):
    pixel_tp = 0
    pixel_fn = 0
    pixel_fp = 0
    pixel_tn = 0

    window_tp = 0
    window_fn = 0
    window_fp = 0

    window_precision = 0
    window_accuracy = 0

    # Use this class to load and manage states
    datasetManager = DatasetManager(directory)
    datasetManager.load_data()

    # Load image names in the given directory
    file_names = sorted(fnmatch.filter(os.listdir(directory), '*.jpg'))

    for name in file_names:
        base, extension = os.path.splitext(name)

        # Read file
        im = imageio.imread('{}/{}'.format(directory, name))
        print('{}/{}'.format(directory, name))

        # Candidate Generation (pixel) ######################################
        pixel_candidates = candidate_generation_pixel(im, pixel_method)

        fd = '{}/{}_{}'.format(output_dir, pixel_method, window_method)
        if not os.path.exists(fd):
            os.makedirs(fd)

        out_mask_name = '{}.png'.format(fd, base)
        # imageio.imwrite(out_mask_name, np.uint8(np.round(pixel_candidates)))

        # Accumulate pixel performance of the current image #################
        pixel_annotation = imageio.imread('{}/mask/mask.{}.png'.format(directory, base)) > 0

        [local_pixel_tp, local_pixel_fp, local_pixel_fn, local_pixel_tn] = evalf.performance_accumulation_pixel(
            pixel_candidates, pixel_annotation)
        pixel_tp += local_pixel_tp
        pixel_fp += local_pixel_fp
        pixel_fn += local_pixel_fn
        pixel_tn += local_pixel_tn

        if window_method != 'None':
            window_candidates = candidate_generation_window(im, pixel_candidates, window_method)

            out_list_name = '{}/{}.pkl'.format(fd, base)

            with open(out_list_name, "wb") as fp:  # Pickling
                pickle.dump(window_candidates, fp)
            [localWindowTP, localWindowFN, localWindowFP] = evalf.performance_accumulation_window(window_candidates,
                                                                                                  window_annotationss)

            windowTP = windowTP + localWindowTP
            windowFN = windowFN + localWindowFN
            windowFP = windowFP + localWindowFP

            # Plot performance evaluation
            [window_precision, window_sensitivity, window_accuracy] = evalf.performance_evaluation_window(window_tp,
                                                                                                          window_fn,
                                                                                                          window_fp)

    [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity] = evalf.performance_evaluation_pixel(
        pixel_tp, pixel_fp, pixel_fn, pixel_tn)

    return Result(
        pixel_precision=pixel_precision,
        pixel_accuracy=pixel_accuracy,
        pixel_specificity=pixel_specificity,
        pixel_sensitivity=pixel_sensitivity,
        window_precision=window_precision,
        window_accuracy=window_accuracy
    )


if __name__ == '__main__':
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dirName')
    parser.add_argument('outPath')
    parser.add_argument('pixel_method', choices=['color_segmentation'])
    parser.add_argument('--windowMethod')

    args = parser.parse_args()

    images_dir = args.dirName  # Directory with input images and annotations
    # For instance, '../../DataSetDelivered/test'
    output_dir = args.outPath  # Directory where to store output masks, etc. For instance '~/m1-results/week1/test'

    result = traffic_sign_detection(images_dir, output_dir, 'normrgb', args.windowMethod)

    if result:
        print(result)
