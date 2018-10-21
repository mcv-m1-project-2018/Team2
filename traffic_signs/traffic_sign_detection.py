#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import fnmatch
import os
import pickle
import sys
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import List

import cv2
import numpy as np
from functional import seq
from tabulate import tabulate

import evaluation.evaluation_funcs as evalf
from data_analysis import data_analysis
from evaluation.bbox_iou import bbox_iou
from methods import hsv_convolution, hsv_integral, hsv_sw, hsv_cc, hsv_cc_template, glob_template
from model import DatasetManager, Data
from model import Result


def validateMethod(train: List[Data], verify: List[Data], method):
    method.train(train)
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    t = 0
    tp_w = 0
    fn_w = 0
    fp_w = 0

    for pos, dat in enumerate(verify):
        print(method, str(pos) + '/' + str(len(verify)))
        im = dat.get_img()

        start = time.time()
        regions, mask, im = method.get_mask(im)
        mask_solution = dat.get_mask_img()
        t += time.time() - start

        [local_tp, local_fp, local_fn, local_tn] = evalf.performance_accumulation_pixel(
            mask, mask_solution)

        [local_tp_w, local_fn_w, local_fp_w] = performance_accumulation_window(regions, dat.gt)

        tp += local_tp
        fp += local_fp
        fn += local_fn
        tn += local_tn

        tp_w += local_tp_w
        fn_w += local_fn_w
        fp_w += local_fp_w

        """import matplotlib.pyplot as plt
        for region in regions:
            cv2.rectangle(mask, (region.top_left[1], region.top_left[0]),
                          (region.get_bottom_right()[1], region.get_bottom_right()[0]), (255,), thickness=5)
        for gt in dat.gt:
            cv2.rectangle(mask, (gt.top_left[1], gt.top_left[0]),
                          (gt.get_bottom_right()[1], gt.get_bottom_right()[0]), (128,), thickness=5)

        plt.imshow(mask, 'gray', vmax=255)
        plt.title(cv2.countNonZero(mask))
        plt.show()
        pass"""

    return Result(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        time=(t / len(verify)),
        tp_w=tp_w,
        fn_w=fn_w,
        fp_w=fp_w
    )


def validate(analysis: bool, dataset_manager: DatasetManager, pixel_methods: List):
    """In each job, the methods are executed with the same dataset split and their results are put in an array."""
    train, verify = dataset_manager.get_data_splits()
    if analysis:
        data_analysis(train)

    with ThreadPoolExecutor() as executor:
        results = [executor.submit(validateMethod, train, verify, pixel_method)
                   for pixel_method in pixel_methods]

    results = seq(results) \
        .map(lambda f: f.result()) \
        .to_list()

    return results


def performance_accumulation_window(detections_gt, annotations_gt):
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
    detections = []
    for element in detections_gt:
        detections.append([element.top_left[0], element.top_left[1], element.get_bottom_right()[0],
                           element.get_bottom_right()[1]])

    annotations = []
    for element in annotations_gt:
        annotations.append([element.top_left[0], element.top_left[1], element.get_bottom_right()[0],
                            element.get_bottom_right()[1]])

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


def combine_results(result1: Result, result2: Result, executions):
    result1.tp += result2.tp / executions
    result1.fp += result2.fp / executions
    result1.fn += result2.fn / executions
    result1.tn += result2.tn / executions
    result1.time += result2.time / executions
    result1.tp_w += result2.tp_w / executions
    result1.fp_w += result2.fp_w / executions
    result1.fn_w += result2.fn_w / executions
    return result1


def train_mode(train_dir: str, methods, analysis=False, threads=4, executions=10):
    """In train mode, we split the dataset and evaluate the result of several executions"""
    # Use this class to load and manage states
    dataset_manager = DatasetManager(train_dir)
    # Perform the executions in parallel
    with ThreadPoolExecutor(max_workers=threads) as executor:
        results = [executor.submit(validate, analysis, dataset_manager, methods)
                   for _ in range(executions)]

        # Average the results of each execution
        results = seq(results) \
            .map(lambda fut: seq(fut.result())) \
            .reduce(lambda a, b: seq(a).zip(b).map(lambda l: combine_results(l[0], l[1], executions)).to_list(),
                    [Result() for _i in range(len(methods))]) \
            .to_list()

        return results


def test_mode(train_dir: str, test_dir: str, output_dir: str, method):
    """In test mode, the output of the method is stored in disk and no evaluation is performed."""
    dataset_manager = DatasetManager(train_dir)
    method.train(dataset_manager.data)

    file_names = sorted(fnmatch.filter(os.listdir(test_dir), '*.jpg'))
    file_names = seq(file_names).map(lambda s: s.replace('.jpg', '')).to_list()
    for name in file_names:
        img_path = '{}/{}.jpg'.format(test_dir, name)
        im = cv2.imread(img_path)
        regions, mask, im = method.get_mask(im)
        mask = cv2.divide(mask, 255)
        cv2.imwrite('{}/mask.{}.png'.format(output_dir, name), mask)
        with open('{}/mask.{}.pkl'.format(output_dir, name), 'wb') as file:
            pickle.dump(seq(regions)
                        .map(lambda r: (r.top_left[0], r.top_left[1], r.get_bottom_right()[0],
                                        r.get_bottom_right()[1]))
                        .to_list(),
                        file)


def main():
    # read arguments
    parser = argparse.ArgumentParser(description='Detect traffic signs in images using non ML methods.')
    parser.add_argument('train_path', help="Training directory")
    parser.add_argument('pixel_methods', help='Methods to use separated by ";". If using --output, pass a single '
                                              'param. Valid values are method1, method2, method3 and method4')
    parser.add_argument('--test', help='Test directory, if using test dataset to generate masks.')
    parser.add_argument('--output', help='Output directory, if using test dataset to generate masks.')
    parser.add_argument('--analysis', action='store_true',
                        help='Whether to perform an analysis of the train split before evaluation. Train mode only.')
    parser.add_argument('--threads', type=int, help='Number of threads to use. Train mode only.', default=4)
    parser.add_argument('--executions', type=int, help='Number of executions of each method. Train mode only.',
                        default=10)
    args = parser.parse_args()

    method_names = args.pixel_methods.split(';')
    method_refs = {
        'hsv_sw': hsv_sw,
        'hsv_convolution': hsv_convolution,
        'hsv_integral': hsv_integral,
        'hsv_cc': hsv_cc,
        'hsv_cc_template': hsv_cc_template,
        'glob_template': glob_template
    }
    methods = seq(method_names).map(lambda x: method_refs.get(x, None)).to_list()
    if not all(methods):
        raise Exception('Invalid method')
    results = None
    if args.output:
        if len(methods) != 1:
            print('Test mode must receive a single method', file=sys.stderr)
        else:
            test_mode(args.train_path, args.test, args.output, methods[0])
    else:
        results = train_mode(args.train_path, methods, threads=args.threads,
                             executions=args.executions)

    if results:
        print(tabulate(seq(results)
                       .zip(method_names)
                       .map(lambda pair: [pair[1], pair[0].get_precision(), pair[0].get_accuracy(),
                                          pair[0].get_recall(), pair[0].get_f1(), pair[0].get_precision_w(),
                                          pair[0].get_recall_w(), pair[0].get_f1_w(), pair[0].tp, pair[0].fp,
                                          pair[0].fn, pair[0].time])
                       .reduce(lambda accum, r: accum + [r], []),
                       headers=['Method', 'Precision', 'Accuracy', 'Recall', 'F1', 'Precision w', 'Recall w', 'F1 w',
                                'TP', 'FP', 'FN', 'Time']))


if __name__ == '__main__':
    main()
