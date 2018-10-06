#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import fnmatch
import os

import cv2

import evaluation.evaluation_funcs as evalf
from data_analysis import data_analysis
from dataset_manager import DatasetManager
from methods import method1, method2, method3, method4
from model import Result


def train_mode(input_dir: str, pixel_method, window_method: str, analysis=False):
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    # Use this class to load and manage states
    dataset_manager = DatasetManager(input_dir)
    dataset_manager.load_data()

    train, verify = dataset_manager.get_data_splits()
    if analysis is True:
        data_analysis(train)

    pixel_method.train(train)

    for dat in verify:
        im = dat.get_img()

        mask, im = pixel_method.get_mask(im)
        mask_solution = dat.get_mask_img()

        [local_tp, local_fp, local_fn, local_tn] = evalf.performance_accumulation_pixel(
            mask, mask_solution)
        tp += local_tp
        fp += local_fp
        fn += local_fn
        tn += local_tn

    return Result(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn
    )


def test_mode(input_dir: str, output_dir: str, pixel_method, window_method: str):
    file_names = sorted(fnmatch.filter(os.listdir(input_dir), '*.jpg'))
    for name in file_names:
        img_path = '{}/{}.jpg'.format(input_dir, name)
        im = cv2.imread(img_path)
        mask, im = pixel_method.get_mask(im)
        cv2.imwrite('{}/{}.jpg'.format(output_dir, name), mask)


def main():
    # read arguments
    parser = argparse.ArgumentParser(description='Detect traffic signs in images using non ML methods')
    parser.add_argument('input', help="Input directory")
    parser.add_argument('pixel_method', choices=['method1', 'method2', 'method3', 'method4'], help='Method to use')
    parser.add_argument('--output', help='Output directory, if using test dataset to generate masks')
    parser.add_argument('--window-method', help='Window method, if needed')
    parser.add_argument('--analysis', action='store_true',
                        help='Whether to perform an analysis of the train split before evaluation')
    args = parser.parse_args()
    methods = {
        'method1': method1,
        'method2': method2,
        'method3': method3,
        'method4': method4
    }
    method = methods.get(args.pixel_metdho, lambda: 'Invalid method')
    result = None
    if args.output:
        test_mode(args.input, args.output, method, args.window_method)
    else:
        result = train_mode(args.input, method, args.window_method)
    if result:
        print('Precision:', result.get_precision())
        print('Recall:', result.get_recall())


if __name__ == '__main__':
    main()
