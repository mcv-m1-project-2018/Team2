#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import fnmatch
import os
from timeit import default_timer as timer

import cv2
from functional import seq
from joblib import Parallel, delayed
from tabulate import tabulate

import evaluation.evaluation_funcs as evalf
from data_analysis import data_analysis
from model.dataset_manager import DatasetManager
from methods import method1, method2, method3, method4
from model import Result


def validate(analysis, dataset_manager, pixel_methods):
    """In each job, the methods are executed with the same dataset split and their results are put in an array."""
    results = []
    train, verify = dataset_manager.get_data_splits()
    if analysis is True:
        data_analysis(train)
    for pixel_method in pixel_methods:
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        time = 0

        pixel_method.train(train)

        start = timer()
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

        time += timer() - start

        results.append(Result(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            time=(time / len(verify))
        ))
    return results


def combine_results(result1, result2, executions):
    result1.tp += result2.tp / executions
    result1.fp += result2.fp / executions
    result1.fn += result2.fn / executions
    result1.tn += result2.tn / executions
    result1.time += result2.time / executions
    return result1


def train_mode(train_dir: str, pixel_methods, window_method: str, analysis=False, threads=4, executions=10):
    """In train mode, we split the dataset and evaluate the result of several executions"""
    # Use this class to load and manage states
    dataset_manager = DatasetManager(train_dir)

    # Perform the executions in parallel
    results = Parallel(n_jobs=threads)(
        delayed(lambda x: validate(analysis, dataset_manager, pixel_methods))(i) for i in range(executions))

    # Average the results of each execution
    results = seq(results) \
        .reduce(lambda a, b: seq(a).zip(b).map(lambda l: combine_results(l[0], l[1], executions)).to_list(),
                [Result() for _i in range(executions)]) \
        .to_list()

    return results


def test_mode(train_dir: str, test_dir: str, output_dir: str, pixel_method, window_method: str):
    """In test mode, the output of the method is stored in disk and no evaluation is performed."""
    dataset_manager = DatasetManager(train_dir)
    pixel_method.train(dataset_manager.data)

    file_names = sorted(fnmatch.filter(os.listdir(test_dir), '*.jpg'))
    file_names = seq(file_names).map(lambda s: s.replace('.jpg', '')).to_list()
    for name in file_names:
        img_path = '{}/{}.jpg'.format(test_dir, name)
        im = cv2.imread(img_path)
        mask, im = pixel_method.get_mask(im)
        mask = cv2.divide(mask, 255)
        cv2.imwrite('{}/mask.{}.png'.format(output_dir, name), mask)


def main():
    # read arguments
    parser = argparse.ArgumentParser(description='Detect traffic signs in images using non ML methods.')
    parser.add_argument('train_path', help="Training directory")
    parser.add_argument('pixel_methods', help='Methods to use separated by ";". If using --output, pass a single '
                                              'param. Valid values are method1, method2, method3 and method4')
    parser.add_argument('--test', help='Test directory, if using test dataset to generate masks.')
    parser.add_argument('--output', help='Output directory, if using test dataset to generate masks.')
    parser.add_argument('--window-method', help='Window method to use.')
    parser.add_argument('--analysis', action='store_true',
                        help='Whether to perform an analysis of the train split before evaluation. Train mode only.')
    parser.add_argument('--threads', type=int, help='Number of threads to use. Train mode only.', default=4)
    parser.add_argument('--executions', type=int, help='Number of executions of each method. Train mode only.')
    args = parser.parse_args()

    methods = args.pixel_methods.split(';')
    method_refs = {
        'method1': method1,
        'method2': method2,
        'method3': method3,
        'method4': method4
    }
    methods = seq(methods).map(lambda x: method_refs.get(x, None)).to_list()
    if not all(methods):
        raise Exception('Invalid method')
    results = None
    if args.output and len(methods) == 1:
        test_mode(args.train_path, args.test, args.output, methods[0], args.window_method)
    else:
        results = train_mode(args.train_path, methods, args.window_method, threads=args.threads, executions=args.executions)

    if results:
        print(tabulate(seq(results)
                       .map(lambda result: [result.get_precision(), result.get_accuracy(), result.get_recall(),
                                            result.get_specificity(), result.tp, result.fp, result.fn, result.time])
                       .reduce(lambda accum, r: accum + [r], []),
                       headers=['Precision', 'Accuracy', 'Recall', 'F1', 'TP', 'FP', 'FN', 'Time']))


if __name__ == '__main__':
    main()
