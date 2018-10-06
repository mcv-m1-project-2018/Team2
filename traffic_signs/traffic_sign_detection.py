#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import fnmatch
import os
from timeit import default_timer as timer

import cv2
from tabulate import tabulate

import evaluation.evaluation_funcs as evalf
from data_analysis import data_analysis
from dataset_manager import DatasetManager
from methods import method1, method2, method3, method4
from model import Result
from functional import seq
from joblib import Parallel, delayed

EXECUTIONS = 12


def validate(analysis, dataset_manager, pixel_methods):
    results = []
    train, verify = dataset_manager.get_data_splits()
    if analysis is True:
        data_analysis(train)
    for i, pixel_method in enumerate(pixel_methods):
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
            tp=tp / EXECUTIONS,
            fp=fp / EXECUTIONS,
            fn=fn / EXECUTIONS,
            tn=tn / EXECUTIONS,
            time=(time / len(verify)) / EXECUTIONS
        ))
    return results


def combine_results(result1, result2):
    result1.tp += result2.tp
    result1.fp += result2.fp
    result1.fn += result2.fn
    result1.tn += result2.tn
    result1.time += result2.time
    return result1


def train_mode(input_dir: str, pixel_methods, window_method: str, analysis=False, threads=4):
    # Use this class to load and manage states
    dataset_manager = DatasetManager(input_dir)
    dataset_manager.load_data()

    results = Parallel(n_jobs=threads)(
        delayed(lambda x: validate(analysis, dataset_manager, pixel_methods))(i) for i in range(EXECUTIONS))

    results = seq(results) \
        .reduce(lambda a, b: seq(a).zip(b).map(lambda l: combine_results(l[0], l[1])).to_list(),
                [Result() for i in range(EXECUTIONS)]) \
        .to_list()

    return results


def test_mode(input_dir: str, output_dir: str, pixel_method, window_method: str):
    file_names = sorted(fnmatch.filter(os.listdir(input_dir), '*.jpg'))
    for name in file_names:
        img_path = '{}/{}'.format(input_dir, name)
        im = cv2.imread(img_path)
        mask, im = pixel_method.get_mask(im)
        cv2.imwrite('{}/{}'.format(output_dir, name), mask)


def main():
    # read arguments
    parser = argparse.ArgumentParser(description='Detect traffic signs in images using non ML methods')
    parser.add_argument('input', help="Input directory")
    parser.add_argument('pixel_methods', help='Methods to use separetad by ;. If using --output, pass a single param')
    parser.add_argument('--output', help='Output directory, if using test dataset to generate masks')
    parser.add_argument('--window-method', help='Window method, if needed')
    parser.add_argument('--analysis', action='store_true',
                        help='Whether to perform an analysis of the train split before evaluation')
    parser.add_argument('--threads', help='Number of threads to use. Only train mode.', default=4)
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
        test_mode(args.input, args.output, methods[0], args.window_method)
    else:
        results = train_mode(args.input, methods, args.window_method, threads=int(args.threads))

    if results:
        print(tabulate(seq(results)
                       .map(lambda result: [result.get_precision(), result.get_accuracy(), result.get_recall(),
                                            result.get_specificity(), result.tp, result.fp, result.fn, result.time])
                       .reduce(lambda accum, r: accum + [r], []),
                       headers=['Precision', 'Accuracy', 'Recall', 'F1', 'TP', 'FP', 'FN', 'Time']))


if __name__ == '__main__':
    main()
