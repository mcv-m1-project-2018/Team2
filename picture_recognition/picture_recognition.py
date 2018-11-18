import argparse
import fnmatch
import os
import pickle
import sys
from typing import List
from itertools import product

import ml_metrics as metrics
import pandas
from functional import seq

from methods import AbstractMethod, w5, w5_no_frame, w5_no_frame_no_text
from model import Data, Picture
from model.rectangle import Rectangle
from tqdm import tqdm


def get_result(method: AbstractMethod, query: Picture):
    return method.query(query)


def query(dataset_dir: str, query_dir: str, methods: List[AbstractMethod]):
    data = Data(dataset_dir)
    file_names = fnmatch.filter(os.listdir(query_dir), '*.jpg')

    print('Training...')
    texts_recs = [method.train(data.pictures) for method in methods]

    print('Querying...')

    query_pictures = seq(file_names).map(lambda query_name: Picture(query_dir, query_name)).to_list()

    results = []
    for method in methods:
        print('\tRunning method', method.__class__.__name__)
        mres = []
        for picture in tqdm(query_pictures, file=sys.stdout):
            mres.append((picture,) + get_result(method, picture))
        results.append(mres)

    return results, texts_recs


def main():
    # read arguments
    parser = argparse.ArgumentParser(description='Search the picture passed in a picture database.')
    parser.add_argument('dataset', help='Source images folder')
    parser.add_argument('query', help='Query images folder')
    parser.add_argument('methods', help='Method list separated by ;')
    parser.add_argument('--out', help='Output directory to run as test execution. Don\'t evaluate results')

    args = parser.parse_args()

    method_refs = {
        'w5': w5,
        'w5_no_frame': w5_no_frame,
        'w5_no_frame_no_text': w5_no_frame_no_text
    }
    method_names = args.methods.split(';')
    methods = seq(method_names).map(lambda x: method_refs.get(x, None)).to_list()
    if not all(methods):
        raise Exception('Invalid method')

    results, text_recs = query(args.dataset, args.query, methods)

    if args.out is not None:
        save_results(method_names, results, args.out)
    else:
        show_results(args.query, method_names, results, text_recs)


def save_results(method_names: List[str], results, output_dir: str):
    for pos, method_name in enumerate(method_names):
        if not os.path.isdir(output_dir + '/' + method_name):
            os.mkdir(output_dir + '/' + method_name)

        result_values = (
            seq(results[pos])
                .map(lambda r: r[1])
                .map(lambda r: seq(r).map(lambda s: s.get_trimmed_name()).to_list())
                .map(replace_empty)
                .to_list()
        )

        with open(output_dir + '/' + method_name + '/result.pkl', 'wb') as f:
            pickle.dump(result_values, f)


def show_results(query_path: str, method_names: List[str], matching_results, text_results):
    # if 'w5' in query_path:
    with open('./w5_query_devel.pkl', 'rb') as file:
        matching_dict = pickle.load(file)
    with open('./w5_text_bbox_list.pkl', 'rb') as file:
        text_dict = pickle.load(file)
        texts_sol = (seq(text_dict)
                     .map(lambda p: Rectangle(p[0:2], (p[2] - p[0]) + 1, (p[3] - p[1]) + 1))
                     .to_list())

    table = []
    for pos, method_name in enumerate(method_names):
        # Matching results
        matching = (
            seq(matching_results[pos])
                .map(lambda r: r[1])
                .map(lambda r: seq(r).map(lambda s: s.id).to_list())
                .map(replace_empty)
                .to_list()
        )
        matching_solution = seq(matching_results[pos]).map(lambda r: matching_dict[r[0].id][1]).to_list()

        # Text results
        text_iou = (seq(texts_sol)
                    .zip(text_results[pos])
                    .map(lambda pair: pair[0].ioi(pair[1]))
                    .average())

        table.append((method_name,
                      metrics.mapk(matching_solution, matching, k=10),
                      metrics.mapk(matching_solution, matching, k=5),
                      metrics.mapk(matching_solution, matching, k=1),
                      text_iou))

    data = pandas.DataFrame(table, columns=['Method', 'MAPK K=10', 'MAPK K=5', 'MAPK K=1', 'Text IoU'])

    print(data)


def replace_empty(lst: List[int]) -> List[int]:
    if len(lst) == 0:
        return [-1]
    else:
        return lst


if __name__ == '__main__':
    main()
