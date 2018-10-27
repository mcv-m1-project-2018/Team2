import argparse
import fnmatch
import os
import pickle
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List

import ml_metrics as metrics
from functional import seq
from tabulate import tabulate

from methods import AbstractMethod, method2, method4, method5
from model import Data, Picture


def get_result(method: AbstractMethod, query: Picture):
    return method.query(query)


def query(dataset_dir: str, query_dir: str, methods: List[AbstractMethod], threads=4):
    data = Data(dataset_dir)
    file_names = fnmatch.filter(os.listdir(query_dir), '*.jpg')
    with ThreadPoolExecutor(max_workers=threads) as executor:
        seq(methods).for_each(lambda m: m.train(data.pictures))
        query_pictures = seq(file_names).map(lambda query_name: Picture(query_dir, query_name)).to_list()
        result = (
            seq(methods)
                .map(lambda method:
                     [[picture, executor.submit(get_result, method, picture)] for picture in query_pictures]
                     )
                .to_list()
        )

    return (
        seq(result)
            .map(lambda r: seq(r).map(lambda f: (f[0], f[1].result())).to_list())
            .to_list()
    )


def main():
    # read arguments
    parser = argparse.ArgumentParser(description='Search the picture passed in a picture database.')
    parser.add_argument('dataset', help='Source images folder')
    parser.add_argument('query', help='Query images folder')
    parser.add_argument('methods', help='Method list separated by ;')
    parser.add_argument('--threads', type=int, help='Number of threads to use.', default=4)

    args = parser.parse_args()

    method_refs = {
        'method2': method2,
        'method4': method4,
        'method5': method5
    }
    method_names = args.methods.split(';')
    methods = seq(method_names).map(lambda x: method_refs.get(x, None)).to_list()
    if not all(methods):
        raise Exception('Invalid method')

    results = query(args.dataset, args.query, methods, args.threads)

    show_results(args.query, method_names, results)


def show_results(query_dir, method_names, results):
    with open(query_dir + '/query_corresp_simple_devel.pkl', 'rb') as file:
        query_dict = pickle.load(file)

    table = []
    for pos, method_name in enumerate(method_names):
        solutions = seq(results[pos]).map(lambda r: [query_dict.get(r[0].id)]).to_list()
        result_values = (
            seq(results[pos])
                .map(lambda r: r[1])
                .map(lambda r: seq(r).map(lambda s: s[0].id).to_list())
                .to_list()
        )

        correct_first_places = (
            seq(solutions)
                .zip(result_values)
                .count(lambda p: p[0][0] == p[1][0])
        )

        table.append((method_name, metrics.mapk(solutions, result_values), correct_first_places))
    print(tabulate(table, headers=['Method name', 'MAPK Score', 'Correct first places']))


if __name__ == '__main__':
    main()
