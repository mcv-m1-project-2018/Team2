import argparse
import fnmatch
import os
import pickle
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List

from functional import seq

from methods import method1, AbstractMethod
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
        'method1': method1
    }
    methods = seq(args.methods).map(lambda x: method_refs.get(x, None)).to_list()
    if not all(methods):
        raise Exception('Invalid method')

    results = query(args.dataset, args.query, methods, args.threads)

    # Evaluate
    with open(args.query + '/query_corresp_simple_devel.pkl', 'rb') as file:
        query_dict = pickle.load(file)

    for r in results:
        for q in r:
            q[0].show()
            q[1][0][0].show()

    """for pos, method_name in enumerate(methods):
        solutions = seq(results[pos]).map(lambda r: query_dict.get(r[0].id)).to_list()
        result_values = (
            seq(results[pos])
                .map(lambda r: r[1])
                .map(lambda r: seq(r).map(lambda s: s[0].id))
                .to_list()
        )
        for i in range(len(solutions)):
            print(solutions[i] in result_values[i])"""


if __name__ == '__main__':
    main()
