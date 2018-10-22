import argparse


def main():
    # read arguments
    parser = argparse.ArgumentParser(description='Search the picture passed in a picture database.')
    parser.add_argument('dataset', help='Source images folder')
    parser.add_argument('query', help='Query images folder')
    parser.add_argument('--threads', type=int, help='Number of threads to use.', default=4)


if __name__ == '__main__':
    main()
