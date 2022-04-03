from datetime import datetime
from os import makedirs, path
from os.path import join as os_join
import argparse
from tqdm import tqdm
import sys

project_root = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
code_path = os_join(project_root,'code')
print (os_join(project_root,'code'))
sys.path.append(code_path)


def inspect_data(args):
    dataset = getattr(
        __import__('datasets'), args.dataset.capitalize() + 'Dataset'
    )
    train_dataset = dataset('train', args)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default = 'lawschool')
    parser.add_argument('--protected-att', type=str, default=None)
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--quantiles', action='store_true')

    args = parser.parse_args()
    inspect_data(args)
