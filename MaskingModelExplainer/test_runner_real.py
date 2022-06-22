import os
from argparse import ArgumentParser
from time import time

from explainers.logger_utils import time_format
from tests.ADRealData import run_test
from utils.argument_parser import DefaultReal


class Real(DefaultReal):
    def __call__(self, parser):
        parser = super().__call__(parser)
        return parser


if __name__ == '__main__':
    conf = Real()
    conf = conf(ArgumentParser())
    args = conf.parse_args()

    path = f"tests/logs/{vars(args)['dataset']}_{time_format(time())}"
    os.mkdir(path)

    with open(os.path.join(path, 'conf.txt'), 'w') as f:
        print('-------- PARAMETERS -------', file=f)
        for k in vars(args).keys():
            print('{} {}'.format(k, vars(args)[k]), file=f)
        print('---------------------------', file=f)

    print('Start test...')
    run_test(path, **vars(args))
    print('test completed')