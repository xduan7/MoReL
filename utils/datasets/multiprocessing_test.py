""" 
    File Name:          MoReL/multiprocessing_test.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/28/19
    Python Version:     3.5.4
    File Description:   

"""
import time

from multiprocessing import Manager, Process
from multiprocessing.managers import DictProxy

import numpy


def modify(d: DictProxy):
    # k = str(len(d))
    if '3' in d:
        d[12] = None
    del d['0']
    d['2'] = [2, 3, 4]
    d['1'] = numpy.array([1, 2, 3])
    time.sleep(0.2)


if __name__ == '__main__':
    # Test on shared data structure
    manager = Manager()
    test_d = manager.dict()

    print(type(test_d))
    test_d['0'] = (numpy.array([1, 2, 3]), numpy.array([1, 2, 3]))
    test_d['2'] = [0, 1, 2]
    test_d['3'] = None
    print(test_d)

    p1 = Process(target=modify, args=(test_d,))
    p1.start()

    for i in range(100):
        print(test_d)
    time.sleep(5)
    # p1.join()

    print(test_d)
    time.sleep(1)

    for k in test_d.keys():
        if k == '1':
            del test_d[k]
    print(test_d)
