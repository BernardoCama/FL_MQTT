import sys
import numpy as np
import _pickle as cPickle
from math import floor
import zlib

def Cut_model(size_model, MAX_MODEL_SIZE, num_layers, WEIGHTS):

    # If model size is above the limit -> cut
    if size_model > MAX_MODEL_SIZE:
        print('Need cutting model')
        cutting_points = [num_layers]
        current_size = size_model  
        up = num_layers
        mid = int(num_layers/2)
        down = 0
        max_iter = 30
        iteration = 0
        while current_size >= MAX_MODEL_SIZE and iteration < max_iter:
            temp_size = sys.getsizeof(zlib.compress(cPickle.dumps([WEIGHTS[i] for i in range(mid, up)])))
            # temp_size = sys.getsizeof(cPickle.dumps([WEIGHTS[i] for i in range(mid, up)]))
            print('current_size: {}'.format(current_size))
            print('temp_size: {}'.format(temp_size))
            print('up: {}'.format(up))
            print('mid: {}'.format(mid))
            print('down: {}'.format(down))
            if temp_size < MAX_MODEL_SIZE:
                cutting_points.append(mid)
                up = mid
                mid = floor((up+down)/2)
                current_size = current_size - temp_size
            else:
                mid = floor((mid+up)/2)
            if mid >= up:
                mid = up - 1 
            iteration = iteration + 1         
        if iteration == max_iter:
            raise ValueError('Model cannot be partitioned, one single layer dimension ({}) > MAX_MODEL_SIZE ({})'.format(temp_size,MAX_MODEL_SIZE))
        cutting_points.append(0)
        cutting_points.reverse()
        MODEL_SNIPPETS = len(cutting_points) - 1 
        print(cutting_points)
    else:
        MODEL_SNIPPETS = 1
        cutting_points = [0, num_layers]

    return cutting_points, MODEL_SNIPPETS