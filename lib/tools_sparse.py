import numpy as np
from lib.tools_pinyin import *

def get_maxLengthListinList(ls):
    length = 0
    for l in ls:
        if len(l)>length: length = len(l)
    return length

def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

def sparseTuples2dense(sparseTensor):
    pred_dense = -np.ones(sparseTensor[2])
    for i in range(len(sparseTensor[0])):
        pred_dense[sparseTensor[0][i][0],sparseTensor[0][i][1]] = sparseTensor[1][i]
    return pred_dense

def report_accuracy(decoded_list, test_targets, pyParser):
    original_list = sparseTuples2dense(test_targets)
    detected_list = sparseTuples2dense(decoded_list)
    print("-------------------")
    for i in range(len(original_list)):
        original_line = []
        detected_line = []
        for stuff in original_list[i]:
            if stuff!=-1:
                original_line.append(stuff)
        for stuff in detected_list[i]:
            if stuff!=-1:
                detected_line.append(stuff)
        print(i)
        print(original_line)
        print(detected_line)
        print(pyParser.decodeIndices(original_line, useUnderline = True))
        print(pyParser.decodeIndices(detected_line, useUnderline = True))
        print("-------------------")
