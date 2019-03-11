import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def get_topk_args(arr, k):
    return arr.argsort()[::-1][:k]

def get_distance(v1, v2):
    if len(v2.shape) != 1:
        raise ValueError("arg2 should be an 1d array.")
    if len(v1.shape) == 1:
        return np.sqrt(np.sum(np.square(v1-v2)))
    elif len(v1.shape) == 2:
        return np.sqrt(np.sum(np.square(v1-v2), axis = 1))
    else:
        raise ValueError("arg1 should be rather 1d or 2d array.")

def get_cos_sim(v1, v2):
    if len(v2.shape) != 1:
        raise ValueError("arg2 should be an 1d array.")
    if len(v1.shape) == 1:
        inner = np.sum((v1*v2))
        normv1 = np.sqrt(np.sum(np.square(v1)))
        normv2 = np.sqrt(np.sum(np.square(v2)))
        return inner/(normv1*normv2)
    elif len(v1.shape) == 2:
        inner = np.sum((v1*v2), axis = 1)
        normv1 = np.sqrt(np.sum(np.square(v1), axis = 1))
        normv2 = np.sqrt(np.sum(np.square(v2)))
        return inner/(normv1*normv2)

def index2onehot(indices, label_range):
    result = np.zeros([len(indices), label_range])
    result[np.arange(len(indices)), indices] = 1.0
    return result

def randomExcept(n, end, start = 0):
    r = range(start, n) + range(n+1, end)
    return np.random.choice(r)

def zero_padding_1d(vec, obj_length):
    result = np.concatenate([vec, np.zeros(obj_length-len(vec))])
    return result

def neg_padding_1d(vec, obj_length):
    result = np.concatenate([vec, -np.ones(obj_length-len(vec))])
    return result
