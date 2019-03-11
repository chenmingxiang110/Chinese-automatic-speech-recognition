import os
import time
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf
import numpy as np
from urllib.request import urlopen

from lib.tools_batch import *
from lib.tools_math import *
from lib.tools_sparse import *
from lib.contrib.audio_featurizer import AudioFeaturizer
from lib.contrib.audio import AudioSegment
from model901 import *
model_name = "v901"

def timeStamp2Num(timeStamp, rate):
    """
    timeStamp str: 00:00:01,879
    rate int: the sampling rate
    return int
    """
    secs, millisec = timeStamp.split(",")
    hour, minute, sec = secs.split(":")
    millisec = float(millisec)*0.001
    sec = float(hour)*3600+float(minute)*60+float(sec)
    num = int(rate*(sec+millisec))
    return num

pyParser = pinyinParser("lib/pinyinDictNoTone.pickle")
model = model(409)
af = AudioFeaturizer()

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "models/"+model_name+"/"+model_name+"_0.ckpt")

    rate, data = read_wav("data/test.wav")
    data = mergeChannels(data)
    data = zero_padding_1d(data, 160240)
    a_seg = AudioSegment(data, rate)
    xs = np.transpose(np.array([af.featurize(a_seg)]), [0,2,1])

    pred = model.predict(sess, xs)[0]
    pred_dense = sparseTuples2dense(pred)
    detected_line = []
    for stuff in pred_dense[0]:
        if stuff!=-1:
            detected_line.append(stuff)
    pinyin = pyParser.decodeIndices(detected_line, useUnderline = False)
    print(pinyin)
    response = urlopen("https://www.google.com/inputtools/request?ime=pinyin&ie=utf-8&oe=utf-8&app=translate&num=10&text="+pinyin)
    html = response.read()
    result = html.split(",")[2][2:-1]
