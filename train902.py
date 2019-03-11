import os
os.environ["CUDA_VISIBLE_DEVICES"] = input("Which GPU? ")
import time
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf
import numpy as np

from lib.tools_batch import *
from lib.tools_math import *
from model902 import *

def get_learningRate(step):
    # return max(4e-4*(0.99999**step), 2e-5)
    return 2e-4

TEST_ROUND = 1
BATCH_SIZE = 64
TEST_SIZE = 16
AUGMENTATION = True
NUM_STEP = int(1e7)

model_name = "v902"
saving_period = 200
num_labels = 407
num_class = 409
bg = BatchGetter("../data/data_aishell/wav", "../data/data_aishell/transcript/aishell_transcript_v0.8.txt",
                    "lib/pinyinDictNoTone.pickle", "../data/backgrounds/", server = True)
bg2 = BatchGetter("../data/youtube_subtitles/wav", "../data/youtube_subtitles/subs.txt",
                    "lib/pinyinDictNoTone.pickle", "../data/backgrounds/", server = True)
pyParser = pinyinParser("lib/pinyinDictNoTone.pickle")
model = model(num_class)
if model_name not in os.listdir('models/'):
    os.mkdir('models/'+model_name)

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True,log_device_placement=False)) as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # tensorboard --logdir logs/
    summary_writer = tf.summary.FileWriter(logdir = "logs", graph = tf.get_default_graph())
    saver.restore(sess, "models/"+model_name+"/"+model_name+"_0.ckpt")

    for i in range(1, NUM_STEP+1):

        lr = get_learningRate(i)
        if i%2==0:
            xs, ys = bg.get_batch(BATCH_SIZE, batch_type = 'train')
        else:
            xs, ys = bg2.get_batch(BATCH_SIZE, batch_type = 'train')
        loss, summary = model.train(sess, lr, xs, ys)
        summary_writer.add_summary(summary, i)
        print(i, loss)

        if i%saving_period == 0:
            print("Learning rate =", lr)
            save_path = saver.save(sess, "models/"+model_name+"/"+model_name+"_"+str(int(i/50000))+".ckpt")
            print("Model saved in path: "+save_path)

            ave_loss = 0.0
            for i in range(2):
                if i==0:
                    xs, ys = bg.get_batch(TEST_SIZE, batch_type = 'test', augmentation = False)
                else:
                    xs, ys = bg2.get_batch(TEST_SIZE, batch_type = 'test', augmentation = False)
                loss = model.get_loss(sess, xs, ys)
                pred = model.predict(sess, xs)[0]
                report_accuracy(pred, ys, pyParser)
                ave_loss+=loss
            print("Test Loss = "+str(ave_loss/float(TEST_ROUND)))
