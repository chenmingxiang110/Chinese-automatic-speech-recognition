import numpy as np
import time
import os
# import logging
from lib.tools_pinyin import *
from lib.tools_audio import *
from lib.tools_math import *
from lib.tools_augmentation import *
from lib.tools_sparse import *
from lib.contrib.audio_featurizer import AudioFeaturizer
from lib.contrib.audio import AudioSegment

# logger = logging.getLogger(__file__)

class BatchGetter:

    # If turn on the server option, save all the noise waves to RAM, else, only
    # the paths will be saved to RAM.
    def __init__(self, ids_path, transcript_path, pinyin_path, background_root, server = False):
        self.server = server
        self.af = AudioFeaturizer()
        self.pyParser = pinyinParser(pinyin_path)
        temp = self._get_addressList(ids_path)
        self.addressList = []
        self.labels = self._get_labelsDict(transcript_path)
        self.unicodes = self._get_unicodes(transcript_path)
        for address in temp:
            id = address.split('/')[-1][:-4]
            if id in self.labels:
                self.addressList.append(address)

        if background_root[-1] != '/': background_root = background_root+'/'
        root_office = background_root+"office_backgrounds/"
        root_youtube = background_root+"youtube_backgrounds/"
        root_talking = background_root+"youtube_talking/"
        bg_office = [root_office+x for x in os.listdir(root_office) if x[0]!='.']
        bg_youtube = [root_youtube+x for x in os.listdir(root_youtube) if x[0]!='.']
        bg_talking = [root_talking+x for x in os.listdir(root_talking) if x[0]!='.']
        if self.server:
            bg_office_w = [mergeChannels(read_wav(w)[1]) for w in bg_office]
            bg_youtube_w = [mergeChannels(read_wav(w)[1]) for w in bg_youtube]
            bg_talking_w = [mergeChannels(read_wav(w)[1]) for w in bg_talking]
            self.bg_libs = [bg_office_w, bg_youtube_w, bg_talking_w]
        else:
            self.bg_libs = [bg_office, bg_youtube, bg_talking]

    def _get_addressList(self, root):
        if root[-1] == '/': root=root[:-1]
        result = []

        ids = [f for f in os.listdir(root) if f[0]!='.']
        for id in ids:
            files = [root+'/'+id+'/'+f for f in os.listdir(root+'/'+id) if f[0]!='.']
            result.extend(files)
        return result

    def _get_labelsDict(self, transcript_path):
        result = {}
        with open(transcript_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                content = ("".join(line[1:]))
                result[line[0]] = self.pyParser.getPinYinIndices(self.pyParser.getPinYin(content))
        return result

    def _get_unicodes(self, transcript_path):
        result = {}
        with open(transcript_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                content = ("".join(line[1:]))
                result[line[0]] = content
        return result

    # If using the fbank, remember to adjust the x_obj. What is more, the
    # frame_width and frame stride will be disabled, while winlen, winstep, nfft,
    # and num_mel will be enabled.
    # If raw frames: x_obj(163000) = your_obj(160000)+frame_width(4000)-frame_stride(1000)
    # If filterbank: x_obj(160240) = your_obj(160000)+winlen(16000*0.025=400)-winstep(16000*0.01=160)
    def get_batch(self, num, x_obj_min = 16000, x_obj = 160160, batch_type = 'train',
                  augmentation = True, returnUnicode = False, bgMaximum = 0.05,
                  isCTC = True, verbose = False):
        if batch_type=='train':
            range_min = 0
            range_max = int(len(self.addressList))*0.95
        elif batch_type=='test':
            range_min = int(len(self.addressList))*0.95
            range_max = len(self.addressList)
        elif batch_type=='all':
            range_min = 0
            range_max = int(len(self.addressList))
        else:
            return

        xs = []
        ys = []
        aug_total = 0
        mfb_total = 0
        while len(xs)<num:
            index = np.random.randint(range_min, range_max)
            file_path = self.addressList[index]
            rate, data = read_wav(file_path)
            if len(data)<x_obj_min or len(data)>x_obj:
                continue

            id = file_path.split('/')[-1][:-4]

            start = time.time()
            if augmentation:
                bg_lib = self.bg_libs[np.random.randint(len(self.bg_libs))]
                if self.server:
                    ns = bg_lib[np.random.randint(len(bg_lib))]
                else:
                    _, ns = read_wav(bg_lib[np.random.randint(len(bg_lib))])
                    ns = mergeChannels(ns)
                data = randomAugment(data, rate, 1, obj_length = x_obj, noiseSource = ns, bgMaximum = bgMaximum)[0]
            else:
                data = zero_padding_1d(data, x_obj)
            time1 = time.time()
            a_seg = AudioSegment(data, rate)
            xs.append(self.af.featurize(a_seg))
            time2 = time.time()
            aug_total += time1-start
            mfb_total += time2-time1

            if returnUnicode:
                ys.append(self.unicodes[id])
            else:
                ys.append(np.array(self.labels[id]).astype(int))

        if verbose:
            # How the fuck to use print? See this:
            # print('a={first:4.2f}, b={second:03d}'.format(first=f(x,n),second=g(x,n)))
            # print("a=%d,b=%d" % (f(x,n),g(x,n)))
            print("Augmentation time = %f sec; Featurization time = %f sec" % (aug_total, mfb_total))
        xs = np.array(xs)
        xs = np.transpose(xs, [0,2,1])
        if returnUnicode:
            return xs, ys
        else:
            if isCTC:
                ys = sparse_tuple_from(ys)
                return xs, ys
            else:
                ys_lengths = [len(y)+1 for y in ys]
                max_length = max(ys_lengths)
                temp = []

                # The first three tokens should be reserved for padding, start, and end tokens.
                for y in ys:
                    if len(y)<(max_length-1):
                        # Add the end token. (Actually 2, but will be 2 after 3 is added.)
                        y = np.concatenate([y, [-1]])
                        temp.append(np.concatenate([y+3, np.zeros(max_length-len(y))]))
                    else:
                        y = np.concatenate([y, [-1]])
                        temp.append(y+3)
                ys = np.array(temp)
                return xs, ys, ys_lengths
