import numpy as np
import pickle
from pypinyin import lazy_pinyin

class pinyinParser:

    def __init__(self, path):
        with open(path, 'rb') as handle:
            self.pinyinDict = pickle.load(handle)
        invPath = path[:-7]+"Inv.pickle"
        with open(invPath, 'rb') as handle:
            self.pinyinDict_inv = pickle.load(handle)

    def getDictSize(self):
        return len(self.pinyinDict)

    def getPinYin(self, unicodeContent):
        return " ".join([x for x in lazy_pinyin(unicodeContent)])

    def _index2OneHot(self, index):
        result = np.zeros(len(self.pinyinDict))
        result[index] = 1.0
        return result

    def _indices2OneHot(self, indices):
        result = np.zeros([len(indices), len(self.pinyinDict)])
        result[np.arange(len(indices)), indices] = 1.0
        return result

    def getPinYinIndices(self, pinyin):
        pinyinList = pinyin.strip().split()
        indices = []
        for pinyin in pinyinList:
            if pinyin in self.pinyinDict:
                indices.append(self.pinyinDict[pinyin])
            else:
                raise ValueError("Could not find "+pinyin+" in the dictionary.")
        if len(indices)==0:
            raise ValueError("Invalid input.")
        return indices

    def getPinYinOneHot(self, pinyin):
        pinyinList = pinyin.strip().split()
        indices = []
        for pinyin in pinyinList:
            if pinyin in self.pinyinDict:
                indices.append(self.pinyinDict[pinyin])
            else:
                raise ValueError("Could not find "+pinyin+" in the dictionary.")
        if len(indices)==0:
            raise ValueError("Invalid input.")
        return self._indices2OneHot(indices)

    def decodeIndices(self, vec, useUnderline = True):
        result = []
        for num in vec:
            if num in self.pinyinDict_inv:
                result.append(self.pinyinDict_inv[num])
        if useUnderline:
            return '_'.join(result)
        else:
            return ''.join(result)
