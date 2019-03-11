import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

from lib.tools_audio import *

def play(vec, Fs):
    sd.play(vec, Fs, blocking=True)

def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data(file_path):
    try:
        _, data_temp = read_mp3(file_path)
    except:
        _, data_temp = read_wav(file_path)
    return data_temp

def plotSound(vec):
    plt.plot(vec)
    plt.ylabel('Amplitude')
    plt.show()
