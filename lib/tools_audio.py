import scipy.io.wavfile as wav
import numpy as np
import os
import pydub
import tempfile
import scipy
import random
from python_speech_features import logfbank

from lib.tools_math import *

def changeRateTo16000(filepath):
    if filepath[-4:].lower() =='.wav':
        sound = pydub.AudioSegment.from_wav(filepath)
        sound = sound.set_frame_rate(16000)
        sound.export(filepath, format="wav")
    elif filepath[-4:].lower() =='.m4a':
        sound = pydub.AudioSegment.from_file(filepath, "m4a")
        sound = sound.set_frame_rate(16000)
        sound.export(filepath[:-3]+"wav", format="wav")
    elif filepath[-4:].lower() =='.mp3':
        sound = pydub.AudioSegment.from_mp3(filepath)
        sound = sound.set_frame_rate(16000)
        sound.export(filepath[:-3]+"wav", format="wav")
    else:
        print("Unsupported Format.")

def read_wav(file_path):
    assert file_path[-4:]=='.wav'
    rate, data = wav.read(file_path)
    return rate, data

def read_m4a(file_path):
    path, ext = os.path.splitext(file_path)
    assert ext=='.m4a'
    aac_version = pydub.AudioSegment.from_file(file_path, "m4a")
    _, path = tempfile.mkstemp()
    aac_version.export(path, format="wav")
    rate, data = scipy.io.wavfile.read(path)
    os.remove(path)
    return rate, data

def read_mp3(file_path):
    path, ext = os.path.splitext(file_path)
    assert ext=='.mp3'
    mp3 = pydub.AudioSegment.from_mp3(file_path)
    _, path = tempfile.mkstemp()
    mp3.export(path, format="wav")
    rate, data = scipy.io.wavfile.read(path)
    os.remove(path)
    return rate, data

def mp3_to_wav(file_path, obj_path):
    path, ext = os.path.splitext(file_path)
    assert ext=='.mp3'
    mp3 = pydub.AudioSegment.from_mp3(file_path)
    mp3.export(obj_path, format="wav")

def mergeChannels(data):
    data = normalize(data)
    if len(data.shape)==1:
        return data
    if len(data.shape)==2:
        return np.mean(data, axis = 1)
    raise ValueError("This is not what an audio file ought to be!")

def getDefaultSpectrogram(rate, data):
    f, t, Sxx = signal.spectrogram(data, fs=rate, window='hamming', nperseg=400, noverlap=240, nfft=1024, scaling='spectrum', return_onesided=True)
    return Sxx

def frame_split(data, frame_width, frame_step):
    if len(data)<frame_width:
        raise ValueError("The length of data is shorter than the frame width.")
    frame_max = int(np.floor((len(data)-frame_width+frame_step)/float(frame_step)))
    result = []
    for i in range(frame_max):
        start = i*frame_step
        end = start+frame_width
        result.append(data[start:end])
    return np.array(result)

def get_mel_db(wave_data, sr, winlen = 0.025, winstep = 0.01, nfft = 512, num_mel = 40, wav_process=False):
    # Input 10.015 sec, output, (1000, 40)
    # nfft >= num data points in one window(frame)
    if wav_process == True:
        frame_shift = int(sr * winstep)
        frame_size = int(sr * winlen)
        wave_data, index = librosa.effects.trim(wave_data, frame_length=frame_size, hop_length=frame_shift)
    mel_db = logfbank(wave_data, samplerate=sr, winlen=winlen, winstep=winstep,
                      nfilt=num_mel, nfft=nfft, lowfreq=0, highfreq=None, preemph=0.97)
    mel_db -= (np.mean(mel_db,axis=1).reshape(-1,1)+1e-8)
    return mel_db
