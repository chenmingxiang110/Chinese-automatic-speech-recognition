import numpy as np
import pyfftw
import resampy
from scipy.ndimage import zoom

def randomAugment(data, rate, num, obj_length = None, noiseSource = None, bgMaximum = 0.08, verbose = False):
    """
    Perform random augmentations. Recommended bgMaximum: random noise:0.07,
    office - 0.1, youtube human: 0.07, youtube backgrounds: 0.15.

    :param np.ndarray data: The audio's data point. One channel, which means the
        length of the shape should be one.
    :param int rate: The sampling rate of the audio.
    :param int num: Number of augmentation.
    :param int obj_length: Output audio lengths. Will not be padded if leave it
        none
    :param np.ndarray noiseSource: The source of noise. Will add white noise if
        leave it none.
    :param float bgMaximum: The maximum background sound.
    :param boolean verbose: If true, print out the adjustment made during the
        process.

    :return: A list of audio data points.
    :raises ValueError: If num < 0 or obj_length <= 0 or 0.75*len(data).
    """
    if obj_length is not None:
        if (obj_length <= 0):
            raise ValueError('Objective length must be above than 0.')
        if (obj_length <= 0.75*len(data)):
            raise ValueError('Objective length too short.')
    if num < 0:
        raise ValueError('Number of augmentation must be above than or equal to 0.')
    if num == 0:
        return []

    result = []
    data = _normalize(data)
    for _ in range(num):
        # 1. shift the data a little bit.
        if len(data)>16000:
            shifty = min(int(len(data)/10), np.random.randint(4000))
            if np.random.random()>0.5: shifty *= -1
            transformed = shift(data, shifty)
        else:
            transformed = data
        # 2. Adjust the speed.
        ub = 1.25
        lb = 0.8
        if obj_length is not None:
            zoomUpperBound = min(ub, obj_length/float(len(transformed)))
            if zoomUpperBound<lb:
                zoomFactor = zoomUpperBound
            else:
                zoomFactor = np.random.random()*(zoomUpperBound-lb)+lb
        else:
            zoomFactor = np.random.random()*(ub-lb)+lb
        transformed = audioResize(transformed, zoomFactor)
        if verbose:
            print("Data zoom factor = "+str(zoomFactor))
        # 3. Add noise.
        if bgMaximum>0:
            noiseFactor = np.random.random()*bgMaximum
            if noiseSource is not None:
                transformed = addNoiseFrom(transformed, noiseSource, noiseFactor)
                if verbose:
                    print("Noise added from source.")
                    print("Noise factor = "+str(noiseFactor))
            else:
                transformed = addNoise(transformed, noiseFactor)
                if verbose:
                    print("Noise generated randomly.")
                    print("Noise factor = "+str(noiseFactor))
        # 4. Adjust the volume
        maximum = np.random.random()*0.6+0.4
        transformed = audioVolume(transformed, maximum)
        if verbose: print("Volume adjusted. Max = "+str(maximum)+".")
        # 5. Zero padding
        if obj_length is not None:
            if verbose: print("Shape before zoom is "+str(transformed.shape))
            # First, make sure the length is smaller than or equal to the
            # obj_length. Then pad to obj_length.
            transformed = _zeroPad(transformed[:obj_length], obj_length)
        if verbose:
            print("Shape after zoom is "+str(transformed.shape))
        result.append(transformed)
        if verbose: print("--------------")
    return result

def freqChange(data, rate, freq_range = None, bias = None):
    """
    lower, upper = freq_range
    Trim and shift the frequency. Remove the frequency higher than 'upper' and
    lower than 'lower', then shift the frequency by 'bias'. Probably would have
    error raised when len(data)<rate. If dealing with speech data, the tightest
    bound recommended is (400,2500), and the recommended range of bias is (-25,
    25).

    :param np.ndarray data: The audio's data point. One channel, which means the
        length of the shape should be one.
    :param int rate: The sampling rate of the audio.
    :param int lower: The lower bound of the frequency. Must smaller than the
        upper bound.
    :param int upper: The upper bound of the frequency. Must higher than the
        lower bound, and smaller than rate//2.
    :param int bias: The frequency bias.

    :return: Transformed audio data points. Should be the same size as the
        'data' (Not Tested).
    :raises AssertionError: If lower >= upper.
    """
    yf = pyfftw.interfaces.numpy_fft.fft(data)
    trans = np.copy(yf)
    trans *= 0

    # Clip the frequency.
    if freq_range is not None:
        # Determine the maximum and the minimum frequency.
        minF, maxF = freq_range
        assert maxF>minF
        fBound = int(rate/2)
        minF = max(0, minF)
        maxF = min(fBound, maxF)
        # Determine the maximum and the minimum point.
        minP = int(len(yf)*minF/(2*fBound))
        maxP = int(len(yf)*maxF/(2*fBound))
        # Trim the fourier form.
        trans[minP:maxP] = yf[minP:maxP]
        trans[-maxP:-minP] = yf[-maxP:-minP]
        yf = trans
        trans = np.copy(yf)
        trans *= 0

    # Shift by the bias.
    for i in range(int(len(yf)/2)):
        obj_index = int(i-bias*len(yf)/rate)
        if (obj_index<=(len(yf)/2)) and (obj_index>=0):
            trans[i] = yf[obj_index]
            trans[-i] = yf[-obj_index]

    s = _normalize(pyfftw.interfaces.numpy_fft.ifft(trans).real)
    return s

def dataTrim(data, trim_lower, trim_upper):
    """
    Trim the audio, which means the data points before trim_lower or after
    trim_upper will be removed. trim_upper can be negative.

    :param np.ndarray data: The audio's data point. One channel, which means the
        length of the shape should be one.
    :param int trim_lower: The lower bound to trim.
    :param int trim_upper: The upper bound to trim.

    :return: Transformed audio data points.
    :raises ValueError: If trim_lower >= trim_upper.
    """
    if (trim_lower < 0):
        raise ValueError('Lower bound must be above than or equal to zero.')
    if (trim_upper == 0):
        raise ValueError('Upper bound cannot be zero.')
    if (trim_upper > 0) and (trim_lower >= trim_upper):
        raise ValueError('Lower bound is larger than or equal to the upper bound.')
    if (trim_upper < 0) and ((trim_lower-trim_upper)>=len(data)):
        raise ValueError('Lower bound is larger than or equal to the upper bound.')

    return data[trim_lower:trim_upper]

def dataPadding(data, padding_lower, padding_upper):
    """
    Add zeros to the beginning or the end of the audio.

    :param np.ndarray data: The audio's data point. One channel, which means the
        length of the shape should be one.
    :param int padding_lower: Number of zeros to be added to the beginning.
    :param int padding_upper: Number of zeros to be added to the end.

    :return: Transformed audio data points.
    :raises ValueError: If padding_lower or padding_upper < 0.
    """
    if (padding_lower <= 0) or (padding_upper <= 0):
        raise ValueError('Number of padding must be above than zero.')
    result = np.zeros(len(data)+padding_lower+padding_upper)
    try:
        result[padding_lower:-padding_upper] = data
    except ValueError:
        print(padding_lower)
        print(padding_upper)
        print(len(data))
        raise ValueError("FUCK YOU!!!!!!!!!")
    return result

def audioResize(data, zoomFactor):
    """
    Resize the audio. Not only the length, but the frequency will also be
    changed. If dealing with speech data, the zoomFactor bound recommended is
    [0.75,1.35].

    :param np.ndarray data: The audio's data point. One channel, which means the
        length of the shape should be one.
    :param int zoomFactor: The objective zoomFactor.

    :return: Transformed audio data points.
    :raises ValueError: If zoomFactor <= 0.
    """
    if (zoomFactor <= 0):
        raise ValueError('The zoomFactor should be larger than zero.')
    return zoom(data, zoomFactor)

def audioVolume(data, maximum):
    """
    Set the maximum volume of the audio. I personally do not recommend setting
        the maximum above than 0.99

    :param np.ndarray data: The audio's data point. One channel, which means the
        length of the shape should be one.
    :param float maximum: The maximum volume.

    :return: Transformed audio data points. Should be the same size as the
        'data'.
    :raises ValueError: If maximum < 0 or >1.
    """
    if (maximum < 0) or (maximum > 1):
        raise ValueError('The maximum should be between 0 and 1.')
    return maximum * data / np.max(np.abs(data))

def audioVolumeLinear(data, maximum_start, maximum_end):
    """
    Set the maximum volume of the audio linearly regarding the time.

    :param np.ndarray data: The audio's data point. One channel, which means the
        length of the shape should be one.
    :param float maximum_start: The maximum volume at the beginning.
    :param float maximum_end: The maximum volume in the end.

    :return: Transformed audio data points. Should be the same size as the
        'data'.
    :raises ValueError: If (maximum_start < 0 or >1) or (maximum_end < 0 or >1).
    """
    if (maximum_start < 0) or (maximum_start > 1) or (maximum_end < 0) or (maximum_end > 1):
        raise ValueError('The maximum should be between 0 and 1.')
    maximum = np.linspace(maximum_start, maximum_end, num=len(data))
    return maximum * data / np.max(np.abs(data))

def addNoise(data, noise_factor):
    """
    Add random noise to the audio.

    :param np.ndarray data: The audio's data point. One channel, which means the
        length of the shape should be one.
    :param float noise_factor: The ratio of noise in volume.

    :return: Transformed audio data points. Should be the same size as the
        'data'.
    :raises ValueError: If noise_factor < 0 or >1.
    """
    if (noise_factor < 0) or (noise_factor > 1):
        raise ValueError('The noise factor should be between 0 and 1.')
    noise = np.random.random(size=len(data))*2-1
    noise /= (np.max(np.abs(noise))/noise_factor)
    audio = (1.0-noise_factor) * data / np.max(np.abs(data))
    return noise+audio

def addNoiseFrom(data, noiseSource, noise_factor):
    """
    Add random noise from source. If noise from people speaking, recommend below
    0.05, else, recommend below 0.3

    :param np.ndarray data: The audio's data point. One channel, which means the
        length of the shape should be one.
    :param np.ndarray noiseSource: The noise data. Must be longer or equal to
        the length of 'data'.
    :param float noise_factor: The ratio of noise in volume.

    :return: Transformed audio data points. Should be the same size as the
        'data'.
    :raises ValueError: If len(noiseSource)<len(data).
    :raises ValueError: If noise_factor < 0 or >1.
    """
    if (noise_factor < 0) or (noise_factor > 1):
        raise ValueError('The noise factor should be between 0 and 1.')
    if (len(noiseSource)<len(data)):
        raise ValueError('The length of the noise source should be longer than the audio data.')
    start = np.random.randint(len(noiseSource)-len(data))
    noise = noiseSource[start:start+len(data)]
    noise = noise_factor * noise / (np.max(np.abs(noise))+1e-6)
    audio = (1.0-noise_factor) * data / np.max(np.abs(data))
    return noise+audio

def shift(audio, shifty):
    """
    Shift the audio in time. If `shifty` is positive, shift with time
    advance; if negative, shift with time delay. Silence are padded to
    keep the duration unchanged.

    :param np.ndarray audio: Audio data points.
    :param float shifty: Shift time in millseconds. If positive, shift with
        time advance; if negative; shift with time delay.
    :raises ValueError: If shifty is longer than audio duration.
    """
    if abs(shifty) > len(audio):
        raise ValueError("Absolute value of shift_ms should be smaller "
                         "than audio duration.")
    if shifty > 0:
        # time advance
        audio[:-shifty] = audio[shifty:]
        audio[-shifty:] = 0
    elif shifty < 0:
        # time delay
        audio[-shifty:] = audio[:shifty]
        audio[:-shifty] = 0
    return audio

def resample(audio, rate, target_sample_rate, filter='kaiser_best'):
    """
    Resample the audio to a target sample rate.

    :param np.ndarray audio: Audio data points.
    :param int target_sample_rate: Target sample rate.
    :param str filter: The resampling filter to use one of {'kaiser_best',
                   'kaiser_fast'}.
    """
    audio = resampy.resample(audio, rate, target_sample_rate, filter=filter)
    return audio

def simple_echo(audio, factor, duration):
    """
    Resample the audio to a target sample rate.

    :param np.ndarray audio: Audio data points.
    :param int factor: Echo factor.
    :param int duration: Echo duration. If duration == 0, then no echo.
    """
    if duration == 0: return audio
    result = np.zeros(len(audio)+duration)
    ratio = 1
    for i in range(duration+1):
        result[i:len(audio)+i] = ratio*audio
        ratio*=factor
    return _normalize(result)

def impulse_echo(audio, impulse_func):
    """
    Resample the audio to a target sample rate.

    :param np.ndarray audio: Audio data points.
    :param np.ndarray impulse_func: Echo factors array.
    """
    result = np.zeros(len(audio)+len(impulse_func))
    result[:len(audio)] = audio
    for i in range(duration):
        result[i+1:len(audio)+i+1] = impulse_func[i]*audio
    return _normalize(result)

def _normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def _zeroPad(dat, l):
    """
    Pad by zeros.

    :param list dat: The data array.
    :param int l: Objective length.
    :raises ValueError: If l is shorter than the dat length.
    """
    if l<len(dat):
        raise ValueError("Obj length smaller than data length, cannot do the zero padding.")
    result = np.zeros(l)
    result[:len(dat)] = dat
    return result
