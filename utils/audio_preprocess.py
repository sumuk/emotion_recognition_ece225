
import os
from multiprocessing import Pool
import argparse
import glob
import tqdm
import numpy as np
from scipy.ndimage.morphology import binary_dilation
from pathlib import Path
from typing import Optional, Union
import webrtcvad
import librosa
import struct

parser=argparse.ArgumentParser()
parser.add_argument('input_folder_path',help='folder with all the emotion wav segregrated in terms of folders')
parser.add_argument('output_folder_path',help='folder to dump the spec feature of the trainig data')
args = parser.parse_args()

assert os.path.isdir(args.input_folder_path),'input folder path is not present check the path'
if not os.path.isdir(args.output_folder_path):
    os.makedirs(args.output_folder_path)

## Mel-filterbank
mel_window_length = 25  # In milliseconds
mel_window_step = 10    # In milliseconds
mel_n_channels = 40


## Audio
sampling_rate = 16000
# Number of spectrogram frames in a partial utterance
partials_n_frames = 160     # 1600 ms


## Voice Activation Detection
# Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# This sets the granularity of the VAD. Should not need to be changed.
vad_window_length = 30  # In milliseconds
# Number of frames to average together when performing the moving average smoothing.
# The larger this value, the larger the VAD variations must be to not get smoothed out. 
vad_moving_average_width = 8
# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 6


## Audio volume normalization
audio_norm_target_dBFS = -30


## Model parameters
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3
int16_max = (2 ** 15) - 1


def process_file(data,max_value=300):
        '''
        extract the mel filter bank feature of the data 
        '''
        file,output_folder=data
        wav = preprocess_wav(file)
        data = wav_to_mel_spectrogram(wav)
        if data.shape[0]<=max_value:
            data=np.expand_dims(data,axis=0)
            app_len =max_value%data.shape[1]
            if max_value//data.shape[1]>0:
                data = np.concatenate([data for _ in range(max_value//data.shape[1])],axis=1)
            data = np.concatenate([data,data[:,:app_len,:]],axis=1)
            np.save(os.path.join(output_folder,os.path.split(file)[-1][:-4]+".npy"),data)

def process_audio(input_folder,output_folder):
    '''
    process the audio files and stores the features in folder in npy format
    '''
    print('preprocessing the data of {0} '.format(input_folder))
    list_of_files = [(i,output_folder) for i in glob.glob(os.path.join(input_folder,"*.wav"))]
    with Pool(processes=8) as pool:
        list(tqdm.tqdm(pool.map(process_file,list_of_files),total=len(list_of_files)))

def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray], source_sr: Optional[int]=None):
    """
    Applies preprocessing operations to a waveform either on disk or in memory such that  
    The waveform will be resampled to match the data hyperparameters.
    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform'speaker sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    
    # Resample the wav
    if source_sr is not None:
        wav = librosa.resample(wav, source_sr, sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav)
    
    return wav


def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        wav,
        sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T


def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.
    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20)) 




list_of_dirs=[]    
for emotion in os.listdir(args.input_folder_path):
    output_folder = os.path.join(args.output_folder_path,emotion)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    process_audio(os.path.join(args.input_folder_path,emotion),output_folder)

