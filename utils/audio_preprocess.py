
import os
from multiprocessing import Pool
import argparse
import glob
import tqdm  
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.features import spectral_magnitude,STFT
from speechbrain.processing.features import Filterbank
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('input_folder_path',help='folder with all the emotion wav segregrated in terms of folders')
parser.add_argument('output_folder_path',help='folder to dump the spec feature of the trainig data')
args = parser.parse_args()

assert os.path.isdir(args.input_folder_path),'input folder path is not present check the path'
if not os.path.isdir(args.output_folder_path):
    os.makedirs(args.output_folder_path)

def process_file(data):
        '''
        extract the mel filter bank feature of the data 
        '''
        file,output_folder=data
        compute_fbanks = Filterbank(n_mels=40)
        compute_STFT = STFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400)
        
        signal = read_audio(file).unsqueeze(0) 
        STFT_data = compute_STFT(signal)
        mag = spectral_magnitude(STFT_data)
        fbanks = compute_fbanks(mag)
        np.save(os.path.join(output_folder,os.path.split(file)[-1][:-4]+".npy"),fbanks.numpy())

def process_audio(input_folder,output_folder):
    '''
    process the audio files and stores the features in folder in npy format
    '''
    print('preprocessing the data of {0} '.format(input_folder))
    list_of_files = [(i,output_folder) for i in glob.glob(os.path.join(input_folder,"*.wav"))]
    with Pool(processes=8) as pool:
        list(tqdm.tqdm(pool.map(process_file,list_of_files),total=len(list_of_files)))
    



list_of_dirs=[]    
for emotion in os.listdir(args.input_folder_path):
    output_folder = os.path.join(args.output_folder_path,emotion)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    process_audio(os.path.join(args.input_folder_path,emotion),output_folder)

