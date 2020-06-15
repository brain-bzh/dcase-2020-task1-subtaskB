import librosa
import scipy.io.wavfile as wav
from scipy.signal import resample 
import numpy as np
import glob
import tqdm
import os
from joblib import Parallel, delayed
import multiprocessing
import argparse

NUM_CORES = multiprocessing.cpu_count()

parser = argparse.ArgumentParser()
parser.add_argument('--URL_audio', type=str, required=True)
parser.add_argument('--URL_out', type=str, required=True)
args = parser.parse_args()

URL_IN = args.URL_audio
URL_OUT = args.URL_out
SR_OUT = 18000 # Hz

def down_sample(i):
    file_name = audio_files[i]
    x, sr = librosa.load(file_name, sr = None, mono = False)
    x = x.T
    N = len(x)
    t = int(N/sr)
    N_resample = int(t*SR_OUT)
    x = resample(x, N_resample)
    name = os.path.basename(file_name)
    wav.write(os.path.join(URL_OUT,name), SR_OUT, np.int16(x*2**15))

if not os.path.exists(URL_OUT):
    os.makedirs(URL_OUT)
    print(f"{URL_OUT} folder created.")

audio_files = glob.glob(URL_IN +'*.wav')
print(f'kernel : {NUM_CORES}')

Parallel(n_jobs=int(NUM_CORES))(delayed(down_sample)(f)
            for f in tqdm.tqdm(range(len(audio_files))))


