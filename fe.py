import glob
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch



def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


def extract_features(fn='test.wav'):
#Takes a wav file and returns a 2x128x128 tensor ready for uploading into AI
        bands = 128
        frames = 128
        window_size = 512 * (frames - 1)
        log_specgrams = []
        
        sound_clip,s = librosa.load(fn)
        if len(sound_clip)<88200:
            sound_clip = np.pad(sound_clip,(0,88200-len(sound_clip)),'constant') #Pad with zeroes to the universal length
        
        for (start,end) in windows(sound_clip,window_size):
            s = int(start)
            e = int(end)
            if(len(sound_clip[s:e]) == window_size):
                signal = sound_clip[s:e]
                melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                logspec = librosa.amplitude_to_db(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)
        log_specgrams = np.asarray(log_specgrams)
        log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
        features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
        for i in range(len(features)):
            features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
        features = np.array(features)
        features = np.swapaxes(features,1,3) #channels go first so we need to swap axis 1 and 3
        sample = torch.from_numpy(features)
        sample = sample.squeeze(0)
        return sample




