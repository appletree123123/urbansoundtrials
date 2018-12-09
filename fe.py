import glob
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np




def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


def extract_features():
    #bands = 60
    #frames = 41
    #window_size = 512 * (frames - 1)
    bands = 128
    frames = 128
    window_size = 512 * (frames - 1)
    log_specgrams = []
    fn = 'test.wav'
    sound_clip,s = librosa.load(fn)
    for (start,end) in windows(sound_clip,window_size):
       print('start:',int(start),'end',int(end))
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

    return np.array(features)




def main():
   print('Start:')
   features = extract_features()
   print(features.shape)




if __name__ == "__main__":
   main()

