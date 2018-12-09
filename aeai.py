import glob
import os 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from python_speech_features import mfcc
from python_speech_features import delta
import scipy.io.wavfile as wav
import warnings
import torch.optim as optim
from torchvision import transforms
import librosa
from torch.autograd import Variable

from torch.utils.data.sampler import SubsetRandomSampler

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)



class ConvNet(nn.Module):

    
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3_drop = nn.Dropout2d(0.5)


        self.fc1 = torch.nn.Linear(8192*4, 64)
        self.fc2 = torch.nn.Linear(64, 1)
        torch.nn.init.xavier_uniform(self.conv1.weight) #initialize weights
        torch.nn.init.xavier_uniform(self.conv2.weight)
        torch.nn.init.xavier_uniform(self.conv3.weight)
      
    def forward(self, x):
        x = F.relu(self.conv1(x.cuda()))
        x = self.pool1(x)
        #print('Conv1 layer: X shape:',x.shape)
        x = F.relu(self.conv2(x.cuda()))
        x = self.pool2(x)
        #print('Conv2 layer: X shape:',x.shape)        
        x = F.relu(self.conv3(x.cuda()))
        x = self.pool3(x)
        #print('Conv3 layer: X shape:',x.shape)    
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0),-1)   #Rectify 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.sigmoid(x)





class DataSetAir(Dataset):
    
    def __init__(self, root_dir,transform): #download,read,transform the data
        self.root_dir = root_dir
        #self.class_list = ('air_conditioner','children_playing','dog_bark','drilling','engine_idling','jackhammer','siren','street_music')
        self.class_list = ('air_conditioner','children_playing')
        self.transform = transform

    def __getitem__(self, index): 
        file_number = int(index / len(self.class_list))
        folder_number = index % len(self.class_list)
        class_folder = os.path.join(self.root_dir, self.class_list[folder_number])
        for filepath in glob.iglob(class_folder):
            wav_file = filepath + '/' + self.class_list[folder_number] + '.'  + str(file_number+1).zfill(4) + '_.wav'


        label = folder_number
        bands = 128
        frames = 128
        window_size = 512 * (frames - 1)
        log_specgrams = []
        fn = wav_file
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
        return sample, label

    def __len__(self): #return data length 

        return 800*len(self.class_list) #899


class DataSetAir_test(Dataset):
    #Test set
    
    def __init__(self, root_dir,transform): #download,read,transform the data
        self.root_dir = root_dir
        #self.class_list = ('air_conditioner','children_playing','dog_bark','drilling','engine_idling','jackhammer','siren','street_music')
        self.class_list = ('air_conditioner','children_playing')
        self.transform = transform

    def __getitem__(self, index):
        file_number = int(index / len(self.class_list))
        folder_number = index % len(self.class_list)
        class_folder = os.path.join(self.root_dir, self.class_list[folder_number])
        for filepath in glob.iglob(class_folder):
            wav_file = filepath + '/' + self.class_list[folder_number] + '.' +  str(file_number+1).zfill(4) + '_.wav'
        
        
        label = folder_number
        bands = 128
        frames = 128
        window_size = 512 * (frames - 1)
        log_specgrams = []
        fn = wav_file
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

        return sample, label
     
    def __len__(self): #return data length 

        return 92


def main():

    warnings.filterwarnings("ignore")  #not to dlood the output
    torch.set_printoptions(precision=10)   #to get a nice output
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #working on cuda, not on the CPU

    dtype=torch.cuda.FloatTensor

    train_transformer = transforms.ToTensor()  

    db = DataSetAir('audio',train_transformer) #initiate DataBase
    train_loader = DataLoader(dataset = db, batch_size =32, shuffle=True, num_workers=2)

    cnn = ConvNet() #Create the instanse of net 
    cnn = cnn.cuda()


    criterion = torch.nn.BCELoss().cuda() #tried Cross Entropy Loss
    optimizer = optim.Adam(cnn.parameters(), lr=0.001) #Optimizer with learning rate 0.001
    #optimizer = optim.SGD(cnn.parameters(), lr = 0.01, momentum=0.9)
    running_loss = 0
    for epoch in range(32):  #32 it was
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = Variable(inputs.type(dtype)), Variable(labels.type(dtype))
            optimizer.zero_grad()             #Set the parameter gradients to zero
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            #print('loss_size',loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
        print('Running loss was:',running_loss)
        print('Finishing Epoch #',epoch)


    #Moving to testing:
    #running_loss = 0
    cnn.eval()
    torch.save(cnn, 'aeai.pt')
    db_test = DataSetAir_test('test', train_transformer)
    test_loader = DataLoader(dataset = db_test, shuffle=True,num_workers=2)
    n_errors = 0
    i = 0
    for inputs, labels in test_loader:
            inputs, labels = Variable(inputs.type(dtype)), Variable(labels.type(torch.cuda.LongTensor))
            outputs = cnn(inputs)
            i=i+1
            #print('Outputs=',outputs)
            value,index = torch.max(outputs,1)
            print('Output:', index, 'Ground truth:', labels)
            if (index!=labels):
                n_errors = n_errors+1
                
            
    print('Total amount of errors:',n_errors)
    print('Accuraccy:',1-n_errors/i)


 
    print('Done.')




if __name__ == "__main__":
   main()
