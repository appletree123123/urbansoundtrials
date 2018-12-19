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
from beautifultable import BeautifulTable

from fe import extract_features

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
        self.fc2 = torch.nn.Linear(64, 4)
        
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

        return F.softmax(x)

class DataSetAir_test(Dataset):
    #Test set
    
    def __init__(self, root_dir,transform): #download,read,transform the data
        self.root_dir = root_dir
        #self.class_list = ('air_conditioner','children_playing','dog_bark','drilling','engine_idling','jackhammer','siren','street_music')
        self.class_list = ('dog_bark','engine_idling','children_playing','siren')
        self.transform = transform

    def __getitem__(self, index):
        file_number = int(index / len(self.class_list))
        folder_number = index % len(self.class_list)
        class_folder = os.path.join(self.root_dir, self.class_list[folder_number])
        for filepath in glob.iglob(class_folder):
            wav_file = filepath + '/' + self.class_list[folder_number] + '.' +  str(file_number+1).zfill(4) + '_.wav'

        label = folder_number
        sample = extract_features(wav_file)
        return sample, label

     
    def __len__(self): #return data length 

        return 83*len(self.class_list)


def main():

    warnings.filterwarnings("ignore")  #not to dlood the output
    torch.set_printoptions(precision=4)   #to get a nice output
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #working on cuda, not on the CPU

    dtype=torch.cuda.FloatTensor

    train_transformer = transforms.ToTensor()  


    cnn = ConvNet() #Create the instanse of net 
    cnn = cnn.cuda()

    class_list = ('dog_bark','engine_idling','children_playing','siren')
    print('Loading model...')
    
    cnn.load_state_dict(torch.load('aeai.pt'))
    #cnn=torch.load('aeai.pt')
    cnn.eval()
    db_test = DataSetAir_test('test', train_transformer)
    test_loader = DataLoader(dataset = db_test, shuffle=True,num_workers=2)
    n_errors = 0
    i = 0
    output_table = BeautifulTable()
    output_table.column_headers = ["N","Guessed class","Probability","Real class"]
    for inputs, labels in test_loader:
            inputs, labels = Variable(inputs.type(dtype)), Variable(labels.type(torch.cuda.LongTensor))
            outputs = cnn(inputs)
            i=i+1
            value,index = torch.max(outputs,1)
            output_table.append_row([i, class_list[index], value.data.cpu().numpy()[0], class_list[labels.data.cpu().numpy()[0]]])
            #print('N',i,'Got class:   ', class_list[index], 'Real class:   ', class_list[labels.data.cpu().numpy()[0]])
            if (index!=labels):
                n_errors=n_errors+1
                #print('Found error in class',class_list[labels.data.cpu().numpy()[0]])
    print(output_table)
    print('Total amount of errors:',n_errors)
    print('Accuraccy:',100*(1-n_errors/i),'%')

    print('Done.')




if __name__ == "__main__":
   main()
