import glob
import os 
import sys
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
import getopt
import datetime
import visdom

from fe import extract_features

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def check_existence(FILENAME):
    if os.path.isfile(FILENAME) == False:
        print('File', FILENAME, 'not found')
        sys.exit(2)
    else:
        return True


class ConvNet(nn.Module):


    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.BatchNorm1 = nn.BatchNorm2d(64)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


        self.fc1 = torch.nn.Linear(8192*4, 64)
        self.BatchNorm2 = nn.BatchNorm1d(64)
        self.fc2 = torch.nn.Linear(64, 4)
        #torch.nn.init.xavier_uniform(self.conv1.weight) #initialize weights
        #torch.nn.init.xavier_uniform(self.conv2.weight)
        #torch.nn.init.xavier_uniform(self.conv3.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x.cuda()))
        x = self.pool1(x)
        #print('Conv1 layer: X shape:',x.shape)
        x = F.relu(self.conv2(x.cuda()))
        x = self.pool2(x)
        #print('Conv2 layer: X shape:',x.shape)
        x = F.relu(self.conv3(self.BatchNorm1(x.cuda())))
        x = self.pool3(x)
        #print('Conv3 layer: X shape:',x.shape)
        x = x.view(x.size(0),-1)   #Rectify
        x = F.relu(self.fc1(x))
        x = self.BatchNorm2(x)
        x = self.fc2(x)
        return F.softmax(x)





def main(argv):
    #print('Starting at',datetime.datetime.now())

    wav_file=''
    try:
        opts, args = getopt.getopt(argv,"shi:",["ifile="])
    except getopt.GetoptError:
        print("passtoAI.py -i <filename.wav>")
        sys.exit(2)
    for opt,arg in opts:
        if opt == "-h":
            print("Passes a *wav file into the AI. Usage: aeai.py -i <file.jpg>. \n -s For plotting a heatmap in Visdom")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            wav_file = arg
            check_existence(wav_file)
            sample = extract_features(wav_file,False)
        elif opt == ("-s"):
            vis = visdom.Visdom()
            heat_window = vis.heatmap(sample[0].cpu(),opts=dict(title=wav_file))
    #vis.line(sample.cpu(),win=heat_window,update='append')

    warnings.filterwarnings("ignore")  #not to dlood the output
    torch.set_printoptions(precision=4)   #to get a nice output
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #working on cuda, not on the CPU
    dtype=torch.cuda.FloatTensor
    train_transformer = transforms.ToTensor()  
    cnn = ConvNet() #Create the instanse of net 
    cnn = cnn.cuda()

    class_list = ('dog_bark','engine_idling','children_playing','siren')
    class_list_pretty = ('DOG','CAR','CHILDREN','SIREN')
    #print('Loading model...')
    cnn.load_state_dict(torch.load('_aeai.pt'))
    #cnn=torch.load('aeai.pt')
    cnn.eval()
    output_table = BeautifulTable()
    output_table.column_headers = ["Guessed class","Probability"]
    inputs = sample
    inputs = Variable(inputs.type(dtype))
    inputs = inputs.unsqueeze(0) #Pretend that we have a batch_size of 1
    outputs = cnn(inputs)
    #print('Forward pass complete')
    value,index = torch.max(outputs,1)
    #output_table.append_row([class_list[index], value.data.cpu().numpy()[0]])
    #print(output_table)
    res = class_list_pretty[index]
    print('AT ' + str(datetime.datetime.now()) + ' DETECTED ' + res)


if __name__ == "__main__":
   main(sys.argv[1:])
