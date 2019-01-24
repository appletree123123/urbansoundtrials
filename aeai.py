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
import visdom
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import random
from fe import extract_features


def check_existence(FILENAME):
    if os.path.isfile(FILENAME) == False:
        return False
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






class DataSetAir(Dataset):
    
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
            wav_file = filepath + '/' + self.class_list[folder_number] + '.'  + str(file_number+1).zfill(4) + '_.wav'

        label=folder_number
        if (check_existence(wav_file)):
            sample = extract_features(wav_file,augment=False)
        else:
            wav_file = filepath + '/' + self.class_list[folder_number] + '.'  + str(file_number-700).zfill(4) + '_.wav'
            sample = extract_features(wav_file,augment=True)
        return sample, label

    def __len__(self): #return data length

        return 900*len(self.class_list)




def main():
    vis = visdom.Visdom()
    loss_window = vis.line(Y=torch.zeros((1)).cpu(),X=torch.zeros((1)).cpu(),opts=dict(xlabel='epoch',ylabel='Loss',title='64 ep,batch_norm, Adam',legend=['Loss']))

    warnings.filterwarnings("ignore")  #not to flood the output
    torch.set_printoptions(precision=10)   #to get a nice output
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #working on cuda, not on the CPU

    dtype=torch.cuda.FloatTensor

    train_transformer = transforms.ToTensor()  

    db = DataSetAir('audio',train_transformer) #initiate DataBase
    train_loader = DataLoader(dataset = db, batch_size =128, shuffle=True, num_workers=2) #put 32

    cnn = ConvNet() #Create the instanse of net 
    cnn = cnn.cuda()


    criterion = torch.nn.CrossEntropyLoss().cuda() 
    optimizer = optim.Adam(cnn.parameters(), lr = 0.0009)
    running_loss = 0
    for epoch in range(64):
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = Variable(inputs.type(dtype)), Variable(labels.type(torch.cuda.LongTensor))
            optimizer.zero_grad()             #Set the parameter gradients to zero
            outputs = cnn(inputs)
            #print('Output:',outputs,'with labels',labels)
            loss = criterion(outputs, labels)
            #print('loss_size',loss)
            loss.backward()
            optimizer.step()
            running_loss += loss
            #.data[0]
        torch.save(cnn.state_dict(), str(epoch) + '_aeai.pt')
        vis.line(X=torch.ones((1,1)).cpu()*epoch,Y=torch.Tensor([running_loss]).unsqueeze(0).cpu(),win=loss_window,update='append')
        print('Running loss was:',running_loss)
        print('Finishing Epoch #',epoch)


    print('Finished training.')
    torch.save(cnn, 'aeai_full.pt')
    torch.save(cnn.state_dict(), 'aeai.pt')
    print('Saving model..')




if __name__ == "__main__":
   main()
