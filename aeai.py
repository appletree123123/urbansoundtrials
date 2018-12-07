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
import wavio
from torch.autograd import Variable

from torch.utils.data.sampler import SubsetRandomSampler


class ConvNet(nn.Module):



    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=80, kernel_size=2, stride=2, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #self.conv12_drop = nn.Dropout2d(0.25)

        self.conv2 = torch.nn.Conv2d(in_channels=80, out_channels=160, kernel_size=3, stride=3, padding=1)

        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(7920*2, 4096)
        self.fc2 = torch.nn.Linear(4096,2048)
        self.fc3 = torch.nn.Linear(2048,4)
        torch.nn.init.xavier_uniform(self.conv1.weight) #initialize weights
        torch.nn.init.xavier_uniform(self.conv2.weight)


    def forward(self, x):
        #print('Begin forward pass, x shape:', x.shape)
        x = F.relu(self.conv1(x.cuda()))
        #print('First convolution complete, x.shape',x.shape)
        x = self.pool1(x)
        #print('First pooling complete,x.shape:',x.shape)
        x = F.relu(self.conv2(x.cuda()))
        x = self.pool2(x)

        x = x.view(x.size(0),-1) 
        #print('Shape after rectifying',x.shape)
        x = F.relu(self.fc1(x))
        #print('Shape after fc1',x.shape)
        x = F.relu(self.fc2(x))
        #print('Shape after fc2',x.shape)
        x = self.fc3(x)
        #print('Before softmax',x.shape)
    
        return F.softmax(x)




class DataSetAir(Dataset):
    
    def __init__(self, root_dir,transform): #download,read,transform the data
        self.root_dir = root_dir
        #self.class_list = ('air_conditioner','children_playing','dog_bark','drilling','engine_idling','jackhammer','siren','street_music')
        self.class_list = ('air_conditioner','children_playing','jackhammer','street_music')
        self.transform = transform

    def __getitem__(self, index): 
        file_number = int(index / len(self.class_list))
        folder_number = index % len(self.class_list)
        class_folder = os.path.join(self.root_dir, self.class_list[folder_number])
        for filepath in glob.iglob(class_folder):
            wav_file = filepath + '/' + self.class_list[folder_number] + '.'  + str(file_number+1).zfill(4) + '_.wav'

        label = folder_number  
        #print(wav_file)
        #print('File#',file_number,'Folder:',folder_number) #For debug purposes
        #Feature extraction goes here:
        wav = wavio.read(wav_file)
        mfcc_cf = mfcc(wav.data,wav.rate,winlen=0.072,numcep=26,nfft=4000) 
        d_mfcc = delta(mfcc_cf,2)  #calculate delta mfcc
        dd_mfcc = delta(d_mfcc,2)
        sample = np.concatenate((mfcc_cf,d_mfcc),axis=1) #append delta to regular mfcc  
        sample = np.concatenate((sample,dd_mfcc),axis=1) #delta-delta
        sample = np.pad(sample, [(0, 800-sample.shape[0]), (0, 0)], 'constant') #All of the samples are different length, append with 0
        sample = torch.from_numpy(sample)
        sample = sample / sample.sum(0).expand_as(sample)  #normalize to range of 0:1
        sample = sample.unsqueeze(0) #Adding an empty axis because we don't work with images and there are no channels
        return sample, label


    def __len__(self): #return data length 

        return 890*len(self.class_list) #899


class DataSetAir_test(Dataset):
    #Test set
    
    def __init__(self, root_dir,transform): #download,read,transform the data
        self.root_dir = root_dir
        #self.class_list = ('air_conditioner','children_playing','dog_bark','drilling','engine_idling','jackhammer','siren','street_music')
        self.class_list = ('air_conditioner','children_playing','jackhammer','street_music')
        self.transform = transform

    def __getitem__(self, index):
        file_number = int(index / len(self.class_list))
        folder_number = index % len(self.class_list)
        class_folder = os.path.join(self.root_dir, self.class_list[folder_number])
        for filepath in glob.iglob(class_folder):
            wav_file = filepath + '/' + self.class_list[folder_number] + '.' +  str(file_number+1).zfill(4) + '_.wav'
        
        
        label = folder_number
        wav = wavio.read(wav_file)
        mfcc_cf = mfcc(wav.data,wav.rate,winlen=0.072,numcep=26,nfft=4000) 
        d_mfcc = delta(mfcc_cf,2)  #calculate delta mfcc
        dd_mfcc = delta(d_mfcc,2)
        sample = np.concatenate((mfcc_cf,d_mfcc),axis=1) #append delta to regular mfcc  
        sample = np.concatenate((sample,dd_mfcc),axis=1) #delta-delta
        sample = np.pad(sample, [(0, 800-sample.shape[0]), (0, 0)], 'constant') #All of the samples are different length, append with 0
        sample = torch.from_numpy(sample)
        sample = sample / sample.sum(0).expand_as(sample)  #normalize to range of 0:1
        sample = sample.unsqueeze(0) #Adding an empty axis because we don't work with images and there are no channels
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
    train_loader = DataLoader(dataset = db, batch_size =64, shuffle=True, num_workers=2)

    cnn = ConvNet() #Create the instanse of net 
    cnn = cnn.cuda()


    criterion = torch.nn.CrossEntropyLoss().cuda() #tried Cross Entropy Loss
    #optimizer = optim.Adam(cnn.parameters(), lr=0.001) #Optimizer with learning rate 0.001
    optimizer = optim.SGD(cnn.parameters(), lr = 0.01, momentum=0.9)
    running_loss = 0 
    total_train_loss = 0
    for epoch in range(32):  #32 it was
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = Variable(inputs.type(dtype)), Variable(labels.type(torch.cuda.LongTensor))
            optimizer.zero_grad()             #Set the parameter gradients to zero
            outputs = cnn(inputs)
            loss_size = criterion(outputs, labels) 
            loss_size.backward()
            optimizer.step()   
            running_loss += loss_size.data[0]
        print('Running loss was:',running_loss)
        print('Finishing Epoch #',epoch)
        total_train_loss += loss_size.data[0]
         


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
            print('Outputs=',outputs)
            value,index = torch.max(outputs,1)
            print('Output:', index, 'Ground truth:', labels)
            if (index!=labels):
                n_errors = n_errors+1
                
            
    print('Total amount of errors:',n_errors)
    print('Accuraccy:',n_errors/i-1)


 
    print('Done.')




if __name__ == "__main__":
   main()
