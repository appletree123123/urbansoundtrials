import glob
import os 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import warnings
import torch.optim as optim
from torchvision import transforms
import wavio
from torch.autograd import Variable
from skimage import io#, transform

#https://drive.google.com/drive/folders/1XaFM8BJFligrqeQdE-_5Id0V_SubJAZe?usp=sharing
from torch.utils.data.sampler import SubsetRandomSampler


class ConvNet(nn.Module):

    #Classifying RGB images, therefore number of input channels = 3
    #We want to apply 32 feature detectors (filters), so out channels is 32
    #3x3 filter moves 1 pixel at a time
    #ReLU" all negative values become 0, all positive values remain


    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #self.conv3_drop = nn.Dropout2d(0.5)


        self.fc1 = torch.nn.Linear(12800, 2048)
        self.fc2 = torch.nn.Linear(2048, 1024)
        self.fc3 = torch.nn.Linear(1024,128)
        self.fc4 = torch.nn.Linear(128,8)
        torch.nn.init.xavier_uniform(self.conv1.weight) #initialize weights
        torch.nn.init.xavier_uniform(self.conv2.weight)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        


    def forward(self, x):
        #print('Begin forward pass. X shape:',x.shape)        
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
        #x = x.view(1, 16384)  #Rectify but you need to rectify to minibatch size poluebok
        x = x.view(x.size(0),-1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
    
        return F.softmax(x)




class DataSetAir(Dataset):
    
    def __init__(self, root_dir,transform): #download,read,transform the data
        self.root_dir = root_dir
        #self.class_list = ('air_conditioner','children_playing','dog_bark','drilling','engine_idling','jackhammer','siren','street_music')
        self.class_list = ('air_conditioner','engine_idling')
        self.transform = transform

    def __getitem__(self, index): #superfast 0(1) method, return item by index
        #Goes into the folder with the database
        #According to the class list specified in the constructor goes into every folder 
        #and loads all the images in the folder. The line "img ="
        #varies to every database
        file_number = int(index / len(self.class_list))
        folder_number = index % len(self.class_list)
        class_folder = os.path.join(self.root_dir, self.class_list[folder_number])
        for filepath in glob.iglob(class_folder):
            wav_file = filepath + '/' + self.class_list[folder_number] + '.'  + str(file_number+1).zfill(4) + '_.wav'
        
        
        #It was Float tensor before!
        #label = torch.LongTensor(1,8) #Created a label tensor
        #label = label.new_zeros(1,8) #made it zero
        label = torch.LongTensor(1)        
       
        label[0] = folder_number  #Class scores should also be in a range of 0..1
        #print(wav_file)
        #print('File#',file_number,'Folder:',folder_number) #For debug purposes
        #Feature extraction goes here:
        wav = wavio.read(wav_file)
        sample = mfcc(wav.data,wav.rate,nfft=2048) 
        sample = np.pad(sample, [(0, 800-sample.shape[0]), (0, 0)], 'constant') #All of the samples are different length, append with 0
        sample = torch.from_numpy(sample)
        sample = sample / sample.sum(0).expand_as(sample)  #normalize to range of 0:1
        sample = sample.unsqueeze(0) #Adding an empty axis because we don't work with images and there are no channels
        return sample, folder_number 


    def __len__(self): #return data length 

        return 120 #899


class DataSetAir_test(Dataset):
    #Test set
    
    def __init__(self, root_dir,transform): #download,read,transform the data
        self.root_dir = root_dir
        self.class_list = ('air_conditioner','children_playing','dog_bark','drilling','engine_idling','jackhammer','siren','street_music')
        self.transform = transform

    def __getitem__(self, index):
        file_number = int(index / len(self.class_list))
        folder_number = index % len(self.class_list)
        class_folder = os.path.join(self.root_dir, self.class_list[folder_number])
        for filepath in glob.iglob(class_folder):
            wav_file = filepath + '/' + self.class_list[folder_number] + '.' +  str(file_number+1).zfill(4) + '_.wav'
        
        
        label = torch.FloatTensor(1,8) #Created a label tensor
        label = label.new_zeros(1,8) #made it zero
        label[0,folder_number] = 1 #Put 1 in a correct coistion according to the class name
   
        print('Got file:', wav_file, 'with label:',label)
        #print('File#',file_number,'Folder:',folder_number) #For debug purposes

        wav = wavio.read(wav_file)
        sample = mfcc(wav.data,wav.rate,nfft=2048) 
        sample = np.pad(sample, [(0, 1600-sample.shape[0]), (0, 0)], 'constant') #All of the samples are different length, append with 0
        sample = torch.from_numpy(sample) #Make it to torch
        #sample = sample / sample.sum(0).expand_as(sample)  #normalize to range of 0:1
        sample = sample.unsqueeze(0) #Adding an empty access because we don't work with images and there are no channels
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
    train_loader = DataLoader(dataset = db, batch_size =4, shuffle=True, num_workers=2)

    cnn = ConvNet() #Create the instanse of net 
    cnn = cnn.cuda()


    criterion = torch.nn.CrossEntropyLoss().cuda() #tried Cross Entropy Loss
    #optimizer = optim.Adam(cnn.parameters(), lr=0.001) #Optimizer with learning rate 0.001
    optimizer = optim.SGD(cnn.parameters(), lr = 0.001, momentum=0.9)
    running_loss = 0 
    total_train_loss = 0
    for epoch in range(64):  #32 it was
        running_loss = 0
        for inputs, labels in train_loader:
            #inputs, labels = data
            inputs, labels = Variable(inputs.type(dtype)), Variable(labels.type(torch.cuda.LongTensor))
            optimizer.zero_grad()             #Set the parameter gradients to zero
            outputs = cnn(inputs)
            #print('Inputs.shape:',inputs.size())
            #print('Current output: ', outputs.unsqueeze(0), 'Target output:', labels)
            #print('Outputs:',outputs)
            #print('Labels:',labels)
            #print('Outputs.shape',outputs.size())
            #print('Labels.shape',labels.size())
            loss_size = criterion(outputs, labels) 
            #print('Loss size',loss_size,'Running loss:', running_loss)
            loss_size.backward()
            optimizer.step()   
            running_loss += loss_size.data[0]
        print('Running loss on previous cycle was:',running_loss)
        print('starting Epoch #',epoch)
        total_train_loss += loss_size.data[0]
         


    #Moving to testing:
    #running_loss = 0
    cnn.eval()
    torch.save(cnn, 'aeai.pt')
    db_test = DataSetAir_test('test', train_transformer)
    test_loader = DataLoader(dataset = db_test, shuffle=True,num_workers=2)
    n_errors = 0

    error_size = 0
    for i,data in enumerate(test_loader,0):
            inputs, labels = data
            inputs, labels = Variable(inputs.type(dtype)), Variable(labels.type(dtype))
            outputs = cnn(inputs)
            #print('Processing file#',i)
            #print('Output: ', outputs.unsqueeze(0), 'Ground truth:', labels)
            #print('----------')
            #error_size = labels - outputs.unsqueeze(0)
            #print('Error size',torch.abs(error_size))
            #print('----------')    
            hamming_score = 1 - (labels != outputs).sum() / float(labels.nelement())        
            
    print('Total amount of errors:',n_errors)
    print('Last cycle loss was:', running_loss)
            



 
    print('Done.')




if __name__ == "__main__":
   main()





