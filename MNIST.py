import torch
import torch.nn as nn
import torch.nn.functional as F



''' 

Convolution: Adds each element of an image to its local neighbors, weighted by a kernel, or
a small matrix, that helps us extract certain features (edge detection, sharpness, bluriness, etc.) from the input image. 

__init__ references nn.Module, which is the base class for all neural network modules.

1 input image channel --> output match our target of 10 labels (0 - 9)

'''

class Net(nn.Module):
    def __init__(self): 
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1) #First 2D convolutional layer outputting 32 convolutional features with a square kernel size of 3 
        self.conv2 = nn.Conv2d(32,64,3,1) #Second 2D convolutional layer taking in 32 input features andoutputting 64 convolutional features with a square kernel size of 3 
        
        #Designed to ensure adjacent pixels are either all 0s or all active with an input probability
        self.dropout1 = nn.Dropout2d(0.25) 
        self.dropout2 = nn.Dropout2d(0.5) 

        self.fc1 = nn.Linear(9216, 128) #first fully conected layer where 9216 is the number of features after flattening the 2D convolutional layers
        self.fc2 = nn.Linear(128, 10) #second fully connected layer that outputs our 10 labels 

    def forward(self, x): # x represents our data input
        x = self.conv1(x) #pass data through conv1
        x = F.relu(x) #activation function to introduce non-linearity
        
        x = self.conv2(x) #pass data through conv2
        x = F.relu(x) #activation function to introduce non-linearity
        
        x = F.max_pool2d(x, 2) #Run max pooling to reduce spatial dimensionso over x 
        x = self.dropout1(x) #pass data through dropout1 
        x = torch.flatten(x, 1) #Flatten x with start_dim = 1
        x = self.fc1(x) #pass data through fc1
        x = F.relu(x) #activation function to introduce non-linearity
        x = self.dropout2(x)
        x = self.fc2(x) 

        output = F.log_softmax(x, dim=1)
        return output 

random_data = torch.rand((1,1,28,28)) # Equates to one random 28  x 28 image

my_nn = Net()
result = my_nn(random_data)
print (result) 