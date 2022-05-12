# -*- coding: utf-8 -*-
"""FinalProject.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1impSr1dgGQ34n55B-U5E3DSXePjncHeN
"""

# import libraries
import torch
import numpy as np
import torch.nn as nn

from torchvision import datasets
import torchvision.transforms as transforms

# how many samples per batch to load
batch_size = 512

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

#transform=transforms.Compose([transforms.ToTensor(),
#                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                             ])

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

print(train_data[0])

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels= 1,out_channels= 16, kernel_size=5 ,stride=1 ,padding=2 ),                                                   
            nn.MaxPool2d(kernel_size=2 ),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, kernel_size=5 ,stride=1 ,padding=2  ),                           
            nn.MaxPool2d( kernel_size=2),                
        )
        # fully connected layer, output 10 classes
        self.dense1 = nn.Linear(32*7*7 , 32*7*7)
        self.out = nn.Linear(32*7*7 , 10)
    def forward(self, x):
        x= self.conv1(x)
        x= self.conv2(x)
        x= x.view(x.size(0), -1)
        x= self.dense1(x)
        output =self.out(x)
        return output, x    # return x for visualization

# initialize the NN
model_cnn = CNN()
print(model_cnn)
pcount = 0
for parameter in model_cnn.parameters():
    print(parameter.shape)

# training code
def trainCNN(model, optimizer, epochs=10):
    model.train() # prep model for training

    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for data, target in train_loader:
            data = data
            target = target
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)[0]
            #print(output, data.shape)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*data.size(0)
            
        # print training statistics 
        # calculate average loss over an epoch
        train_loss = train_loss/len(train_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch+1, 
            train_loss
            ))

# initialize lists to monitor test loss and accuracy
def testCNN(model):
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval() # prep model for *evaluation*

    for data, target in test_loader:
        data = data
        target = target
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)[0]
        # calculate the loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(data.shape[0]):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

# number of epochs to train the model
n_epochs = 10  # suggest training between 20-50 epochs
# specify optimizer
#model = Net()
print("SGD Optimizer")
optimizerSGD = torch.optim.SGD(model_cnn.parameters(), lr=0.05)
trainCNN(model_cnn, optimizerSGD)
testCNN(model_cnn)

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('/content/drive/MyDrive/handwrittenDigitNN/4-2.jpg',cv2.IMREAD_GRAYSCALE)
res = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_NEAREST)
plt.imshow(img,cmap='gray')
res.shape

plt.imshow(res,cmap='gray')

pred2 = model_cnn(torch.Tensor(res).view(1,1,28,28))[0]
print(torch.argmax(pred2))

img = cv2.imread('/content/drive/MyDrive/handwrittenDigitNN/4.jpg',cv2.IMREAD_GRAYSCALE)
res = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_NEAREST)
plt.imshow(img,cmap='gray')
res.shape
print(torch.Tensor(res).view(1,1,28,28))
pred2 = model_cnn(torch.Tensor(res).view(1,1,28,28))[0]
print(torch.argmax(pred2))
print(pred2)



!pip install hls4ml
import hls4ml

#Fetch a keras model from our example repository
#This will download our example model to your working directory and return an example configuration file
file= '/content/drive/MyDrive/handwrittenDigitNN/mnist.onnx'

#Convert it to a hls project
hls_model = hls4ml.converters.convert_from_pytorch_model(model_cnn, model_cnn.parameters(), project_name='model_cnn')

# Print full list of example models if you want to explore more
hls4ml.utils.fetch_example_list()

hls_model.build()

!sudo apt install iverilog
!pip3 install veriloggen numpy onnx
!python Setup.py install
import nngen as ng

!pip3 list

#!pip3 install onnx -U
#!pip3 install nngen -U
#!python Setup.py install
!pip3 install polymath
#!pip3 list
# import os.path
# setup = '/content/drive/MyDrive/handwrittenDigitNN/nngenSetup.py'
# file_exists = os.path.exists(setup)

# print(file_exists)

#!pip3 list

# import onnx
# from onnx import numpy_helper

# onnx_filename = 'mnist.onnx'
# model = onnx.load(onnx_filename)
# print(model.graph.output)

#!git clone --recurse-submodules https://github.com/VeriGOOD-ML/public
!pip3 install -r /content/public/tabla/requirements.txt

!python3 /content/public/axiline/axiline/run_benchmark.py --benchmark /content/mnist.onnx  --output /content/

import os.path

import nngen as ng

# data types
act_dtype = ng.int16
weight_dtype = ng.int16
bias_dtype = ng.int16
scale_dtype = ng.int16
disable_fusion = False

# Pytorch to ONNX
onnx_filename = 'mnist2.onnx'
dummy_input = torch.Tensor(res).view(1,1,28,28)
input_names = [ "input_%d" % i for i in range(5) ]
output_names = [ "output" ]
model_mlp_Relu.eval()
torch.onnx.export(model_mlp_Relu, dummy_input, onnx_filename, input_names=input_names, output_names=output_names)

file_exists = os.path.exists(onnx_filename)
print(onnx_filename)
print(file_exists)


# ONNX to NNgen
dtypes = {}
(outputs, placeholders, variables,
 constants, operators) = ng.from_onnx(onnx_filename,
                                      value_dtypes=dtypes,
                                      default_placeholder_dtype=act_dtype,
                                      default_variable_dtype=weight_dtype,
                                      default_constant_dtype=weight_dtype,
                                      default_operator_dtype=act_dtype,
                                      default_scale_dtype=scale_dtype,
                                      default_bias_dtype=bias_dtype,
                                      disable_fusion=disable_fusion)