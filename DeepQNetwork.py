import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from torch.utils.tensorboard.writer import SummaryWriter
import os
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, num_actions, input_dims, name, checkpoint_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], out_channels=32,
                               kernel_size=8, stride=4) 
        self.conv2 = nn.Conv2d(32, out_channels=64,
                               kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, out_channels=64,
                               kernel_size=3, stride=1) 
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        # print(f'---Input Dims: {input_dims}')
        # print(f'---FC INPUT DIMS:{fc_input_dims}---')
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, num_actions) # Define network layers

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr, alpha=0.95, eps=0.01)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(f'--- T.cuda.is_available(): {T.cuda.is_available()}---')

    def calculate_conv_output_dims(self, input_dims):
        mock_input = T.zeros(1, *input_dims)
        conv1 = self.conv1(mock_input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        return int(np.prod(conv3.size()))

    def forward(self, input_data):
        input_data = input_data.float()

        conv1 = F.relu(self.conv1(input_data))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # print(f'---conv3 shape: {conv3.size()}')
        conv3 = conv3.view(conv3.size()[0], -1) # Flattens conv3 output, note that first axis is minibatch_size
        # print(f'---conv3 shape: {conv3.size()}')
        layer1 = F.relu(self.fc1(conv3))
        layer2 = self.fc2(layer1)
        # print(f'---Layer2 shape: {layer2.size()}')
        return layer2

    def save(self): # save model parameters as a dictionary
        print("---Saving checkpoint---")
        T.save(self.state_dict(), self.checkpoint_file)
        # return filepath
    
    def load(self): 
        print("---Loading checkpoint---")
        state_dict = T.load(self.checkpoint_file)
        self.load_state_dict(state_dict)