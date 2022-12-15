import torch.nn as nn
import torch

class CNBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(CNBlock, self).__init__()

        self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding

        self.conv1 = nn.Conv1d(in_channels= self.in_channels, out_channels= 32,kernel_size= 2, stride= 2,padding= 0)
        self.conv2 = nn.Conv1d(in_channels= self.in_channels, out_channels= 32,kernel_size= 4, stride= 2,padding= 1)
        self.conv3 = nn.Conv1d(in_channels= self.in_channels, out_channels= 32,kernel_size= 8, stride= 2,padding= 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1,x2,x3], dim=1)
        return x

class ChronoNet(nn.Module):
    def __init__(self):
        super().__init__()

        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding

        self.block1 = CNBlock(22)
        self.block2 = CNBlock(96)
        self.block3 = CNBlock(96)

        self.gru1 = nn.GRU(input_size= 96, hidden_size=32, batch_first = True)
        self.gru2 = nn.GRU(input_size= 32, hidden_size=32, batch_first = True)
        self.gru3 = nn.GRU(input_size= 64, hidden_size=32, batch_first = True)
        self.gru4 = nn.GRU(input_size= 96, hidden_size=32, batch_first = True)
        self.gru_linear = nn.Linear(1875,1)
        self.flatten = nn.Flatten()
        self.fcl = nn.Linear(32,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.permute(0,2,1)

        gru_out1,_ = self.gru1(x)
        gru_out2,_ = self.gru2(gru_out1)
        gru_out = torch.cat([gru_out1,gru_out2], dim=2)
        gru_out3,_ =self.gru3(gru_out)
        gru_out = torch.cat([gru_out1,gru_out2,gru_out3],dim=2)
        
        linear_out=self.relu(self.gru_linear(gru_out.permute(0,2,1)))
        gru_out4,_=self.gru4(linear_out.permute(0,2,1))
        x=self.flatten(gru_out4)
        x = self.fcl(x)
        return x
