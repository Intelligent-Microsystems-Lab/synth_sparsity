import torch
import argparse
import torch.nn.functional as F
import yaml
import subprocess
import numpy as np
import sys
import pandas as pd
import csv

import itertools
class SpikingNN(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return input.gt(0).type(torch.cuda.FloatTensor)
    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0.0] = 0
        return grad_input
def LIF_sNeuron(membrane_potential, threshold, l):
    # check exceed membrane potential and reset
    ex_membrane = F.threshold(membrane_potential, threshold, 0)
    membrane_potential = membrane_potential - ex_membrane # hard reset
    # generate spike
    out = SpikingNN.apply(ex_membrane)
    # decay
    membrane_potential = l * membrane_potential.detach()
    out = out.detach() + torch.div(out, threshold) - torch.div(out, threshold).detach()

    return out , membrane_potential
class Network(torch.nn.Module):
    def __init__(self,timesteps):
        super(Network, self).__init__() #What this is for really???!
        self.conv1 = torch.nn.Conv2d(128,128,3,1,1)
        self.fc1   = torch.nn.Linear(2048,512)
        self.dropout = torch.nn.Dropout(0.25)
        self.threshold = 1
        self.leak = 0.9
        self.timesteps = timesteps
    def forward(self, input):
        membrane = torch.nn.parameter.Parameter(torch.zeros(1, 128, 32, 32).cuda(), requires_grad=False)
        output = torch.nn.parameter.Parameter(torch.zeros(input.size(0), self.timesteps, 128, 32, 32).cuda(), requires_grad=False)
        for t in range(self.timesteps):
            membrane = membrane + self.conv1(input[:,t,:,:,:])
            output[:,t,:,:,:],  membrane=  LIF_sNeuron(membrane, self.threshold, self.leak)
        return output

def main():

    parser = argparse.ArgumentParser(description='PyTorch Sparsity Example')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--timesteps', type=int, default=100, metavar='S',
                        help='Number of timesteps (default: 100)')  
    parser.add_argument('--batch_size', type=int, default=1, metavar='S',
                        help='Number of batches (default: 1)')                                                
    args = parser.parse_args()  
    device = 'cuda'                      
    torch.manual_seed(args.seed)
    timesteps = args.timesteps
    batch_size = args.batch_size
    model = Network(timesteps).to(device)
    
    weight_density = [100,80,60,50,20,10,5]
    # weight_density = [1]
    out_density = []
    in_density = []
    w_density = []
    with torch.no_grad():
        model.conv1.weight.div_(torch.norm(model.conv1.weight, dim=2, keepdim=True))
    og_weight = model.conv1.weight
    input = torch.ones(batch_size,timesteps,128,32,32).to(device='cuda')

    for j in weight_density:
        drop_weight= torch.nn.Dropout(1-j/100)
        model.conv1.weight = torch.nn.parameter.Parameter(drop_weight(model.conv1.weight))   
        for i in range (1,100):
            drop_input = torch.nn.Dropout(1-i/100)
            x = drop_input(input).bool()
            x = x*1.0
            output = model(x)
            w_density.append(round(j/100,3))
            in_density.append(round(np.count_nonzero(x.cpu().detach().numpy()) / torch.numel(x),3))
            out_density.append(round(np.count_nonzero(output.cpu().detach().numpy()) / torch.numel(output),3))
        model.conv1.weight = torch.nn.parameter.Parameter(og_weight)    
    
    data = pd.DataFrame(
    {'weight_density': w_density,
     'input_density': in_density,
     'output_density': out_density
    })
    '''Copying the headers for layer description into a list'''
    network = pd.read_csv("workload.csv")
    header = []
    for row in network:
        header.append(row)
        if row == 'WS':
            break
    ''' Initializing keys of a dictionary to add the values later.
        It is conviniet to genrate csv file with the layer specification permutation.
        '''
    layer = {key: [] for key in header}
    '''Defining the permutation specifiacations'''
    for i in range(len(data)):
        layer["name"].append("w" + str(data.iloc[i]['weight_density'])+ '_' + "i" + str(data.iloc[i]['input_density']) + '_' + "o" + str(data.iloc[i]['output_density']))
        layer['C'].append(128)
        layer['M'].append(128)
        layer['P'].append(32)
        layer['Q'].append(32)                    
        layer['R'].append(3)
        layer['S'].append(3)
        layer['HS'].append(1)
        layer['WS'].append(1)
        layer['weights'].append(data.iloc[i]['weight_density'])
        layer['inputs'].append(data.iloc[i]['input_density'])
        layer['outputs'].append(data.iloc[i]['output_density'])
        layer['T'].append(20)

    '''Converting initialized dictionary into Pandas data frame'''
    data = pd.DataFrame.from_dict(layer)
    data.to_csv('layersweep.csv', sep=',', index=False)


if __name__ == '__main__':
    main()    
    # np.count_nonzero(model.conv1.weight.cpu().detach().numpy()) / torch.numel(model.conv1.weight)
