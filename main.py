import torch
import argparse
import torch.nn.functional as F
import yaml
import subprocess
import numpy as np
import sys
import pandas as pd
import csv
import os
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
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
        super(Network, self).__init__() #What is this for
        self.conv1 = torch.nn.Conv2d(128,128,3,1,1)
        self.fc1   = torch.nn.Linear(2048,512)
        self.dropout = torch.nn.Dropout(0.25)
        self.threshold = 1
        self.leak = 0.9
        self.timesteps = timesteps
    def forward(self, input, layer_type):
        #log_conv_out = torch.nn.parameter.Parameter(torch.zeros(self.timesteps, 128, 32, 32).cuda(), requires_grad=False)
        if layer_type == 'linear':
            membrane = torch.nn.parameter.Parameter(torch.zeros(1, 512).cuda(), requires_grad=False)
            output = torch.nn.parameter.Parameter(torch.zeros(input.size(0), self.timesteps, 512).cuda(), requires_grad=False)
            for t in range(self.timesteps):
                membrane = membrane + self.fc1(input[:,t,:])
                output[:,t,:],  membrane =  LIF_sNeuron(membrane, self.threshold, self.leak)
        elif layer_type == 'conv':        
            membrane = torch.nn.parameter.Parameter(torch.zeros(1, 128, 32, 32).cuda(), requires_grad=False)      
            output = torch.nn.parameter.Parameter(torch.zeros(input.size(0), self.timesteps, 128, 32, 32).cuda(), requires_grad=False)
            for t in range(self.timesteps):
                membrane = membrane + self.conv1(input[:,t,:,:,:])
                output[:,t,:],  membrane =  LIF_sNeuron(membrane, self.threshold, self.leak)
        return output

def main():

    parser = argparse.ArgumentParser(description='PyTorch Sparsity Example')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--timesteps', type=int, default=100, metavar='S',
                        help='Number of timesteps (default: 100)')  
    parser.add_argument('--batch_size', type=int, default=1, metavar='S',
                        help='Number of batches (default: 1)')   
    parser.add_argument('--output', type=str, default="output", metavar='S',
                        help='Output name (default: output)') 
    parser.add_argument('--layer_type', type=str, default="conv", metavar='S',
                        help='Layer type (default: conv)')                                                        
    args = parser.parse_args()  
    device = 'cuda'                      
    torch.manual_seed(args.seed)
    timesteps = args.timesteps
    batch_size = args.batch_size
    layer_type = args.layer_type    
    model = Network(timesteps).to(device)
################### Loading PLIF parameters ###################
    plif_directory = os.path.join(os.getcwd()+'/plif_checkpoint')
    check_point_max_path = os.path.join(plif_directory, 'check_point_max.pt')
    check_point = None
    if os.path.exists(plif_directory):
        check_point = torch.load(check_point_max_path, map_location=device)
        plif_net = check_point['net']
        with torch.no_grad():
            if layer_type == 'conv':
                model.conv1.weight = plif_net.conv[4].weight   
            elif layer_type == 'linear':
                model.fc1.weight = plif_net.fc[2].weight   
    else:
        with torch.no_grad():
            model.conv1.weight.div_(torch.norm(model.conv1.weight, dim=2, keepdim=True))
         
###############################################################
    hist = torch.histc(model.fc1.weight.cpu().detach(), bins = 100)
    # max = np.max(model.conv1.weight.cpu().detach().numpy())
    # min = np.min(model.conv1.weight.cpu().detach().numpy())
    # x = np.linspace(min, max, 100)
    # # sns.histplot(y=hist, bins=100, kde=True)
    # plt.ylim(-2, 8000)
    # plt.bar(x, hist, align='center', color='#fdae6b')
    # plt.yscale("log")
    # tmp = [] 
    # tmp.append([i/100 for i in range(-100,100)])
    # tmp = tmp[0]
    x =range(100)
    plt.bar(x, hist, align='center', color='#fdae6b')
    plt.savefig("weight_dist.png", dpi=300)
    plt.tight_layout()
    plt.show()
    weight_density = [100,80,60,50,25,10]
    activation_density = [50,40,30,20,10,5,1,0.1]

    out_density = []
    in_density = []
    w_density = []
    std = []
    conv_og_weight = model.conv1.weight
    fc_og_weight = model.fc1.weight
    conv_input  = torch.ones(batch_size,timesteps,128,32,32).to(device='cuda')
    dense_input = torch.ones(batch_size,timesteps,2048).to(device='cuda')
    if layer_type == 'linear':
        input = dense_input
    elif layer_type == 'conv':
        input = conv_input
    for j in weight_density:
        drop_weight= torch.nn.Dropout(1-j/100)
        #Weight initialization for different layer types
        if layer_type == 'linear':
            model.fc1.weight = torch.nn.parameter.Parameter(drop_weight(model.fc1.weight)) 
        elif layer_type == 'conv':        
            model.conv1.weight = torch.nn.parameter.Parameter(drop_weight(model.conv1.weight))   

        for i in activation_density:
            drop_input = torch.nn.Dropout(1-i/100)
            x = drop_input(input).bool()
            x = x*1.0
            x.to(device)
            output = model(x,layer_type)
            w_density.append(round(j/100,3))
            in_density.append(round(np.count_nonzero(x.cpu().detach().numpy()) / torch.numel(x),3))
            out_density.append(round(np.count_nonzero(output.cpu().detach().numpy()) / torch.numel(output),4))
        # reseting the model weight for the next iteration   
        if layer_type == 'linear':
            model.fc1.weight = torch.nn.parameter.Parameter(fc_og_weight)    
        elif layer_type == 'conv':          
            model.conv1.weight = torch.nn.parameter.Parameter(conv_og_weight) 

        


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
        layer["name"].append("w" + str(data.iloc[i]['weight_density'])+ '_' + "i" + str(data.iloc[i]['input_density']) + '_' + "o" + str(data.iloc[i]['output_density'])+"_C_128")
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
        layer['T'].append(100)

    '''Converting initialized dictionary into Pandas data frame'''
    data = pd.DataFrame.from_dict(layer)
    if layer_type == 'linear':
        data.to_csv('activation_weight_sparsity_linear_multi_layer.csv', sep=',', index=False, mode='a')
    elif layer_type == 'conv':
        data.to_csv('activation_weight_sparsity_conv_multi_layer.csv', sep=',', index=False, mode='a')

if __name__ == '__main__':
    main()    
    # np.count_nonzero(model.conv1.weight.cpu().detach().numpy()) / torch.numel(model.conv1.weight)
