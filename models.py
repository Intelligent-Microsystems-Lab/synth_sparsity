import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, layer, surrogate, accelerating
from spikingjelly.clock_driven.neuron import BaseNode, LIFNode
from torchvision import transforms
import math
import csv
import numpy as np


import time
import os
from memory_profiler import profile
class PLIFNode(BaseNode):
    def __init__(self, init_tau=2.0, v_threshold=1.0, v_reset=0.0, detach_reset=True, surrogate_function=surrogate.ATan(), monitor_state=False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, monitor_state)
        init_w = - math.log(init_tau - 1)
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))
        #init_drop_threshold = 5 #this is hardcoded
        #self.threshold = nn.Parameter(torch.tensor(init_drop_threshold, dtype=torch.float))
    def forward(self, dv: torch.Tensor):
        if self.v_reset is None:
            # self.v += dv - self.v * self.w.sigmoid()
            self.v += (dv - self.v) * self.w.sigmoid()
        else:
            # self.v += dv - (self.v - self.v_reset) * self.w.sigmoid()
            self.v += (dv - (self.v - self.v_reset)) * self.w.sigmoid()
        spike = self.spiking()
        # if self.threshold != 5:
        #     print(self.threshold)
        # for b in range(spike.shape[0]):                                  
        #     for c in range(spike.shape[1]):
        #         #A tensor of a single channel of one batch over the whole inference time
        #         single_channel = spike[b,c,:,:,:].reshape(spike.shape[-1], input_conv2[0].shape[2],input_conv2[0].shape[3]).int()
        #         window_size = 2
        #         bit_wise_and = 0
        #         IoUT = []
        #         for t in range(single_channel.shape[0]):
        #             if t < 1448:
        #                 sum = 0
        #                 for index_inside_window in range(window_size):
        #                     sum += single_channel[t + index_inside_window]
        #                 intersection_mask = torch.bitwise_and( single_channel[t], single_channel[t + 1])
        #                 # intersection_mask = torch.bitwise_and( intersection_mask, single_channel[t + 3])
        #                 intersection = np.count_nonzero(intersection_mask.cpu().detach().numpy())
        #                 union = np.count_nonzero(sum.cpu().detach().numpy())
        #                 if union !=0:
        #                     IoUT.append(float(intersection/union))
        #         mIoUT = np.mean(IoUT)
        #         print("For C: " + str(c) + " mIoUT is: "+ str(mIoUT))
        
        '''
        if len(spike.shape) == 4 and spike.shape[2] > 16:
            # distr = []
            # for batch in range(spike.shape[0]):
            #     for channel in range(spike.shape[1]):
            #         channelDensity = np.count_nonzero(spike[batch,channel,:,:].cpu().detach().numpy())
            #         distr.append(channelDensity)
            # import pdb;pdb.set_trace()
            #threshold = np.percentile(distr, 30)                              
            dropped_total_nnz = 0
            dropped_tensors_counter = 0
            tensor_window_size = int(spike.shape[2]/4) #Making sure we are dividing the tensor by 4 along each dimension
            total_nnz = 0
            #import pdb;pdb.set_trace()
            for channel in range(spike.shape[1]):  
                for batch in range(spike.shape[0]):
                    for index_x in range(4):
                        for index_y in range(4):
                            window = spike[batch,channel , index_x*tensor_window_size : (index_x+1) * tensor_window_size, index_y*tensor_window_size : (index_y+1) * tensor_window_size]
                            nnz_of_window = np.count_nonzero(window.cpu().detach().numpy())
                            if nnz_of_window < 0.04*tensor_window_size*tensor_window_size and  nnz_of_window > 0:
                                spike[batch,channel , index_x*tensor_window_size : (index_x+1) * tensor_window_size, index_y*tensor_window_size : (index_y+1) * tensor_window_size]= 0 
                                #Shows how many of tensors have less than 30 events. 
                                #    Note that some of them are completely zero and I should account for the ones that I actually removed
                                dropped_tensors_counter += 1
                                dropped_total_nnz += nnz_of_window
                            total_nnz += nnz_of_window
            with open('./experiments/dropout_16_tensors_finetune/' + 'dropped_total_nnz.csv', 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow([dropped_total_nnz,total_nnz]) # this show how many events we removed
            with open('./experiments/dropout_16_tensors_finetune/' + 'tensor_dropped.csv', 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow([int(dropped_tensors_counter)])
            # with open('./experiments/top_bot_ratio_trainset/' + 'top2bot_ratio.csv', 'a', encoding='UTF8') as f:
            #     writer = csv.writer(f)
            #     writer.writerow([top/(top+bot)])
            '''                        
        return spike
        #return self.spiking()

    def tau(self):
        return 1 / self.w.data.sigmoid().item()

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, tau={self.tau()}'

def create_conv_sequential(in_channels, out_channels, number_layer, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset):
    # 首层是in_channels-out_channels
    # 剩余number_layer - 1层都是out_channels-out_channels
    conv = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
        nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
    ]

    for i in range(number_layer - 1):
        conv.extend([
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
        ])
    return nn.Sequential(*conv)


def create_2fc(channels, h, w, dpp, class_num, init_tau, use_plif, alpha_learnable, detach_reset):
    return nn.Sequential(
        nn.Flatten(),
        layer.Dropout(dpp),
        nn.Linear(channels * h * w, channels * h * w // 4, bias=False),
        PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
        layer.Dropout(dpp, dropout_spikes=True),
        nn.Linear(channels * h * w // 4, class_num * 10, bias=False),
        PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
    )


class StaticNetBase(nn.Module):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset):
        super().__init__()
        self.T = T
        self.init_tau = init_tau
        self.use_plif = use_plif
        self.use_max_pool = use_max_pool
        self.alpha_learnable = alpha_learnable
        self.detach_reset = detach_reset
        self.train_times = 0
        self.max_test_accuracy = 0
        self.epoch = 0
        self.static_conv = None
        self.conv = None
        self.fc = None
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):
        x = self.static_conv(x)
        out_spikes_counter = self.boost(self.fc(self.conv(x)).unsqueeze(1)).squeeze(1)
        for t in range(1, self.T):
            out_spikes_counter += self.boost(self.fc(self.conv(x)).unsqueeze(1)).squeeze(1)

        return out_spikes_counter

class MNISTNet(StaticNetBase):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset):
        super().__init__(T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)

        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.conv = nn.Sequential(
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5, dropout_spikes=use_max_pool),
            nn.Linear(128 * 7 * 7, 128 * 4 * 4, bias=False),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            layer.Dropout(0.5, dropout_spikes=True),
            nn.Linear(128 * 4 * 4, 100, bias=False),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset)
        )

class FashionMNISTNet(MNISTNet):
    pass  # 与MNISTNet的结构完全一致

class Cifar10Net(StaticNetBase):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset):
        super().__init__(T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        self.static_conv = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )

        self.conv = nn.Sequential(
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5, dropout_spikes=use_max_pool),
            nn.Linear(256 * 8 * 8, 128 * 4 * 4, bias=False),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.Linear(128 * 4 * 4, 100, bias=False),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset)
        )
def get_transforms(dataset_name):
    transform_train = None
    transform_test = None
    if dataset_name == 'MNIST':
        transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.11)),
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081),
        ])
    elif dataset_name == 'FashionMNIST':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.2860, 0.3530),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.2860, 0.3530),
        ])
    elif dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transform_train, transform_test

class NeuromorphicNet(nn.Module):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset):
        super().__init__()
        self.T = T
        self.init_tau = init_tau
        self.use_plif = use_plif
        self.use_max_pool = use_max_pool
        self.alpha_learnable = alpha_learnable
        self.detach_reset = detach_reset

        self.train_times = 0
        self.max_test_accuracy = 0
        self.epoch = 0
        self.conv = None
        self.fc = None
        self.boost = nn.AvgPool1d(10, 10)
    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        import pdb;pdb.set_trace()
        out_spikes_counter = self.boost(self.fc(self.conv(x[0])).unsqueeze(1)).squeeze(1)
        for t in range(1, x.shape[0]):
            out_spikes_counter += self.boost(self.fc(self.conv(x[t])).unsqueeze(1)).squeeze(1)
        return out_spikes_counter

class NMNISTNet(NeuromorphicNet):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset, channels, number_layer):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        w = 34
        h = 34  # 原始数据集尺寸
        self.conv = create_conv_sequential(2, channels, number_layer=number_layer, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        self.fc = create_2fc(channels=channels, w=w >> number_layer, h=h >>number_layer, dpp=0.5, class_num=10, init_tau=init_tau, use_plif=use_plif, alpha_learnable=alpha_learnable, detach_reset=detach_reset)


class CIFAR10DVSNet(NeuromorphicNet):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, channels, number_layer, detach_reset):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        w = 128
        h = 128
        self.conv = create_conv_sequential(2, channels, number_layer=number_layer, init_tau=init_tau, use_plif=use_plif,
                                           use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        self.fc = create_2fc(channels=channels, w=w >> number_layer, h=h >> number_layer, dpp=0.5, class_num=10,
                             init_tau=init_tau, use_plif=use_plif, alpha_learnable=alpha_learnable, detach_reset=detach_reset)


class Interpolate(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwargs)

class ASLDVSNet(NeuromorphicNet):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset, channels, number_layer):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        # input size 256 * 256
        w = 256
        h = 256

        self.conv = nn.Sequential(
            Interpolate(size=256, mode='bilinear'),
        )

class DVS128GestureNet(NeuromorphicNet):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset, channels, number_layer):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        w = 128
        h = 128
        self.conv = create_conv_sequential(2, channels, number_layer=number_layer, init_tau=init_tau, use_plif=use_plif,
                                           use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        self.fc = create_2fc(channels=channels, w=w >> number_layer, h=h >> number_layer, dpp=0.5, class_num=11,
                             init_tau=init_tau, use_plif=use_plif, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
