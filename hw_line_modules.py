import enum
import os
import random
from numpy.core.defchararray import encode, join
import torch
import numpy as np
import cv2
import math
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
import torchvision.models as models
import torchvision.transforms as T
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter
from torchvision.models.vgg import vgg19
from torch.autograd import Variable


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=math.sqrt(5), mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class HWLine_Module(nn.Module):
    def __init__(self, channel_size, hidden_size, output_size, dropout_p=0.1, max_length = 228, D_ch = 16, nWriter = 496):
        super(HWLine_Module, self).__init__()
        self.encoder = CNN(channel_size) # CNN-based feature encoder--F
        self.decoder_forradical = AttnDecoderRNN(hidden_size, output_size, dropout_p, max_length)    # MARK attention-based decoder--A
        
    def forward(self, image, text_radical, length_radical):
        encode = self.encoder(image)    
        b, c, _, _ = encode.size() #batch,256
        loss_forradical, new_encode = self.decoder_forradical(encode,image,text_radical,length_radical)   # 结构保持损失Lstrc | 部首级注意力特征

        return loss_forradical
    
    def predict(self, image, text_radical):
        encode = self.encoder(image)    
        b, c, _, _ = encode.size()
        pred_forradical, attention_maps = self.decoder_forradical.predict(encode,image,text_radical)

        return pred_forradical, attention_maps

class AttnDecoderRNN(nn.Module):
    def __init__(self,hidden_size, output_size, dropout_p,max_length,teach_forcing_prob=0.5):
        super(AttnDecoderRNN,self).__init__()
        self.attention_cell =AttnDecoderRNN_Cell(hidden_size, output_size, dropout_p,max_length)
        self.teach_forcing_prob = teach_forcing_prob
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.hidden_size = hidden_size
        self.pil = T.ToPILImage()
        self.tensor = T.ToTensor()
        self.resize = T.Resize(size = (128,128))
   
    def cal(self,image,alpha_map):
        alpha_map = alpha_map.cpu().detach().numpy().reshape(8,8)
        alpha_map =((alpha_map /alpha_map.max())*255).astype(np.uint8)
        alpha_map[alpha_map>0]=1
        alpha_map = cv2.resize(alpha_map,(image.shape[3],image.shape[2]))
        alpha_map_tensor = torch.from_numpy(alpha_map).expand_as(image[0]).cuda()
        return alpha_map_tensor

    def forward(self,encode,image,text,text_length):
        batch_size = image.shape[0]
        decoder_input = text[:,0]
        decoder_hidden = torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda()        
        
        attention_map_list = []
        loss = 0.0
        teach_forcing = True if random.random() > self.teach_forcing_prob else False
        if teach_forcing:   # teach_foring: 直接使用真实值作为下一时间步的输入
            for di in range(1, text.shape[1]):
                decoder_output, decoder_hidden, decoder_attention = self.attention_cell(decoder_input, decoder_hidden, encode) #decoder_output:torch.Size([4, 472]); decoder_hidden:torch.Size([1, 4, 256])
                attention_map_list.append(decoder_attention)
                loss += self.criterion(decoder_output, text[:,di])
                decoder_input = text[:,di]
        else:   # teach_foring: 使用预测值作为下一时间步的输入
            for di in range(1, text.shape[1]):
                decoder_output, decoder_hidden, decoder_attention = self.attention_cell(decoder_input, decoder_hidden, encode)
                attention_map_list.append(decoder_attention)
                loss += self.criterion(decoder_output, text[:,di])
                topv, topi = decoder_output.data.topk(1)
                ni = topi.squeeze()
                decoder_input = ni
        
        _,c,h,w=encode.shape
        num_labels = text_length.data.sum()
        new_encode = torch.zeros(num_labels,c,h,w).type_as(encode.data)
        start = 0
        for i,length in enumerate(text_length.data):
            #import pdb;pdb.set_trace()               
            attention_maps = attention_map_list[0:length]
            for j,alpha_map in enumerate(attention_maps):
                #import pdb;pdb.set_trace()
                alpha_map_weight = ((alpha_map[i]-alpha_map[i].min())/(alpha_map[i].max()-alpha_map[i].min())).reshape(1,h,w)
                encode_weight = encode[i]*alpha_map_weight
                new_encode[start] = encode_weight
                start +=1
        
        return loss, new_encode
    
    def predict(self, encode, image, text):
        batch_size = image.shape[0]
        decoder_input = text[:,0]
        decoder_hidden = torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda()        
        attention_map_list = []
        
        for di in range(1, text.shape[1]):
            decoder_output, decoder_hidden, decoder_attention = self.attention_cell(decoder_input, decoder_hidden, encode)
            attention_map_list.append(decoder_attention)
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze()
            decoder_input = ni
            text[:,di] = ni

        return text, attention_map_list


class AttnDecoderRNN_Cell(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1,max_length = 64):
        super(AttnDecoderRNN_Cell, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hidden, encoder_outputs): #input:torch.Size([batch]); hidden:torch.Size([1, 4, 256]);encoder_ouput：torch.Size([64, batch, 256])
        bs, c, h, w = encoder_outputs.shape
        T = h*w # 4 * xxx
        encoder_outputs = encoder_outputs.reshape(bs, c, T)
        encoder_outputs = encoder_outputs.permute(2,0,1) #torch.Size([64, batch, 256])

        embedded = self.embedding(input) #torch.Size([batch, 256])
        embedded = self.dropout(embedded) #torch.Size([batch, 256])

        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1) #torch.Size([batch, 64])
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.permute(1, 0, 2)) #torch.Size([batch, 1, 256])

        output = torch.cat((embedded, attn_applied.squeeze(1)), 1) #torch.Size([batch, 512])
        output = self.attn_combine(output).unsqueeze(0) #torch.Size([1,batch, 512])

        output = F.relu(output) #torch.Size([1, batch, 256])
        output, hidden = self.gru(output, hidden)#both:torch.Size([1, batch, 256])

        output = self.out(output[0]) #torch.Size([4, 366])
        return output, hidden, attn_weights


class CNN(nn.Module):
    def __init__(self, channel_size):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(channel_size, 96, 7, stride=1, padding=3),
            nn.BatchNorm2d(96),
            nn.PReLU(),
            nn.MaxPool2d(2,2),  # 8->4

            nn.Conv2d(96, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(3,1,1),

            nn.Conv2d(128, 160, 3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.PReLU(),
            nn.MaxPool2d(3,1,1),

            nn.Conv2d(160, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(3,1,1))

    def forward(self, input):
        conv = self.cnn(input)
        return conv