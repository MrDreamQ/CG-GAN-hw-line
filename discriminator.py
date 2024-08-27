import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn import init
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, dropout, d_model, height=500, width=500) -> None:
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        
        pe = pe.unsqueeze(0)
        super(PositionalEncoding2D, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = d_model


    def forward(self, emb):
        emb = emb * math.sqrt(self.dim)
        emb = emb + self.pe[:, :, :emb.size(2), :emb.size(3)]
        emb = self.dropout(emb)
        return emb

class Discriminator(nn.Module): # Discriminator defined here
    def __init__(self, bnorm=True, leakyparam=0.0, 
                 bias=False,  use_leaky=True, use_bn=True, device='cuda'):
        super(Discriminator, self).__init__()
        
        self.add_position2D = PositionalEncoding2D(dropout=0.1, d_model=256)
        self.dis_block = define_D(input_nc=256*2, ndf=256*2, n_layers_D=2)

    def forward(self, x, condition):
        x = self.add_position2D(x)
        condition = self.add_position2D(condition)
        feat = torch.cat((x, condition), dim=1) # b, 512, 8, 114
        out = self.dis_block(feat)

        return out

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=512, n_layers=2, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        # sequence = [
        #     nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
        #     nn.LeakyReLU(0.2, True)
        # ]
        sequence = []

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

def define_D(input_nc, ndf,
             n_layers_D=2, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, device='cuda'):
    norm_layer = get_norm_layer(norm_type=norm)

    net = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)

    return init_net(net, init_type, init_gain, device)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'none':
        norm_layer = None
    return norm_layer

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, device='cuda'):
    # net.to(device)
    init_weights(net, init_type, gain=init_gain)
    return net


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
            
        return self.loss(input, target_tensor)