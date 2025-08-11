# coding: utf-8

import os, argparse, glob, time, math

import tqdm, skvideo.io, numpy as np
import torch
from torch import matmul
from torch import stack, zeros, ones, manual_seed, load, randn, cat
from torch.autograd import Variable
from torch.nn import Module, Parameter, Sequential, ConvTranspose2d, BatchNorm2d, ReLU, Tanh, BatchNorm1d, Sigmoid
from torch.nn.init import constant_, xavier_uniform_

################################################################################

class MyLinear(Module):
    def __init__(self, input_size, output_size):
        super(MyLinear, self).__init__()
        self.weight = randn((output_size, input_size))
        self.bias = randn(output_size)
        
    def forward(self, inputs):
        return matmul(inputs, self.weight.t()) + self.bias

class MyLSTMCell(Module):
    def __init__(self, input_size, hidden_size):
        super(MyLSTMCell, self).__init__()
        self.weight_ih = randn((4 * hidden_size, input_size))
        self.bias_ih = randn((4 * hidden_size,))
        self.weight_hh = randn((4 * hidden_size, hidden_size))
        self.bias_hh = randn((4 * hidden_size,))
        self.hidden_size = hidden_size

    def forward(self, x, h_in, c_in):
        bias_ih = self.bias_ih.reshape((4 * self.hidden_size, 1))
        bias_hh = self.bias_hh.reshape((4 * self.hidden_size, 1))
        sigmoid = Sigmoid()
        tanh = Tanh()

        i = sigmoid(matmul(self.weight_ih[0:self.hidden_size], x.t()) + bias_ih [0:self.hidden_size] + matmul(self.weight_hh[0:self.hidden_size], h_in.t()) + bias_hh[0:self.hidden_size])
        f = sigmoid(matmul(self.weight_ih[self.hidden_size:self.hidden_size*2], x.t()) + bias_ih [self.hidden_size:self.hidden_size*2] + matmul(self.weight_hh[self.hidden_size:self.hidden_size*2], h_in.t()) + bias_hh[self.hidden_size:self.hidden_size*2])
        g = tanh(matmul(self.weight_ih[self.hidden_size*2:self.hidden_size*3], x.t()) + bias_ih [self.hidden_size*2:self.hidden_size*3] + matmul(self.weight_hh[self.hidden_size*2:self.hidden_size*3], h_in.t()) + bias_hh[self.hidden_size*2:self.hidden_size*3])
        o = sigmoid(matmul(self.weight_ih[self.hidden_size*3:], x.t()) + bias_ih [self.hidden_size*3:] + matmul(self.weight_hh[self.hidden_size*3:], h_in.t()) + bias_hh[self.hidden_size:self.hidden_size*2])
        c_out = f * c_in.t() + i * g
        h_out = o * tanh(c_out)
        
        return h_out.t(), c_out.t()

class LSTM(Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.lstm = MyLSTMCell(input_size, hidden_size)
        self.linear = MyLinear(hidden_size, input_size)
        self.bn = BatchNorm1d(input_size, affine=False)
        self.hidden_size = hidden_size

    def initWeight(self, init_forget_bias=1):
        xavier_uniform_(self.lstm.weight_hh)
        xavier_uniform_(self.lstm.weight_ih)

        self.lstm.bias_hh = zeros(self.lstm.bias_hh.shape)
        self.lstm.bias_hh[self.hidden_size:self.hidden_size*2] = init_forget_bias * ones(self.hidden_size)

        self.lstm.bias_ih = zeros(self.lstm.bias_ih.shape)
        self.lstm.bias_ih[self.hidden_size:self.hidden_size*2] = init_forget_bias * ones(self.hidden_size)

        xavier_uniform_(self.linear.weight)
        self.linear.bias = zeros(self.linear.bias.shape)

    def initHidden(self, batch_size):
        self.h = zeros((batch_size, self.hidden_size))
        self.c = zeros((batch_size, self.hidden_size))

    def forward(self, inputs, n_frames):
        outputs = zeros((n_frames, inputs.shape[0], inputs.shape[1]))
        for frame in range(n_frames):
           self.h, self.c = self.lstm(inputs, self.h, self.c)
           outputs[frame] = self.bn(self.linear(self.h))
        return outputs

################################################################################

if(__name__ == "__main__"):
    img_size = 96
    nc = 3
    ndf = 64
    ngf = 64
    d_E = 10
    hidden_size = 100 # arbitrary
    d_C = 50
    d_M = d_E
    nz  = d_C + d_M

    class Generator_I(Module):
        def __init__(self, nc=3, ngf=64, nz=60, ngpu=1):
            super(Generator_I, self).__init__()
            self.ngpu = ngpu
            self.main = Sequential(
                ConvTranspose2d(     nz, ngf * 8, 6, 1, 0, bias=False),
                BatchNorm2d(ngf * 8),
                ReLU(True),
                ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                BatchNorm2d(ngf * 4),
                ReLU(True),
                ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                BatchNorm2d(ngf * 2),
                ReLU(True),
                ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                BatchNorm2d(ngf),
                ReLU(True),
                ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                Tanh()
            )

        def forward(self, input):
            return self.main(input)

    parser = argparse.ArgumentParser(description='Start using MoCoGAN')
    parser.add_argument('--batch-size', type=int, default=16, help='set batch_size, default: 16')
    parser.add_argument('--niter', type=int, default=16, help='set num of iterations, default: 120000')

    args       = parser.parse_args()
    batch_size = args.batch_size
    n_iter     = args.niter

    seed = 2020
    manual_seed(seed)
    np.random.seed(seed)

    current_path = os.getcwd()

    # Lengths of videos in the dataset "Actions as Space-Time Shapes".
    video_lengths = [84, 85, 55, 51, 54, 70, 146, 56, 105, 103, 67, 67, 63, 45, 43, 72, 47, 40, 39, 38, 62, 54, 42, 65, 48, 127, 49, 55, 45, 56, 42, 41, 76, 36, 93, 56, 52, 56, 36, 64, 67, 53, 52, 48, 39, 64, 64, 60, 39, 46, 43, 57, 48, 60, 38, 39, 63, 92, 85, 37, 77, 68, 84, 68, 101, 43, 88, 61, 119, 112, 50, 111, 120, 82, 60, 125, 55, 103, 61, 53, 54, 60, 61, 81, 51, 54, 67, 114, 79, 89, 57, 62, 59]
    n_videos = len(video_lengths)
    T = 16

    def trim_noise(noise):
        start = np.random.randint(0, noise.size(1) - (T+1))
        end = start + T
        return noise[:, start:end, :, :, :]


    gen_i = Generator_I(nc, ngf, nz).requires_grad_(False)
    lstm = LSTM(d_E, hidden_size).requires_grad_(False)
    lstm.initWeight()

    trained_path = os.path.join(current_path, 'data')

    def save_video(fake_video, index):
        outputdata = fake_video * 255
        outputdata = outputdata.astype(np.uint8)
        dir_path = os.path.join(current_path, 'outputs')
        file_path = os.path.join(dir_path, 'video_%d.mp4' % index)
        skvideo.io.vwrite(file_path, outputdata, outputdict={'-pix_fmt': 'yuv420p'})

    gen_i.load_state_dict(load(os.path.join(trained_path, 'Generator_I-120000.model'), map_location=torch.device('cpu')))
    lstm.load_state_dict(load(os.path.join(trained_path, 'LSTM-120000.model'), map_location=torch.device('cpu')), strict=False)

    ''' generate motion and content latent space vectors '''
    def gen_z(n_frames):
        z_C = Variable(randn(batch_size, d_C))
        z_C = z_C.unsqueeze(1).repeat(1, n_frames, 1)
        eps = Variable(randn(batch_size, d_E))

        lstm.initHidden(batch_size)
        z_M = lstm(eps, n_frames).transpose(1, 0)
        z = cat((z_M, z_C), 2)
        return z.view(batch_size, n_frames, nz, 1, 1)

    for epoch in tqdm.tqdm(range(n_iter)):
        # note that n_frames is sampled from video length distribution
        n_frames = video_lengths[np.random.randint(0, len(video_lengths))]
        Z = gen_z(n_frames)  # Z.size() => (batch_size, n_frames, nz, 1, 1)
        # trim => (batch_size, T, nz, 1, 1)
        Z = trim_noise(Z)
        # generate videos
        Z = Z.contiguous().view(batch_size*T, nz, 1, 1)
        fake_videos = gen_i(Z)
        fake_videos = fake_videos.view(batch_size, T, nc, img_size, img_size)
        fake_videos = fake_videos.transpose(2, 1)
        fake_img = fake_videos[:, :, np.random.randint(0, T), :, :]

    for iii in range(fake_videos.shape[0]):
        save_video(fake_videos[iii].data.cpu().numpy().transpose(1, 2, 3, 0), iii)
