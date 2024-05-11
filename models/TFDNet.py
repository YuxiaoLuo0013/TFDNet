import torch
from torch import nn
from layers.STFT_Enc import Encoder
from layers.RevIN import RevIN


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size = kernel_size, stride = stride, padding = 0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim = 1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride = 1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.channels = configs.enc_in
        self.encoder = nn.ModuleList()
        self.n_fft = configs.n_fft
        self.encoder_list = nn.ModuleList()
        for i in self.n_fft:
            self.encoder_list.append(Encoder(enc_in = configs.enc_in, seq_len = configs.seq_len, n_fft = i,
                                             dropout = configs.dropout, kernel_num = configs.kernel_num,
                                             individual = configs.individual_factor,
                                             mode = configs.mode))
        self.mlp1 = nn.Linear(len(self.n_fft), 1, bias = False)
        self.mlp2 = nn.Linear(len(self.n_fft), 1, bias = False)
        self.dropout = nn.Dropout(p = configs.dropout)
        self.linear1 = nn.Linear(configs.seq_len, configs.pred_len)

        self.decompsition = series_decomp(configs.kernel_size)

        self.revin_layer = RevIN(configs.enc_in, affine = False, subtract_last = False)

    def forward(self, x_enc):
        B, L, variation = x_enc.shape
        x_enc = self.revin_layer(x_enc, 'norm')

        seasonal_init, trend_init = self.decompsition(x_enc)

        trend_init = trend_init.permute(0, 2, 1)
        trend_init = trend_init.reshape(B * variation, L)

        seasonal_init = seasonal_init.permute(0, 2, 1)
        seasonal_init = seasonal_init.reshape(B * variation, L)

        out_seasonal = torch.zeros((B * variation, L, len(self.n_fft))).to(seasonal_init.device)
        out_trend = torch.zeros((B * variation, L, len(self.n_fft))).to(trend_init.device)
        for index, encoder in enumerate(self.encoder_list):
            out_seasonal[:, :, index], out_trend[:, :, index] = encoder(seasonal_init, trend_init)
        if len(self.n_fft) > 1:
            out_seasonal = self.mlp1(out_seasonal).squeeze(dim = -1)
            out_trend = self.mlp2(out_trend).squeeze(dim = -1)
        else:
            out_seasonal = out_seasonal.squeeze(dim = -1)
            out_trend = out_trend.squeeze(dim = -1)

        out = out_seasonal + out_trend
        out = self.linear1(out)
        out = self.dropout(out)
        out = out.reshape(B, variation, self.pred_len)

        out = out.permute(0, 2, 1)
        out = self.revin_layer(out, 'denorm')
        return out
