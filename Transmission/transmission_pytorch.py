import _paths_transmission

from numpy import sqrt, ceil
from torch import stack, normal

from fabians_rrcosfilter import fabians_rrcosfilter
from pulse_shaping_pytorch import pulse_shaping_pytorch
from upsample import upsample
from ChannelSaleh import ChannelSaleh
from ChannelAWGN import ChannelAWGN
import torch


def transmission_pytorch(x, tx_parameters):

    if x.is_cuda:
        dev = 'cuda:0'
    else:
        dev = 'cpu'

    # params
    p = tx_parameters
    snr = p.snr
    input_backoff = p.input_backoff
    amam_params = p.amam_params
    ampm_params = p.ampm_params
    rolloff = p.rolloff
    samples_per_symbol = p.samples_per_symbol
    n_symbols_pulse_shaping = p.n_symbols_pulse_shaping

    # input backoff
    test0 = torch.mean(torch.square(x), dim=0)
    e = 10 ** (input_backoff / 10)
    x = sqrt(e) * x
    test1 = torch.mean(torch.square(x), dim=0)

    # filter design
    rrcos_filter = fabians_rrcosfilter(rolloff, n_symbols_pulse_shaping, samples_per_symbol)
    rrcos_filter = rrcos_filter.to(dev)
    filter_length = len(rrcos_filter)

    # add tail symbols
    n_tail = int(ceil(filter_length / 2))
    tail = x[torch.randint(0, len(x), size=(n_tail,), device=dev)]
    x_w_tail = torch.cat((tail, x, tail))

    # upsampling
    tx_symbols_up = upsample(x_w_tail, samples_per_symbol)

    # pulse shaping
    pad_length = filter_length // 2
    pad = torch.zeros((pad_length, 2), device=dev)
    tx_signal = pulse_shaping_pytorch(torch.cat((tx_symbols_up, pad)), rrcos_filter)
    tx_signal = tx_signal[pad_length:None, :]

    # high power amplifier
    saleh_channel = ChannelSaleh(amam_params, ampm_params)
    tx_signal_hpa = saleh_channel(tx_signal)  # line1
    # @tony
    # comment swap these lines (line1/2) to get tranmssion w/ awgn only and w/o saleh
    # tx_signal_hpa = tx_signal # line2

    # additive noise
    channel = ChannelAWGN(snr)

    # awgn channel ouput
    channel_out = channel(tx_signal_hpa)

    # matched filtering
    norm_factor = 1
    channel_out_mf = pulse_shaping_pytorch(torch.cat((channel_out, pad)), norm_factor * rrcos_filter)
    channel_out_mf = channel_out_mf[pad_length:None, :]

    # sampling
    rx_symbols = channel_out_mf[0:None:samples_per_symbol, :]

    # remove tails
    y = rx_symbols[n_tail:len(rx_symbols)-n_tail, :]
    test2 = torch.mean(torch.square(y), dim=0)

    return y
