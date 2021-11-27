import torch


def get_dev():

    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'

    return dev
