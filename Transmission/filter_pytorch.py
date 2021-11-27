from torch import zeros, cat
from torch.nn.functional import conv1d


def filter_pytorch(signal, filter):
    # filter and signal need to be pytorch tensors of size (n, 1)

    if signal.is_cuda:
        dev = 'cuda:0'
    else:
        dev = 'cpu'

    filter = filter.to(dev)

    # signal = signal.squeeze(1)
    filter = filter.squeeze(1)

    filter_length = len(filter)
    pad_length = filter_length - 1
    pad = zeros(pad_length, device=dev)

    signal = cat((pad, signal, pad))

    signal = signal.unsqueeze(0)
    signal = signal.unsqueeze(0)
    filter = filter.unsqueeze(0)
    filter = filter.unsqueeze(0)

    conv = conv1d(signal.float(), filter.float())
    filtered = conv[0, 0, :]
    filtered = filtered[0:len(filtered) - pad_length]

    return filtered
