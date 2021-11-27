import torch
import torch.nn as nn
import numpy as np

from fabians_rrcosfilter import fabians_rrcosfilter

from torch.nn.functional import conv1d



torch.backends.cudnn.deterministic = True


f = fabians_rrcosfilter(0.3, 4, 4)
f = torch.from_numpy(f)
f = f.squeeze(1)
filter_length = len(f)

# get same random numbers as matlab example
np.random.seed(712)
signal = np.random.random(100)
signal = torch.from_numpy(signal)

pad = torch.zeros(filter_length - 1)
signal = torch.cat((pad, signal, pad))

f = f.unsqueeze(0)
f = f.unsqueeze(0)
signal = signal.unsqueeze(0)
signal = signal.unsqueeze(0)

y_conv = conv1d(signal, f)

y_conv = y_conv[0, 0, :]
# this is equivalent to matlab conv
y_conv = y_conv.detach().numpy()

# this is equivalent to matlab filter
y_filter = y_conv[0:len(y_conv) - (filter_length - 1)]


a = 1



# conv = nn.Conv1d(1, 1, 17, padding=16)
# with torch.no_grad():
#     conv.weight = torch.nn.Parameter(f)
# y_conv = conv(signal)
