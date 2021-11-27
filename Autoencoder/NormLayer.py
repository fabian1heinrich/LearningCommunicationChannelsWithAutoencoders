import torch.nn as nn

from torch import sqrt, mean, square, sum, div, floor, max, ones


class NormLayer(nn.Module):

    def __init__(self):
        super(NormLayer, self).__init__()

    def forward(self, x):
        norm_tensor = sqrt(2 * mean(square(x), dim=0))
        x = x / norm_tensor
        return x


# def forward(self, x):
#         norm_tensor = sqrt(2 * mean(square(x), dim=0))
#         x = x / norm_tensor
#         return x


#  def forward(self, x):
#         norm_tensor = sqrt(sum(square(x), dim=1))
#         x = x / norm_tensor.unsqueeze(1)
#         return x

# def forward(self, x):
#         norm_tensor = sqrt(sum(square(x), dim=1))
#         # norm_tensor = max(norm_tensor, ones(len(x), device='cuda:0'))
#         norm_tensor = max(norm_tensor)
#         x = x / norm_tensor.unsqueeze(0)
#         return x
