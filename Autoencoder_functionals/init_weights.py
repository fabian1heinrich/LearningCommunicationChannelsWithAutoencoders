import torch

from numpy import sqrt


def init_weights_n(model):

    torch.manual_seed(0)

    classes = model.__class__.__name__
    if classes.find('Linear') != -1:
        n = model.in_features
        model.weight.data.normal_(0.0, 1/sqrt(n))
        model.bias.data.fill_(0)


def init_weights_u(model):

    torch.manual_seed(0)

    classes = model.__class__.__name__
    if classes.find('Linear') != -1:
        n = model.in_features
        n = 1/sqrt(n)
        model.weight.data.uniform_(-n, n)
        model.bias.data.fill_(0)


# i = torch.randn(2, 2)
# i = i.detach().numpy()
# print(i)

# model = torch.nn.Linear(2, 2)
# model = init_weights_u(model)
# weights = model.weight.data
# print(weights.detach().numpy())
