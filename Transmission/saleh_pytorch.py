import torch


def saleh_pytorch(b_x, amam_params, ampm_params):

    alpha_a, beta_a = amam_params
    alpha_phi, beta_phi = ampm_params

    u = torch.norm(b_x, p=2, dim=1)
    a = torch.atan2(b_x[:, 0], b_x[:, 1])

    u_y = AMAM(u, alpha_a, beta_a)
    a_y = a + AMPM(u, alpha_phi, beta_phi)

    real = u_y * torch.cos(a_y)
    imag = u_y * torch.sin(a_y)

    b_y = torch.stack((real, imag), dim=1)
    return b_y


def AMAM(u, alpha_a, beta_a):

    A = alpha_a * u / (1 + beta_a * u ** 2)
    return A


def AMPM(u, alpha_phi, beta_phi):

    PHI = alpha_phi * u ** 2 / (1 + beta_phi * u ** 2)
    return PHI
