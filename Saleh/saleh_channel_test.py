# this is an implementation for pytorch tensors!
# real and imaginary part separated

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import sqrt, sum, mean, square, stack


def saleh_channel(b_x, amam_params, ampm_params):

    alpha_a, beta_a = amam_params
    alpha_phi, beta_phi = ampm_params

    u = torch.norm(b_x, p=2, dim=1)
    a = torch.atan2(b_x[:, 0], b_x[:, 1])

    u_y = AMAM(u, alpha_a, beta_a)
    a_y = a + AMPM(u, alpha_phi, beta_phi)

    real = u_y * torch.cos(a_y)
    imag = u_y * torch.sin(a_y)

    b_y = stack((real, imag), dim=1)
    return b_y


def AMAM(u, alpha_a, beta_a):

    A = alpha_a * u / (1 + beta_a * u ** 2)
    return A


def AMPM(u, alpha_phi, beta_phi):

    PHI = alpha_phi * u ** 2 / (1 + beta_phi * u ** 2)
    return PHI


# # test
# qam_coeffs = np.array([
#     1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j,
#     1 + 3j, 1 - 3j, -1 + 3j, -1 - 3j,
#     3 + 1j, 3 - 1j, -3 + 1j, -3 - 1j,
#     3 + 3j, 3 - 3j, -3 + 3j, -3 - 3j])

# coeffs = qam_coeffs
# # coeffs = qam_coeffs * np.exp(1j * np.pi/4)
# coeffs = [np.real(coeffs), np.imag(coeffs)]
# coeffs = np.transpose(coeffs)
# coeffs = torch.from_numpy(coeffs)
# coeffs = coeffs / sqrt(sum(2 * mean(square(coeffs))))
# coeffs = coeffs.float()

# coeffs_saleh = saleh_channel(coeffs, [2.1587, 1.1517], [4.0033, 9.1040])

# plt.figure()
# ax = plt.gca()
# ax.scatter(coeffs[:, 0], coeffs[:, 1])
# ax.scatter(coeffs_saleh[:, 0], coeffs_saleh[:, 1])
# ax.set_aspect('equal', adjustable='box')
# ax.grid()
# plt.show()
