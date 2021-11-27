import _paths_ae_functionals

import torch.utils.data as D
import numpy as np

from AutoencoderDataset import AutoencoderDataset
from Transmission import Transmission


def calc_ber_hpa(autoencoder, snr_bler, tx_parameters):

    autoencoder.eval()
    encoder = autoencoder.encoder
    decoder = autoencoder.decoder

    M = autoencoder.M

    N = 100000

    bler = np.zeros(len(snr_bler))

    test_data = AutoencoderDataset(M, N)
    test_loader = D.DataLoader(test_data, batch_size=N)
    test_features, test_labels = next(iter(test_loader))

    encoded_signal = encoder(test_features)

    for i in range(0, len(snr_bler)):

        snr = snr_bler[i]
        tx_parameters.snr = snr
        channel = Transmission(tx_parameters)

        n_errors = 0
        n_test = 0
        while n_errors <= 100:
            transmitted_signal = channel(encoded_signal)
            predicted_signal = decoder(transmitted_signal)

            predicted_signal = predicted_signal.to('cpu')
            predicted_signal = predicted_signal.detach().numpy()
            predicted_output = np.argmax(predicted_signal, axis=1)
            test_labels = test_labels.to('cpu')
            test_targets = test_labels.detach().numpy()

            n_errors += np.sum(predicted_output != test_targets)
            n_test += N

        bler[i] = n_errors / n_test
        # snr[i] = channel.snr

    return bler
