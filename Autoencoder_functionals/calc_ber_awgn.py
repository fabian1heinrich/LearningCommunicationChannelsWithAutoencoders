import _paths_ae_functionals

import torch.utils.data as D
import numpy as np

from AutoencoderDataset import AutoencoderDataset
from ChannelAWGN import ChannelAWGN


def calc_ber_awgn(autoencoder, eb_no):

    autoencoder.eval()
    encoder = autoencoder.encoder
    decoder = autoencoder.decoder

    M = autoencoder.M

    N = 100000

    ber = np.zeros(len(eb_no))
    snr = np.zeros(len(eb_no))

    test_data = AutoencoderDataset(M, N)
    test_loader = D.DataLoader(test_data, batch_size=N)
    test_features, test_labels = next(iter(test_loader))

    encoded_signal = encoder(test_features)

    for i in range(0, len(eb_no)):

        cur_ebno = eb_no[i]
        channel = ChannelAWGN(cur_ebno)

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

        ber[i] = n_errors / n_test
        snr[i] = channel.snr

    return ber
