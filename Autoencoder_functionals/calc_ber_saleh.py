import torch.utils.data as D
import numpy as np

from AutoencoderDataset import AutoencoderDataset
from ChannelSaleh import ChannelSaleh


def calc_ber_saleh(autoencoder, amam, ampm):

    autoencoder.eval()
    encoder = autoencoder.encoder
    decoder = autoencoder.decoder
    channel = autoencoder.channel

    M = autoencoder.M

    N = 100000

    test_data = AutoencoderDataset(M, N)
    test_loader = D.DataLoader(test_data, batch_size=N)
    test_features, test_labels = next(iter(test_loader))
    encoded_signal = encoder(test_features)

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

    ber = n_errors / n_test
    print(ber)
    return ber
